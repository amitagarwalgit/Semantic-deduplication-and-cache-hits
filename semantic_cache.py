"""Semantic prompt cache and document deduplication with Delta Score.

Lookup pipeline for an incoming prompt:
  1. exact match on the normalised string                       -> HIT (free)
  2. prefilter: mean-pooled embedding cosine over all entries   -> top-N candidates (cheap)
  3. for each candidate, Delta Score between the two texts
       rel = DeltaF / baseline   (baseline = mean DeltaF of unrelated pairs, frozen at calibration)
       ~0 same text, ~1 unrelated
  4. critical-token check, BOTH directions: numbers and proper nouns in either prompt must appear
     literally in the other.  Soft token matching cannot tell "Mumbai" from "Chennai" or 4821 from
     4823 reliably, and those are exactly the differences that make a cached answer wrong.
  5. HIT when rel < hit_threshold and the critical-token check passes.

Every decision returns the per-token support of the incoming prompt against the chosen candidate
so a miss can be explained ("'Chennai' has no match").

The same `near_duplicate` test drives `Deduplicator`, which clusters documents at ingestion and
keeps one canonical copy per cluster.
"""
from __future__ import annotations
import re, math, random, string, time, unicodedata
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer, AutoModel

STOPWORDS = set("""a an the and or but if of at by for with about as into like through after over
between out against during without before under around among to from in on is are was were be been
being am do does did doing have has had having it its this that these those i you he she we they me
him her us them my your his our their what which who whom whose where when why how all any both each
few more most other some such no nor not only own same so than too very can will just should now
would could may might must shall there here then also please help need want""".split())


def _entropy2(p): return -(p * torch.log2(p.clamp_min(1e-300))).sum(-1)

def js_matrix(Pa, Pb):
    M = 0.5 * (Pa[:, None, :] + Pb[None, :, :])
    return _entropy2(M) - 0.5 * (_entropy2(Pa)[:, None] + _entropy2(Pb)[None, :])


class DeltaEncoder:
    def __init__(self, model_name="roberta-large", layer=17, device=None, temperature=1.0, max_len=512):
        self.model_name, self.layer, self.T, self.max_len = model_name, layer, temperature, max_len
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32).eval().to(self.device)
        self.special = {self.tok.cls_token_id, self.tok.sep_token_id, self.tok.bos_token_id,
                        self.tok.eos_token_id, self.tok.pad_token_id} - {None}
        self.cache: dict[str, "Encoded"] = {}

    def _content(self, tid, ttext):
        if tid in self.special: return False
        w = ttext.replace("Ġ", "").strip().lower()
        return bool(w) and not all(c in string.punctuation for c in w) and w not in STOPWORDS

    @torch.no_grad()
    def encode(self, text: str) -> "Encoded":
        if text in self.cache: return self.cache[text]
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len).to(self.device)
        H = self.model(**enc, output_hidden_states=True).hidden_states[self.layer][0].float().cpu().double()
        ids = enc["input_ids"][0].cpu().tolist(); toks = self.tok.convert_ids_to_tokens(ids)
        keep = [self._content(i, t) for i, t in zip(ids, toks)]
        if sum(keep) == 0: keep = [i not in self.special for i in ids]
        H = H[torch.tensor(keep)]; H = H / H.norm(dim=-1, keepdim=True)
        pooled = H.mean(0); pooled = pooled / pooled.norm()
        e = Encoded(text, [t.replace("Ġ", "") for t, k in zip(toks, keep) if k], H, torch.softmax(H / self.T, -1), pooled)
        self.cache[text] = e; return e


@dataclass
class Encoded:
    text: str
    tokens: list[str]
    H: torch.Tensor        # unit-norm content-token embeddings
    P: torch.Tensor        # softmax distributions (Delta Score)
    pooled: torch.Tensor   # mean-pooled unit vector (prefilter / naive baseline)


def delta_f(a: Encoded, b: Encoded) -> tuple[float, torch.Tensor]:
    D = js_matrix(a.P, b.P)
    P = D.min(dim=1).values.mean(); R = D.min(dim=0).values.mean()
    return (float(2 * P * R / (P + R)) if float(P + R) > 0 else 0.0), D


# ---------- critical tokens: numbers and proper nouns must match literally ----------
_NUM = re.compile(r"\d[\d,./:-]*")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                       "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}
_NUMWORDS = set(_UNITS) | set(_TENS) | set(_SCALES)

def number_words_to_digits(text: str) -> set[str]:
    """'four hundred' -> '400', 'twenty-six' -> '26', 'thirty' -> '30'. Returns the set of numbers found."""
    words = re.findall(r"[a-z]+", text.lower().replace("-", " ")); out = set(); total = cur = 0; active = False
    def flush():
        nonlocal total, cur, active
        if active: out.add(str(total + cur))
        total = cur = 0; active = False
    for w in words:
        if w in _UNITS: cur += _UNITS[w]; active = True
        elif w in _TENS: cur += _TENS[w]; active = True
        elif w in _SCALES:
            if not active: continue
            if _SCALES[w] == 100: cur *= 100
            else: total += cur * _SCALES[w]; cur = 0
        elif w == "and" and active: continue
        else: flush()
    flush(); return out

def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t))

def critical_tokens(text: str) -> set[str]:
    nums = {m.replace(",", "").rstrip(".") for m in _NUM.findall(text)} | number_words_to_digits(text)
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    props = set()
    for s in sents:
        words = _WORD.findall(s)
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and w.lower() not in STOPWORDS and w.lower() not in _NUMWORDS:
                props.update(part for part in w.lower().split("-") if len(part) > 1)   # 'Delhi-Mumbai' -> delhi, mumbai; drop 'Q' of 'Q3'
    return nums | props

def _number_set(text: str) -> set[str]:
    return {m.replace(",", "").rstrip(".") for m in _NUM.findall(text)} | number_words_to_digits(text)

def critical_mismatch(a: str, b: str) -> list[str]:
    """Critical tokens of either text that the other does not contain (numbers compared after
    converting number words to digits, proper nouns compared on the normalised string)."""
    na, nb = normalise(a), normalise(b); numa, numb = _number_set(a), _number_set(b)
    def missing(src, other_norm, other_nums):
        out = set()
        for t in critical_tokens(src):
            if t.isdigit() or _NUM.fullmatch(t):
                if t not in other_nums: out.add(t)
            elif f" {t} " not in f" {other_norm} ": out.add(t)
        return out
    return sorted(missing(a, nb, numb) | missing(b, na, numa))


@dataclass
class Match:
    hit: bool
    reason: str
    rel: float                      # relative Delta divergence to the best candidate (nan if none)
    candidate: str | None
    response: str | None
    mismatched_critical: list[str]
    token_support: list[tuple[str, float]]
    latency_ms: float


class SemanticCache:
    def __init__(self, encoder: DeltaEncoder, baseline: float, hit_threshold: float = 0.45,
                 strict_critical: bool = True, prefilter_k: int = 10):
        self.enc, self.baseline, self.hit_threshold = encoder, baseline, hit_threshold
        self.strict_critical, self.prefilter_k = strict_critical, prefilter_k
        self.entries: list[tuple[str, str, Encoded]] = []     # (prompt, response, encoded)
        self.exact: dict[str, int] = {}

    @staticmethod
    def calibrate(encoder: DeltaEncoder, pool: list[str], n_pairs: int = 60, seed: int = 0) -> float:
        rng = random.Random(seed); vals = []
        while len(vals) < n_pairs:
            a, b = rng.choice(pool), rng.choice(pool)
            if a != b: vals.append(delta_f(encoder.encode(a), encoder.encode(b))[0])
        return sum(vals) / len(vals)

    def add(self, prompt: str, response: str):
        self.exact[normalise(prompt)] = len(self.entries)
        self.entries.append((prompt, response, self.enc.encode(prompt)))

    def _support(self, D: torch.Tensor, q: Encoded) -> list[tuple[str, float]]:
        d_random = D.mean().item() or 1.0
        s = (1 - D.min(dim=1).values / d_random).clamp(0, 1)
        return [(t, round(float(v), 2)) for t, v in zip(q.tokens, s)]

    def lookup(self, prompt: str) -> Match:
        t0 = time.time(); key = normalise(prompt)
        if key in self.exact:
            p, r, _ = self.entries[self.exact[key]]
            return Match(True, "exact", 0.0, p, r, [], [], (time.time() - t0) * 1000)
        if not self.entries:
            return Match(False, "empty cache", float("nan"), None, None, [], [], (time.time() - t0) * 1000)
        q = self.enc.encode(prompt)
        pooled = torch.stack([e.pooled for _, _, e in self.entries])
        cand = torch.topk(pooled @ q.pooled, k=min(self.prefilter_k, len(self.entries))).indices.tolist()
        best = None
        for i in cand:
            p, r, e = self.entries[i]; f, D = delta_f(q, e); rel = f / self.baseline
            if best is None or rel < best[0]: best = (rel, p, r, D)
        rel, p, r, D = best; sup = self._support(D, q)
        mism = critical_mismatch(prompt, p) if self.strict_critical else []
        if rel >= self.hit_threshold:
            return Match(False, f"rel {rel:.2f} >= {self.hit_threshold}", rel, p, None, mism, sup, (time.time() - t0) * 1000)
        if mism:
            return Match(False, f"critical token mismatch: {', '.join(mism)}", rel, p, None, mism, sup, (time.time() - t0) * 1000)
        return Match(True, f"semantic (rel {rel:.2f})", rel, p, r, [], sup, (time.time() - t0) * 1000)

    def near_duplicate(self, a: str, b: str) -> tuple[bool, float, list[str]]:
        f, _ = delta_f(self.enc.encode(a), self.enc.encode(b)); rel = f / self.baseline
        mism = critical_mismatch(a, b) if self.strict_critical else []
        return (rel < self.hit_threshold and not mism), rel, mism


class Deduplicator:
    """Greedy single-pass clustering at ingestion: each new document is compared with the canonical
    copies kept so far (after a pooled-cosine prefilter) and either joins a cluster or starts one."""
    def __init__(self, encoder: DeltaEncoder, baseline: float, dup_threshold: float = 0.45,
                 strict_critical: bool = False, prefilter_k: int = 20):
        self.cache = SemanticCache(encoder, baseline, dup_threshold, strict_critical, prefilter_k)
        self.enc = encoder; self.canonical: list[str] = []; self.clusters: dict[int, list[str]] = {}
        self.log: list[dict] = []

    def ingest(self, doc: str) -> int:
        q = self.enc.encode(doc)
        if self.canonical:
            pooled = torch.stack([self.enc.encode(c).pooled for c in self.canonical])
            cand = torch.topk(pooled @ q.pooled, k=min(self.cache.prefilter_k, len(self.canonical))).indices.tolist()
            best = None
            for i in cand:
                dup, rel, mism = self.cache.near_duplicate(doc, self.canonical[i])
                if best is None or rel < best[1]: best = (i, rel, dup, mism)
            i, rel, dup, mism = best
            if dup:
                self.clusters[i].append(doc); self.log.append({"doc": doc[:60], "action": "dropped", "dup_of": i, "rel": round(rel, 3)}); return i
            self.log.append({"doc": doc[:60], "action": "kept", "nearest": i, "rel": round(rel, 3), "critical_mismatch": mism})
        else:
            self.log.append({"doc": doc[:60], "action": "kept", "nearest": None, "rel": None, "critical_mismatch": []})
        self.canonical.append(doc); self.clusters[len(self.canonical) - 1] = [doc]
        return len(self.canonical) - 1
