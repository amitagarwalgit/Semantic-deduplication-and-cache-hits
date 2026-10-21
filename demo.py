"""Prompt cache: three strategies on a labelled set.  Document dedup at ingestion."""
import time, torch
from semantic_cache import DeltaEncoder, SemanticCache, Deduplicator, delta_f, critical_mismatch, normalise

CACHED = [
    ("How do I reset my password?", "Go to Settings > Security > Reset password and follow the email link."),
    ("What is the baggage allowance on a flight from Delhi to Mumbai?", "Economy fares include 15 kg checked and 7 kg cabin baggage."),
    ("Can I get a refund for order 4821?", "Order 4821 is eligible for a full refund within 30 days."),
    ("What are your opening hours on Saturday?", "Saturday: 9am to 5pm."),
    ("How long does delivery to Bangalore take?", "Standard delivery to Bangalore takes 2 to 3 business days."),
    ("Summarise the Q3 2024 revenue report", "Q3 2024 revenue was $4.2B, up 12% year over year."),
    ("Translate 'good morning' into French", "Bonjour."),
    ("Which plan includes API access?", "API access is included in the Team and Enterprise plans."),
]
# (incoming prompt, should_hit, note)
INCOMING = [
    ("How do I reset my password?",                                   True,  "exact"),
    ("password reset - how?",                                          True,  "reworded"),
    ("I forgot my password, how can I change it",                      True,  "reworded"),
    ("what's the baggage allowance Delhi to Mumbai",                   True,  "reworded"),
    ("How much luggage can I take on the Delhi-Mumbai flight?",        True,  "reworded"),
    ("What is the baggage allowance on a flight from Delhi to Chennai?", False, "entity swap (city)"),
    ("Can I get a refund for order 4823?",                             False, "number swap (order id)"),
    ("refund possible on order 4821?",                                 True,  "reworded"),
    ("What are your opening hours on Sunday?",                         False, "entity swap (day)"),
    ("when are you open on saturday",                                  True,  "reworded"),
    ("How long does delivery to Hyderabad take?",                      False, "entity swap (city)"),
    ("delivery time to Bangalore?",                                    True,  "reworded"),
    ("Summarise the Q3 2023 revenue report",                           False, "number swap (year)"),
    ("Summarise the Q2 2024 revenue report",                           False, "number swap (quarter)"),
    ("give me a summary of the revenue report for Q3 2024",            True,  "reworded"),
    ("Translate 'good morning' into German",                           False, "entity swap (language)"),
    ("how do you say good morning in French",                          True,  "reworded"),
    ("Which plan includes SSO?",                                       False, "different feature, same template"),
    ("Does the Team plan have API access?",                            True,  "reworded"),
    ("What is the capital of Australia?",                              False, "unrelated"),
]

def evaluate(name, decisions):
    tp = sum(h and g for h, g in decisions); fp = sum(h and not g for h, g in decisions)
    fn = sum((not h) and g for h, g in decisions); tn = sum((not h) and (not g) for h, g in decisions)
    prec = tp / (tp + fp) if tp + fp else 1.0; rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"  {name:38s} hits={tp+fp:2d}  correct hits={tp:2d}  WRONG hits={fp:2d}  missed rewordings={fn:2d}  precision={prec:.2f} recall={rec:.2f}")
    return fp, fn

if __name__ == "__main__":
    t0 = time.time(); enc = DeltaEncoder()
    pool = [p for p, _ in CACHED] + [p for p, _, _ in INCOMING]
    base = SemanticCache.calibrate(enc, pool, n_pairs=80)
    print(f"calibration baseline (unrelated prompt pairs) DeltaF = {base:.3e}\n")

    cache = SemanticCache(enc, base, hit_threshold=0.45, strict_critical=True)
    for p, r in CACHED: cache.add(p, r)

    # ---------- 1. Delta Score + critical-token check ----------
    print("=== 1. Delta Score cache with critical-token check ===")
    print(f"{'incoming prompt':62s} {'want':5s} {'got':5s} {'rel':>5s}  reason")
    delta_dec = []
    for p, want, note in INCOMING:
        m = cache.lookup(p); ok = "ok" if m.hit == want else "XX"
        delta_dec.append((m.hit, want))
        print(f"{p[:60]:62s} {'HIT' if want else 'MISS':5s} {'HIT' if m.hit else 'MISS':5s} {m.rel:5.2f}  {m.reason}  [{note}] {ok if ok=='XX' else ''}")
    print()

    # ---------- 2. the same decisions with the critical check OFF (pure soft matching) ----------
    soft = SemanticCache(enc, base, hit_threshold=0.45, strict_critical=False)
    for p, r in CACHED: soft.add(p, r)
    soft_dec = [(soft.lookup(p).hit, want) for p, want, _ in INCOMING]

    # ---------- 3. naive mean-pooled cosine cache, best threshold chosen with hindsight ----------
    pooled = torch.stack([enc.encode(p).pooled for p, _ in CACHED])
    sims = [float((pooled @ enc.encode(p).pooled).max()) for p, _, _ in INCOMING]
    best = None
    for thr in sorted(set(sims)):
        dec = [(s >= thr, want) for s, (_, want, _) in zip(sims, INCOMING)]
        err = sum(h != g for h, g in dec)
        if best is None or err < best[0]: best = (err, thr, dec)
    cos_dec = best[2]
    # ---------- 4. exact match ----------
    exact_keys = {normalise(p) for p, _ in CACHED}
    exact_dec = [(normalise(p) in exact_keys, want) for p, want, _ in INCOMING]

    print("=== Strategy comparison on the 20 labelled prompts (10 should hit, 10 must miss) ===")
    evaluate("exact match", exact_dec)
    evaluate(f"mean-pooled cosine (best thr={best[1]:.3f}, hindsight)", cos_dec)
    evaluate("Delta soft matching only", soft_dec)
    evaluate("Delta + critical-token check", delta_dec)

    # which prompts does the cosine cache get wrong?
    print("\n  cosine cache wrong decisions:")
    for (p, want, note), (h, _), s in zip(INCOMING, cos_dec, sims):
        if h != want: print(f"    {'WRONG HIT ' if h else 'MISSED    '} cos={s:.3f}  {p[:55]}  [{note}]")

    # ---------- 5. explainability: why did the Chennai prompt miss? ----------
    m = cache.lookup("What is the baggage allowance on a flight from Delhi to Chennai?")
    print(f"\n=== Why the Chennai prompt misses (nearest: '{m.candidate}') ===")
    print("  per-token support:", " ".join(f"{t}:{v:.2f}" for t, v in m.token_support))
    print("  critical mismatch:", m.mismatched_critical, "| rel", round(m.rel, 2))

    # ---------- 6. document dedup at ingestion ----------
    DOCS = [
        "Northwind Logistics was founded in 1998 as a two-truck courier service in Leeds and now operates more than four hundred vehicles.",
        "Northwind Logistics was founded in 1998 as a two-truck courier service in Leeds, and today it operates more than four hundred vehicles.",   # near-dup (punctuation)
        "Founded in Leeds in 1998 with just two trucks, Northwind Logistics has grown into an operator of over 400 vehicles.",                       # reworded dup
        "All expense claims must be submitted through the finance portal within thirty days of purchase, with itemised receipts for meals.",
        "Expense claims go through the finance portal and must be filed within thirty days of the purchase; meals need itemised receipts.",         # reworded dup
        "All expense claims must be submitted through the finance portal within sixty days of purchase, with itemised receipts for meals.",         # same template, different policy number
        "The API rate limit is 600 requests per minute per key; exceeding it returns HTTP 429 with a Retry-After header.",
        "Rate limiting: each API key may make 600 requests a minute. Requests beyond that receive a 429 response and a Retry-After header.",        # reworded dup
        "The API rate limit is 1200 requests per minute per key; exceeding it returns HTTP 429 with a Retry-After header.",                         # different number
        "Parental leave is available after twenty-six weeks of service; primary carers receive twenty weeks at full pay.",
    ]
    print("\n=== 7. Document dedup at ingestion (strict_critical=True so a changed number is NOT a duplicate) ===")
    dd = Deduplicator(enc, base, dup_threshold=0.45, strict_critical=True)
    for d in DOCS: dd.ingest(d)
    for e in dd.log:
        extra = f"dup_of=#{e['dup_of']}" if e["action"] == "dropped" else (f"nearest=#{e['nearest']} rel={e['rel']}" + (f" critical_mismatch={e['critical_mismatch']}" if e.get('critical_mismatch') else "") if e["nearest"] is not None else "")
        print(f"  {e['action']:8s} {e['doc']:62s} {extra}")
    print(f"\n  {len(DOCS)} docs in -> {len(dd.canonical)} canonical kept, {len(DOCS)-len(dd.canonical)} dropped as near-duplicates")
    print(f"\n{time.time()-t0:.0f}s total")
