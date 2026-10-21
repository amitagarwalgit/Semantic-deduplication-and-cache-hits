# Semantic Deduplication and Cache Hits

A prompt cache that sits in front of the model and a document deduplicator for ingestion, both built on the same test: are two texts the same request, or the same content, once wording is ignored but facts are not?

The test combines two signals.

- **Delta Score** between the two texts: Jensen-Shannon divergence between softmax-normalised contextual token embeddings with greedy token matching, divided by a frozen baseline measured on unrelated pairs. About 0 means the same text, about 1 means unrelated. This is what catches rewordings.
- **A critical-token check**: numbers and proper nouns in either text must appear literally in the other. Number words are converted to digits first, so "four hundred" matches "400" and "sixty days" does not match "thirty days". Hyphenated names are split, so "Delhi-Mumbai" matches "Delhi to Mumbai". This is what rejects a prompt that shares all its vocabulary with a cached one but differs in the one token that changes the answer.

A finding worth stating plainly: **soft matching alone does not reject entity swaps.** "Delhi to Chennai" still has a per-token support of 0.57 against "Delhi to Mumbai", because contextual embeddings of two city names in the same slot are fairly close. In the demo, Delta Score without the critical check makes exactly the same eight wrong hits as a naive cosine cache. The literal check is not an add-on, it is what makes the cache safe.

## Lookup pipeline

1. Exact match on the normalised string. Free.
2. Prefilter: mean-pooled embedding cosine over all entries, keep the top 10.
3. Delta Score against each candidate; take the closest.
4. Critical-token check in both directions.
5. Hit if the relative divergence is below the threshold (0.45) and the check passes. Every decision carries the per-token support so a miss can be explained.

The deduplicator runs the same test for each incoming document against the canonical copies kept so far and either drops it into an existing cluster or keeps it as a new canonical document.

## Quick start

```bash
pip install -r requirements.txt
python demo.py
```

## Usage

```python
from semantic_cache import DeltaEncoder, SemanticCache, Deduplicator

enc = DeltaEncoder()                                           # roberta-large layer 17, auto device
baseline = SemanticCache.calibrate(enc, sample_prompts)        # once; freeze and reuse

cache = SemanticCache(enc, baseline, hit_threshold=0.45, strict_critical=True)
m = cache.lookup(prompt)
if m.hit:
    return m.response                                          # served without calling the model
response = llm(prompt); cache.add(prompt, response)
# on a miss: m.reason, m.rel, m.mismatched_critical and m.token_support explain why

dd = Deduplicator(enc, baseline, dup_threshold=0.45, strict_critical=True)
for doc in incoming_docs:
    dd.ingest(doc)
index.add(dd.canonical)                                        # one copy per cluster
```

## Demo: prompt cache

Eight cached prompts with answers. Twenty incoming prompts, labelled: eleven should hit (one exact, ten rewordings) and nine must miss (five entity swaps, three number swaps, one unrelated). Four strategies on the same prompts:

| strategy | hits served | wrong answers served | rewordings missed | precision | recall |
|---|---|---|---|---|---|
| exact match | 1 | 0 | 10 | 1.00 | 0.09 |
| mean-pooled cosine, best threshold chosen with hindsight | 19 | **8** | 0 | 0.58 | 1.00 |
| Delta Score soft matching only | 19 | **8** | 0 | 0.58 | 1.00 |
| Delta Score + critical-token check | 10 | **0** | 1 | 1.00 | 0.91 |

The eight wrong answers the cosine cache serves, all at cosine above 0.96, are exactly the dangerous ones: refund for order 4823 answered with order 4821's status, delivery time to Hyderabad answered for Bangalore, the Q3 2023 report answered with Q3 2024 figures, Sunday hours answered with Saturday's. The combined strategy serves none of them. Its one miss is "Does the Team plan have API access?" against the cached "Which plan includes API access?": the incoming prompt introduces a product name the cached prompt never mentioned, and the strict rule declines to assume. That is the cost of safety, and the right default for a cache.

Example of an explained miss:

```
incoming : What is the baggage allowance on a flight from Delhi to Chennai?
nearest  : What is the baggage allowance on a flight from Delhi to Mumbai?   rel 0.07
support  : baggage 1.00  allowance 1.00  flight 1.00  Delhi 0.99  Chennai 0.57
reason   : critical token mismatch: chennai, mumbai
```

## Demo: document deduplication

Ten documents: three near-duplicate groups (a punctuation variant, two full rewordings), one document that repeats a policy with a different number spelled out in words ("sixty days" vs "thirty days"), one with a different rate limit ("1200" vs "600"), and unrelated documents.

Result: 7 canonical documents kept, 3 dropped. All three true duplicates are dropped, including the rewording that says "over 400 vehicles" where the original says "four hundred". Both documents that differ only in a number are kept, which is what you want for policies and specifications. One reworded rate-limit document is kept rather than merged because it omits the acronym "HTTP"; acronyms count as critical tokens, and the rule errs toward keeping.

## Files

- `semantic_cache.py` - encoder, Delta Score, critical-token logic, `SemanticCache`, `Deduplicator`
- `demo.py` - labelled prompt set, four-strategy comparison, dedup run

## Limitations

- Proper-noun detection is a capitalisation heuristic. All-lowercase prompts lose the entity check for names, though numbers are still caught. Plug in a named-entity tagger for production.
- The strict rule misses rewordings that add an entity the cached prompt lacked. Loosen with `strict_critical=False` only where a wrong cached answer is harmless.
- Negation is invisible: "is the API included" and "is the API not included" can hit each other.
- The hit threshold of 0.45 was set on the demo prompts; measure the relative divergence of your own reworded pairs before relying on it.
- Each lookup costs one encoder pass for the prompt plus a Delta comparison per prefiltered candidate, roughly 50 to 150 ms on an Apple M2. For very high traffic, prefilter against a vector index instead of a full scan.
