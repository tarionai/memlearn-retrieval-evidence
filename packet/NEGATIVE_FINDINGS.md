# Negative Findings — OSAM and PPR Rejected

**Status:** frozen · All numbers measured-on-frozen-benchmark (LongMemEval-S, n=200, seed=99).

This page exists because the two mechanisms the architecture originally bet on **failed** their
own pre-written ablation gates, and the failure is published rather than buried. The validated
levers in `METRICS.md` (BM25, cross-encoder, hybrid+CE) were found *because* these arms were
run honestly against them.

## Finding 1 — OSAM rerank: rejected

**Claim tested:** an associative-memory reranker (`osam_simplified_hebbian`, α=0.7) adds
retrieval value over a dense bi-encoder.

**Result: no lift — a small but statistically significant degradation.**

| Evidence | Value |
|---|---|
| dense_plus_osam Δ MRR vs dense | **−0.022**, 95% CI **[−0.038, −0.009]** — entirely below zero |
| pure_osam (α=0) MRR | **0.116 ≈ chance 0.130** — the OSAM signal alone is indistinguishable from random ordering |
| Help/hurt per query (blend vs dense) | helped 31 / hurt 46 / unchanged 123 |
| Session-level control | Δ −0.007, CI [−0.025, +0.010] — a wash, not a rescue |

**Mechanism of failure (diagnosed, not hand-waved):** the OSAM readout is a recency-weighted,
query-similarity-weighted sum of past observations — a recency/association prior, not a relevance
judgment. It is largely redundant with dense cosine similarity and adds a recency bias that hurts
when the evidence is an old turn. The recency-bucket analysis confirms it also fails *within* its
forgetting horizon, so tuning the decay rate (λ) would not recover value.

**Scope of the rejection:** this falsifies the *simplified Hebbian variant as a standalone
reranker* on this benchmark. The richer gated-delta variant was not benchmarked here — but its
learned-projection version had already failed both internal gates at n=125, and the redundancy
critique is mechanism-invariant (the readout is still an associative echo, not a relevance
reader). Source: `results/longmemeval_s_findings_20260604.md`, `results/osam_selection_postmortem_20260604.md`.

## Finding 2 — PPR graph retrieval: rejected (earned null)

**Claim tested:** HippoRAG-style personalized PageRank over an entity graph surfaces evidence
the dense retriever misses (the multi-hop recall claim).

**Result: no lift, and the distinct recall claim is directly refuted.**

| Evidence | Value |
|---|---|
| dense_plus_ppr Δ MRR vs dense | **−0.059**, 95% CI **[−0.116, −0.005]** — entirely below zero |
| ppr_only MRR | 0.341 vs dense 0.491 (Δ −0.151, CI [−0.214, −0.089]) |
| Recall@10, dense → dense+PPR | **0.734 → 0.740 — flat.** PPR surfaces essentially zero dense-missed evidence into the top-10, which is its entire recall-recovery claim |
| Reachability | 0.995 — the graph *does* connect query to evidence; PPR just doesn't rank it better. This makes the null **earned**, not "non-evaluable" |
| Multi-hop stratification | The multi-hop question types (multi-session, temporal-reasoning) have *negative* point deltas; the only positive subset is a single-hop type with a CI touching zero. No subset rescues PPR |

**Mechanism of failure:** for ~75% of questions a query entity appears literally in the evidence
turn — dense already retrieves those. Where multi-hop could matter, the spaCy co-occurrence
graph is a hairball (~9,500 nodes / ~467k edges per question) and PPR mass diffuses to
high-connectivity generic turns instead of concentrating on evidence.

**Scope of the rejection (do not over-read):** this is an earned null for *PPR over a spaCy
NER/noun-chunk co-occurrence graph*. It does not falsify HippoRAG with LLM-OpenIE extraction
(untested here, and gated on an extraction-precision check that was never passed). It also says
nothing about non-retrieval uses of a semantic store. The null is exactly as wide as what was run.
Source: `results/ppr_arm_findings_20260604.md`.

## Small-n warning (why n=200 and CIs are non-negotiable)

At n=5, `ppr_only` read **+0.168**. At n=200 it is **−0.151**. The sign flipped with power.
Any of these arms reported at smoke-test scale would have been a confident, wrong, publishable
number.

## Integrity exhibit — the retracted +0.0813

Before this benchmark, an internal evaluation on local corpora produced a **+0.0813 OSAM
"lift"** that was written into buyer-facing material before the validation gate had passed
(how far it traveled externally is not verifiable from the repo — the defect is the claim's
provenance regardless of audience). It was wrong, and it was retracted:

- **What was wrong:** the corpus was degenerate — the warm-up set was identical to the gold
  set, so the "lift" measured memorization, not retrieval. The two controls that catch this
  (a chance floor and identical-embedding dedup) were absent; bootstrap CIs alone did not
  catch it, because CIs quantify noise, not invalid design.
- **What replaced it:** this public-benchmark run, with both controls mandatory. The honest
  result is the opposite sign: OSAM Δ −0.022.
- **Process verdict (from the postmortem):** selecting OSAM to *evaluate* was a legitimate,
  cheaply-gated bet — the architecture's own design doc flagged it as unvalidated and pre-wrote
  the kill-gate that ultimately fired. The mistake was letting a number produced before the gate
  passed leave the building, and tuning parameters on corpora that could not validly measure
  value. The lesson encoded in this packet: **the gate runs first, and negative results ship.**

Full text: `results/osam_selection_postmortem_20260604.md` (frozen verbatim).

## Standing decisions

- OSAM stays off the retrieval hot path. No further OSAM tuning.
- Graph-retrieval investment is not justified under the tested extraction; the unfalsified
  LLM-extraction refinement is explicitly lower-prior than the cross-encoder lever already in hand.
- These are negative findings, not "future work." Nothing in this packet softens them.
