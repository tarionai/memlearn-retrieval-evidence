# Evaluation Contract — LongMemEval-S Retrieval Benchmark

**Status:** frozen · **Design locked:** 2026-06-04, before the n=200 runs · **Runs executed:** 2026-06-04 (OSAM, cross-encoder, PPR) and 2026-06-05 (BM25, hybrid)

This is the evaluation protocol the benchmark was run under. The design was locked on
2026-06-04 before the n=200 results existed (source: `results/longmemeval_design_decisions.md`,
verbatim frozen copy in this packet). This is **not** a formal external pre-registration, and the
claim is stated precisely: the design document was working-state at lock time and entered git on
2026-06-12 when this packet froze it. What git history independently corroborates is that the
harness implementing this exact protocol (frozen variants, seed=99, pool_k=50, all guards) was
committed together with its findings on 2026-06-04 — and the one post-lock protocol change is
documented below rather than silently absorbed.

## What was being tested

Whether two differentiated retrieval mechanisms — **OSAM** (an associative-memory reranker,
`osam_simplified_hebbian`, α=0.7 frozen pre-run) and **PPR** (HippoRAG-style personalized
PageRank over an entity graph) — add retrieval value over a dense bi-encoder baseline on a public
long-memory benchmark, compared against standard alternatives (BM25 lexical, RRF hybrid
fusion, cross-encoder reranking).

## Frozen protocol

| Parameter | Locked value |
|---|---|
| Dataset | HF `xiaowu0162/longmemeval-cleaned`, config `default`, split `longmemeval_s_cleaned` (500 records) |
| Sample | n=200 uniform over the evaluable set, RNG seed=99 (streaming returns records type-grouped, so uniform sampling across the full set is mandatory) |
| Chunking | Turn-level: one chunk per conversation turn, `chunk_id = f"{session_id}::t{idx}"`, embed turn content |
| Candidate pool | `pool_k = 50` (dense top-50) |
| Embedder | `all-MiniLM-L6-v2` (384-d), L2-distance ascending, `normalize=True` |
| OSAM config | α=0.7 blend, frozen before any benchmark run; `pure_osam` = α=0.0 |
| Metrics | MRR (full pool, lead metric); hit@k for k∈{1,5,10}; recall@k; NDCG@10; help/hurt/unchanged vs dense; oracle@pool_k; OSAM-rerank latency P50/P95; λ-recency buckets |
| Validation rule | A lift is **validated** iff the paired bootstrap 95% CI on per-query (variant − dense) reciprocal-rank deltas has **lower bound strictly > 0** |
| Abstention handling | Abstention questions (no evidence present in the haystack) are excluded before sampling: 500 → 479 evaluable |
| Non-degeneracy guard | Assert evidence ⊊ history per record; exclude and log violations (0 occurred) |
| Dedup | Identical-embedding turns collapsed before pooling (453 collapsed); representative counts as evidence if any duplicate is evidence |
| Chance floor | Per-query deterministic shuffle of the deduped pool — the floor any claimed lift must beat |
| Isolation | Fresh vector store + OSAM state per question; only the embedder is shared |
| Dense-inversion tripwire | If oracle@pool_k is high but dense hit@10 ≈ 0, the L2 ranking direction is inverted — abort (fired: no) |

## Documented post-lock amendment (evidence labeling)

The locked design specified evidence = **all turns of the answer session(s)** (session-gold →
turn expansion). The first turn-level pass under that rule saturated dense MRR to ≈ 1.0 — whole-
session gold makes the task trivially easy and would have manufactured fake headroom. The rule
was replaced with **precise round-level evidence** via the oracle `has_answer` join (100% recovery,
≈1.9 evidence turns per record) before any results were accepted. This is the only change to the
locked protocol, it is recorded in `results/longmemeval_s_findings_20260604.md` (§ Reconciliation),
and it made the benchmark *harder*, not easier, for every variant equally.

## Variant arms

The locked design froze the OSAM arm: `chance · dense_only · dense_plus_osam (α=0.7) ·
pure_osam (α=0.0)`. The remaining arms — `cross_encoder` (2026-06-04), `ppr_only` /
`dense_plus_ppr` (2026-06-04), `lexical_only (BM25)` / `hybrid_rrf` / `hybrid_cross_encoder`
(2026-06-05) — ran under the same harness invariants, on the same n=200 seed=99 subset, against
the same dense baseline. Cross-run tripwires confirm comparability: the later runs reproduce
`dense_only` MRR 0.491, `dense_plus_osam` Δ −0.022, and `cross_encoder` Δ +0.202 exactly.

## Decision rule (set before results)

- `dense_plus_osam` shows no measurable lift over `dense_only` → OSAM is not load-bearing; keep it off the hot path.
- `dense_plus_ppr` shows no Recall@10 lift → further graph-retrieval investment is not justified.
- Negative results are published as negative results — no rehabilitation reruns, no "future work" softening.

## Scope bounds

Retrieval-stage ranking quality only (not answer quality, not end-to-end latency). Single
bi-encoder (`all-MiniLM-L6-v2`), single cross-encoder (`ms-marco-MiniLM-L-6-v2`, CPU). One
execution environment (see `ENVIRONMENT.md`). Every number in this packet is
**measured-on-frozen-benchmark**: LongMemEval-S, n=200, seed=99, under this contract.
