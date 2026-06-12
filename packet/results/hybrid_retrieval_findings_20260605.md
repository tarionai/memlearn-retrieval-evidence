# Hybrid Retrieval (BM25 + dense, RRF) on LongMemEval-S — WP-110 Phase B

**Date:** 2026-06-05 · LongMemEval-S, turn-level, n=200, bench_seed=99 (same subset as the OSAM/CE run).
**Harness:** `src/validation_mvp/run_longmemeval_retrieval.py --n 200 --hybrid` (extended in place; reuses
per-question isolation, identical-embedding dedup, chance baseline, oracle, paired bootstrap CI).
**Result:** `state/intermediate/longmemeval_s_hybrid_results_n200.json` · log `_hybrid_n200_run.log`.
**Lexical backend:** `bm25s` (Postgres not reachable locally → swappable backend; pattern identical for
Postgres FTS / Elasticsearch / OpenSearch). **Fusion:** Reciprocal Rank Fusion, rrf_k=60.

## Non-invasiveness tripwires (BOTH reproduce the canonical run exactly)

- `dense_only` MRR **0.4909** (canonical 0.491 ✓) · `dense_plus_osam` Δ **−0.0223** (canonical −0.022 ✓)
- `cross_encoder` Δ **+0.2021** (canonical +0.202 ✓) — and the CE was refactored to use the docs→content
  map BM25 also uses, so its exact reproduction validates the BM25 corpus content too.
- oracle@pool **0.955**, dense-inversion tripwire **False**, collapsed **453** — all match canonical.

## Results (n=200, seed=99, paired bootstrap 95% CI; validated ⇔ CI lower > 0)

| variant | MRR | hit@5 | recall@5 | Δ vs dense | CI95(Δ) | verdict |
|---|---|---|---|---|---|---|
| chance | 0.130 | — | — | — | — | floor |
| dense_only | 0.491 | 0.690 | 0.534 | 0.000 | — | baseline |
| dense_plus_osam | 0.469 | 0.660 | 0.507 | −0.022 | [−0.038, −0.009] | no lift |
| dense_plus_ppr (prior) | 0.432 | 0.655 | — | −0.059 | [−0.116, −0.005] | no lift |
| **lexical_only (BM25)** | **0.638** | 0.750 | 0.633 | **+0.148** | **[+0.085, +0.206]** | **validated** |
| **hybrid_rrf** | **0.588** | 0.765 | 0.627 | **+0.097** | **[+0.052, +0.142]** | **validated** |
| cross_encoder | 0.693 | 0.855 | 0.749 | +0.202 | [+0.144, +0.255] | validated |
| **hybrid_cross_encoder** | **0.715** | **0.880** | 0.783 | **+0.225** | **[+0.166, +0.281]** | **validated — best config** |

## Honest interpretation (the decomposition matters)

1. **BM25 lexical beats dense (+0.148, CI > 0).** Mechanism (spot-checked, `_hybrid_spotcheck.py`):
   LongMemEval questions quote rare entity/temporal terms verbatim from the evidence turn
   ("last name / changed", "clothing / discount / purchase", "painting / worth / paid"); the weak 384-d
   all-MiniLM-L6-v2 bi-encoder mis-orders them. BM25 top-1 was the true evidence turn in all spot-checked
   cases. **lexical oracle@pool 0.940 ≈ dense 0.955** → this is a pure *ordering* win, not a candidate-
   generation difference. A known-shape result, not a miracle.
2. **Fusion (RRF) is positive but BELOW lexical-alone** (hybrid 0.588 < lexical 0.638). Mixing the weaker
   dense signal *dilutes* the stronger lexical one — the same dilution that hurt the PPR fusion arm, here
   in reverse. So "hybrid wins" is NOT the headline; lexical-alone edges it.
3. **Cross-encoder remains the strongest single reranker (+0.202).** Chaining it onto the fused pool gives
   the **best overall configuration: hybrid+cross-encoder +0.225, hit@5 0.880.**
4. **OSAM and PPR remain falsified** (credibility core intact). The win is an honest keyword+semantic+
   rerank pipeline, not the architecture's pet associative/graph mechanisms.
5. **"Best config" is a MEASURED superiority, not max-picking.** The Δ-vs-dense CIs of hybrid+CE
   ([+0.166,+0.281]) and CE-alone ([+0.144,+0.255]) overlap, so a paired per-query test was run
   (`_paired_hybridce_vs_ce.py`): **paired Δ (hybrid+CE − CE) = +0.0224, CI95 [+0.0047, +0.0468]**,
   excludes 0 → hybrid+CE is significantly better than CE alone. The "best config" crown is earned, not
   the largest point estimate among ties.

## Scope / bounds

LongMemEval-S, n=200, seed=99, turn-level, retrieval-stage MRR/recall/NDCG only (no answer quality).
Single bi-encoder all-MiniLM-L6-v2 (384-d); cross-encoder ms-marco-MiniLM-L-6-v2. Lexical = bm25s
(swappable). Foundry-scope measurement; no end-user/runtime promotion.
