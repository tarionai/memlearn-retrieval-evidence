# Metric Table — LongMemEval-S Retrieval Benchmark

**Status:** frozen · **Every number on this page is measured-on-frozen-benchmark**: LongMemEval-S,
n=200, seed=99, turn-level, precise round-level evidence, under the protocol in `EVAL_CONTRACT.md`,
on the single environment in `ENVIRONMENT.md`. No projections, no extrapolations, no numbers from
any other corpus.

**Validation rule:** a lift is *validated* iff its paired bootstrap 95% CI on per-query
(variant − dense) reciprocal-rank deltas has lower bound strictly > 0.

## Main table — eight variants

MRR, hit@5, Δ, and CI transcribed verbatim from the dashboard's single source of truth
(`src/data/retrievalBenchmark.ts`), which itself transcribes the frozen findings docs.
NDCG@10 transcribed from the frozen result JSONs in `results/`.

| Variant | MRR | hit@5 | NDCG@10 | Δ MRR vs dense | 95% CI (Δ) | Verdict |
|---|---|---|---|---|---|---|
| Chance (floor) | 0.130 | 0.160 | 0.103 | — | — | floor |
| **Dense bi-encoder (baseline)** | **0.491** | 0.690 | 0.508 | 0.000 | — | baseline |
| Dense + OSAM rerank (α=0.7) | 0.469 | 0.660 | 0.486 | **−0.022** | [−0.038, −0.009] | **no lift — rejected** |
| Dense + PPR (graph) | 0.432 | 0.655 | 0.482 | **−0.059** | [−0.116, −0.005] | **no lift — rejected** |
| Lexical (BM25) | 0.638 | 0.750 | 0.615 | +0.148 | [+0.085, +0.206] | validated |
| Hybrid (BM25 + dense, RRF) | 0.588 | 0.765 | 0.598 | +0.097 | [+0.052, +0.142] | validated |
| Cross-encoder rerank | 0.693 | 0.855 | 0.702 | +0.202 | [+0.144, +0.255] | validated |
| **Hybrid + cross-encoder** | **0.715** | **0.880** | **0.734** | **+0.225** | [+0.166, +0.281] | **validated — best config** |

Sources per row: chance/dense/OSAM/cross-encoder from `results/longmemeval_s_retrieval_results_n200.json`;
PPR from `results/ppr_arm_results_n200.json`; BM25/hybrid/hybrid+CE from
`results/longmemeval_s_hybrid_results_n200.json`. The later runs reproduce the canonical
`dense_only` 0.491, `dense_plus_osam` Δ −0.022, and `cross_encoder` Δ +0.202 exactly
(cross-run tripwires), so all eight rows are comparable.

## Reading the table

- **The task is a reranking problem.** Evidence is almost always in the dense top-50
  (oracle@pool_k = 0.955) but mis-ordered. The winning interventions fix ordering.
- **Both differentiated mechanisms fail.** OSAM and PPR are rejected with CIs entirely below
  zero — see `NEGATIVE_FINDINGS.md` for the diagnostics.
- **Best config is a measured win, not max-picking.** The Δ CIs of hybrid+CE and CE-alone
  overlap, so a paired per-query test was run: **hybrid+CE − CE = +0.0224, 95% CI
  [+0.0047, +0.0468]**, excludes 0. The "best config" label is earned by that paired test.
- **BM25 beating dense is an ordering win, not candidate generation:** lexical oracle@pool
  0.940 ≈ dense 0.955. LongMemEval questions quote rare evidence terms verbatim; the 384-d
  bi-encoder mis-orders them.

## Diagnostic variants (mechanism-only arms)

| Variant | MRR | NDCG@10 | Δ MRR vs dense | 95% CI (Δ) | Reading |
|---|---|---|---|---|---|
| pure_osam (α=0.0) | 0.116 | 0.119 | −0.375 | [−0.437, −0.320] | OSAM-only ranking ≈ chance (0.130) — no independent relevance signal |
| ppr_only | 0.341 | 0.372 | −0.151 | [−0.214, −0.089] | PPR-only ranking significantly below dense |

## Session-level control

A controlled session-level run (identical protocol, only the chunk unit differs) confirms the
turn-level reading: dense MRR 0.845; OSAM Δ −0.007, CI [−0.025, +0.010] (wash);
cross-encoder Δ +0.071, CI [+0.029, +0.112] (still significant). OSAM adds no value at either
granularity; the cross-encoder helps at both.
Source: `results/longmemeval_s_retrieval_results_n200_session.json`.

## Latency (scoped — read the label)

| Measurement | P50 | P95 | What it includes |
|---|---|---|---|
| OSAM rerank step (50-candidate pool) | 0.918 ms | 1.489 ms | The `engine.rerank` call only |

Scope: local CPU under the published environment (`ENVIRONMENT.md`); excludes embedding time,
network, and cross-encoder model inference. **Cross-encoder rerank latency was not instrumented
in these runs** — the harness times only the OSAM rerank step (`run_longmemeval_retrieval.py`,
`rerank_ms`). No throughput, cost, or production-latency claims are made anywhere in this packet.

## What is deliberately absent

- No answer-quality / end-to-end QA metrics — retrieval-stage ranking only.
- No cross-corpus generalization claims — one benchmark, one sample.
- No competitive comparisons against other memory systems.
- No "X% improvement" marketing arithmetic — the Δs and CIs above are the claim, in full.
