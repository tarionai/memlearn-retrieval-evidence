# LongMemEval-S Retrieval-Stage Findings — Public-Benchmark Verdict

**Date:** 2026-06-04 · **WP-110 retrieval-stage slice** · mechanism `osam_simplified_hebbian` (α=0.7 frozen)
**Result files:** `longmemeval_s_retrieval_results.json` / `.md` (turn-level, n=200, bench_seed=99);
session-level control `…_n200_session.json` / `.md`; smoke `…_n50.*`.
**Harness:** `src/validation_mvp/run_longmemeval_retrieval.py` · **Adapter:** `src/validation_mvp/bench_datasets/longmemeval_adapter.py`

---

## Headline

On the gold-standard external benchmark (LongMemEval-S, n=200, **precise round-level evidence**,
all prior-session controls applied), legacy OSAM provides **no additive retrieval value** — at turn
granularity a small but **statistically-significant degradation** of dense ranking; at session
granularity a wash (confirmed by a controlled session-level run, see Reconciliation). The real,
significant lever is an **off-the-shelf cross-encoder reranker** (+0.20 MRR at turn level, +0.07 at
session level — significant at both). The task is a **reranking problem** (evidence is in the pool,
oracle@pool 0.955; dense mis-orders it) and OSAM does not improve the ordering at any granularity tested.

| variant | MRR | hit@5 | Δ vs dense | CI95(Δ) | verdict |
|---|---|---|---|---|---|
| chance | 0.130 | 0.160 | — | — | floor |
| **dense_only** | **0.491** | 0.690 | 0.000 | — | baseline |
| dense_plus_osam (α=0.7) | 0.469 | 0.660 | **−0.022** | **[−0.038, −0.009]** | **significantly hurts (small)** |
| pure_osam (α=0.0) | 0.116 | 0.145 | −0.375 | [−0.437, −0.320] | **≈ chance — no signal** |
| **cross_encoder** | **0.693** | 0.855 | **+0.202** | **[+0.144, +0.255]** | **significantly helps (large)** |

Guards: oracle@pool_k **0.955**, dense-inverted tripwire **False**, non-degeneracy exclusions **0**,
duplicate-embeddings collapsed **453**, rerank latency P50 0.92 ms / P95 1.49 ms.

---

## Reconciliation with the prior "definitive null" — isolated by a controlled run

A prior harness (`longmemeval_retrieval_eval.py`, n=50) reported *dense MRR 0.653, OSAM Δ −0.0003,
cross-encoder Δ +0.004 — "both wash, no headroom for any reranker."* It does **session-level** retrieval
(one doc per session, gold = answer session). I first hypothesised the cross-encoder difference was a
**granularity** effect. To test that without confounds I ran **my own harness in session-level mode**
(identical pool_k=50, n=200, seed=99, full-pool MRR, abstention handling — *only* the chunk unit differs):

| metric | **my session-level** | **my turn-level** | prior harness (session, n=50) |
|---|---|---|---|
| dense MRR | 0.845 | 0.491 | 0.653 |
| OSAM Δ vs dense | −0.007 · CI [−0.025, +0.010] (**wash**) | −0.022 · CI [−0.038, −0.009] (**sig. hurt**) | −0.0003 (wash) |
| cross-encoder Δ vs dense | **+0.071 · CI [+0.029, +0.112] (sig.)** | **+0.202 · CI [+0.144, +0.255] (sig.)** | +0.004 (wash) |
| pure-OSAM MRR (chance) | 0.625 (0.140) | 0.116 (0.130) | — |

**This refutes my granularity hypothesis.** In a consistent harness the cross-encoder helps
**significantly at *both* granularities** (+0.07 session, +0.20 turn) — granularity is *not* what made the
prior cross-encoder read as a wash.

Honest reconciliation:
- **OSAM half of the prior null STANDS and is corroborated** — OSAM adds no value at either granularity
  (session wash, turn slight-hurt). Robust.
- **Cross-encoder half — the prior number reproduces; the *interpretation* does not.** The prior +0.004 is a
  valid (deterministic) measurement, but its CI **[−0.10, +0.11] contains my +0.071** — the two results are
  not in statistical conflict. The prior run was **underpowered (n=50)** and **config-handicapped** (MRR@5
  vs full-pool MRR; pool_k=20 vs 50; a 2000-char session truncation) and so could not *resolve* a lift a
  powered run detects. "No reranking headroom" was never established by that data. The session-level CE
  lift (+0.071) is itself a **lower bound** — `ms-marco-MiniLM` truncates ~512 tokens, so it reranks on
  partial session text and still wins.
- (Separately: my *first* turn-level pass used *whole-session* gold — all ~35 turns of the gold session —
  and saturated dense to ≈1.0. That was a labelling bug on my side, fixed by the oracle `has_answer` join;
  it is not what the prior session-level harness did.)

---

## Why OSAM adds nothing (mechanism), and why it isn't just λ-forgetting

The robust, granularity-independent fact is **OSAM never adds value**: the α=0.7 blend Δ has a
non-positive 95% CI at both granularities (turn −0.022 [−0.038, −0.009]; session −0.007 [−0.025, +0.010]).
*How* it fails differs by grain:
- **Turn level:** pure-OSAM (α=0) MRR 0.116 ≈ chance 0.130 — ranking by the OSAM readout alone is
  indistinguishable from random. At fine grain OSAM carries **no usable relevance signal**.
- **Session level:** pure-OSAM MRR 0.625 ≫ chance 0.140, but strictly **below** dense (0.845) and
  **non-additive** (blend is a wash). OSAM re-encodes coarse session identity that dense already captures —
  redundant, not complementary.

Either way OSAM cannot *improve* ordering, which is what this benchmark rewards. This is **not** merely a
λ-horizon (forgetting) artifact: at session level OSAM is warmed with only ~49 updates (within its
~10-step horizon for recent items) and still adds nothing. The turn-level recency buckets are consistent
(the `11-50` bucket, n=38, nets a loss: 4 helped / 10 hurt; the `0-10` bucket n=5 is too small and
count-vs-mean contradictory to lean on). The decisive evidence is the non-positive blend CI at both grains
plus pure-OSAM's behaviour, not the thin recency split.

---

## Decision Rule (§7) — rows that fire

- **"Evidence present but mis-ordered → reranking problem → improve the ranking signal."** ✅ Primary.
  oracle@pool_k 0.955 (evidence is almost always in the dense top-50) while dense MRR is 0.49 → the
  bottleneck is ordering, not candidate generation. The cross-encoder fixes the ordering (+0.20); OSAM does not.
- **"Cross-encoder matches/beats OSAM → no differentiated mechanism → prefer the simpler baseline; be honest."** ✅
  Cross-encoder beats both dense and OSAM with a strictly-positive CI. OSAM is not a differentiated mechanism here.
- **"OSAM hurts on mixed/long histories → routing / state-isolation; revisit λ."** ⚠️ Partial. OSAM does hurt,
  but the recency analysis shows it also hurts within-horizon, so λ tuning will not recover value.
- **NOT** candidate-generation (oracle high), **NOT** memorization/degeneracy (guard clean, chance 0.13,
  pure-OSAM ≈ chance), **NOT** benchmark saturation (dense 0.49, large cross-encoder headroom).

**Next move (per the rule):** do **not** redesign or recalibrate OSAM as the retrieval mechanism — there is
no OSAM lift to gate. If a retrieval lift on LongMemEval-S is wanted, adopt a **cross-encoder reranker**
(retrieve-then-rerank); that is the demonstrated, significant lever. OSAM's associative-echo design does
not read query↔document relevance, which is exactly what this benchmark rewards.

---

## Why this measurement is trustworthy (controls)

- **Precise evidence** (oracle `has_answer` join, 100% recovery, ~1.9 turns/record) — no whole-session saturation.
- **Per-question store isolation** (fresh vector store + OSAM state per record) — no cross-question contamination.
- **Chance baseline** (0.13) and **dedup of identical embeddings** (453 collapsed) — the two controls the
  retracted hard-band "2× lift" lacked; CI alone is insufficient.
- **Non-degeneracy by construction** (evidence ⊊ history; abstention questions dropped) — 0 exclusions.
- **Dense sanity tripwire** False — dense ranking direction is correct (no inverted-L2 bug).
- Bootstrap 95% CIs on paired per-query RR deltas; n=200 of 479 evaluable (uniform, seed=99).

## Caveats / scope
- Single bi-encoder (`all-MiniLM-L6-v2`), pool_k=50, retrieval-stage only (no answer-quality / exact-match).
- Cross-encoder is `ms-marco-MiniLM-L-6-v2` (CPU, deterministic) — a true cross-attention relevance reader.
- 0-10 recency bucket is small (n=5). LongMemEval-M / LoCoMo (WP-110 P2/P3) remain deferred.
- This is a **foundry-scope** measurement (no end-user/runtime promotion).
