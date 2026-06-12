# PPR Arm — §14 Phase 2a-1 Retrieval Ablation Gate (HippoRAG-style)

**Date:** 2026-06-04 · LongMemEval-S, turn-level, n=200, bench_seed=99 (same subset as the OSAM/cross-encoder run).
**Harness:** `src/validation_mvp/ppr_arm.py` · **Result:** `ppr_arm_results_n200.json` · **Smoke:** `ppr_reachability_smoke_n20.json`.
**What this answers:** the PPR / graph-retrieval arm of the doc's gate — the open question blocking Phase 3 (SemanticStore).

## Verdict

**PPR fails the gate.** The fusion-invariant evidence is decisive: `ppr_only` ranks **worse than dense**
(0.341 vs 0.491, Δ −0.151) and a full re-rank giving PPR 50% weight leaves **Recall@10 flat** (0.734 →
0.740) — i.e., PPR surfaces essentially **zero dense-missed evidence** into the top-10, which is its entire
recall-recovery claim. (The `dense_plus_ppr` MRR Δ −0.059 corroborates but is partly blend dilution from
mixing a weaker signal — the recall-flat + `ppr_only` results are the load-bearing ones.) Per §14 Phase 2a-1
("if `dense_plus_ppr` shows no measurable Recall@10 lift → full SemanticStore investment (Phase 3) is not
justified"), **Phase 3 is not justified *as a retrieval investment*, under the available (non-LLM, spaCy)
extraction.** Combined with the OSAM arm (also no lift), **both differentiated retrieval mechanisms the
architecture bet on fail the gate; the only mechanism that helps is the off-the-shelf cross-encoder (+0.20).**

| variant | MRR | hit@5 | Recall@10 | Δ_MRR vs dense | CI95(Δ) | verdict |
|---|---|---|---|---|---|---|
| chance | 0.028 | — | — | — | — | floor |
| **dense_only** | **0.4915** | 0.690 | 0.734 | 0.000 | — | baseline (reproduces OSAM-run 0.491 ✓ tripwire) |
| ppr_only | 0.341 | 0.555 | 0.557 | **−0.151** | [−0.214, −0.089] | significantly worse |
| dense_plus_ppr | 0.432 | 0.655 | 0.740 | **−0.059** | [−0.116, −0.005] | significantly hurts; recall flat |

seed_hit_rate **1.0**, reachability **0.995** (evidence turn graph-connected to query seeds).

## Decision table (advisor's reachability-conditioned logic) — which cell we land in

| reachability | dense_plus_ppr lift | conclusion | **this run** |
|---|---|---|---|
| low | (any) | non-evaluable — extraction can't build a connecting graph (Phase 2a-0 fails) | |
| high | CI lower > 0 | PPR adds value → Phase 3 justified | |
| **high (0.995)** | **no lift (CI < 0)** | **earned null → Phase 3 not justified** | **◄ here** |

This is an *earned* null, not "non-evaluable": the graph reliably connects query to evidence (reachability
0.995), PPR simply doesn't rank the evidence better than dense already does.

## The multi-hop hypothesis is refuted by stratification

Graph PPR's distinct claim is **multi-hop recall** — surfacing evidence dense misses by walking entity
connections. If true, the lift should concentrate in `multi-session` and `temporal-reasoning` (108/200 here).
It doesn't:

| question_type | n | dense_plus_ppr Δ_MRR | CI95 |
|---|---|---|---|
| multi-session | 61 | −0.063 | [−0.165, +0.042] |
| temporal-reasoning | 47 | −0.106 | [−0.221, +0.002] |
| knowledge-update | 29 | −0.085 | [−0.203, +0.041] |
| single-session-user | 23 | +0.019 | [−0.147, +0.208] |
| **single-session-assistant** | 29 | **+0.138** | [−0.001, +0.287] |
| single-session-preference | 11 | −0.454 | [−0.642, −0.254] |

The multi-hop types are **negative**. The only positive point estimate is `single-session-assistant` — a
*single-hop* type, and its CI still touches zero. So the lone bright spot is the opposite of the multi-hop
hypothesis and not significant. No subset rescues PPR.

## Why PPR fails here (mechanism)

`direct_overlap` was 0.75 in the smoke — for 75% of questions a query entity is *literally in* the evidence
turn, which **dense already retrieves**; PPR adds nothing there. For the remaining ~25% (where multi-hop
would matter), the extracted graph is a **dense co-occurrence hairball** (mean ~9,500 nodes / ~467k edges /
~5.6 components per question) — PPR mass diffuses broadly rather than concentrating on the evidence turn.
IDF / node-specificity weighting (HippoRAG's faithful counter to density) was applied and still does not
recover discriminative ranking. Net: PPR re-ranks toward high-connectivity generic turns, hurting MRR.

## Caveats / bound on the null (read before generalizing)

- **Extraction is spaCy NER + noun chunks, NOT HippoRAG's LLM-OpenIE triples.** No `ANTHROPIC_API_KEY` is
  available, so the doc's default LLM `EntityExtractorPort` could not be used. The spaCy graph is denser and
  noisier than LLM triples. This is the **direct parallel to the OSAM "simplified-variant" caveat**: the null
  is for *PPR over a spaCy co-occurrence graph*, which is weaker than "HippoRAG-with-LLM-extraction adds
  nothing." A faithful LLM-OpenIE graph is the one untested refinement — but it is **gated on Phase 2a-0**
  (extraction precision ≥ 0.80 / pollution ≤ 0.15), which itself requires the LLM extractor and ground-truth
  labels. **Do not build it speculatively** — it is higher-cost and lower-prior than the cross-encoder lever
  already in hand.
- **Reachability 0.995 is *connectivity*, not Phase 2a-0 clearance.** The doc's 2a-0 gate is extraction
  *precision/pollution*, which this run did **not** measure (no ground-truth entity labels). An all-pairs
  co-occurrence graph over every noun chunk (~467k edges/question) is plausibly **high-pollution** by
  construction. So: this is an earned null *for the spaCy-graph config*, and 2a-0 was never passed — which
  means the result does **not** falsify HippoRAG-with-LLM-OpenIE. "Earned null" applies to what was run, not
  to graph retrieval in general.
- **The gate tests only the *retrieval* value of SemanticStore.** Phase 3's other functions —
  conflict-flagging (supersession/contradiction), learning-kernel node-weight updates, temporal validity
  windows — are untested by a retrieval ablation. "Phase 3 not justified" here means **not justified as a
  retrieval investment**, not "SemanticStore has no purpose."
- Turn-level granularity, pool over all turns (PPR ranks the full turn set — its recall claim is not defined
  away). Bootstrap 95% CIs on paired per-query RR deltas; n=200 of 479 evaluable.
- Small-n is misleading: at n=5 `ppr_only` looked **+0.168**; at n=200 it is **−0.151**. Same lesson as the
  cross-encoder reconciliation — small-n CIs hide the sign; trust the powered run.

## Combined gate status (both arms now run)

| arm | result | gate (§14 Phase 2a-1) |
|---|---|---|
| OSAM rerank | no additive lift (turn Δ −0.022; pure-OSAM ≈ chance) | **FAIL** — not load-bearing, keep off hot path |
| PPR graph | no lift (Δ −0.059, recall flat); earned null at reachability 0.995 | **FAIL** — Phase 3 not justified (under spaCy extraction) |
| cross-encoder (added) | +0.202 MRR, CI [+0.144, +0.255] | the actual retrieval lever |

**Net:** the architecture's two differentiated retrieval mechanisms (OSAM, PPR) do not beat a dense
bi-encoder on LongMemEval-S; a standard retrieve-then-rerank cross-encoder does. **Scope (do not over-read):**
this falsifies *legacy-OSAM* and *spaCy-graph PPR* as retrieval levers — it does **not** falsify
HippoRAG-with-LLM-OpenIE (untested, gated on Phase 2a-0) nor SemanticStore's non-retrieval functions
(conflict-flagging, learning-kernel weights, temporal validity). The in-hand lever is the cross-encoder;
clean-LLM-extraction graph retrieval remains the one unfalsified — but lower-prior — refinement.
