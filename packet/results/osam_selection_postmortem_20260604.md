# Was selecting OSAM a mistake? — Post-mortem against EXTERNAL-MEMORY-LEARNING-KERNEL.md v2.4

**Date:** 2026-06-04 · Source: `docs/architecture/EXTERNAL-MEMORY-LEARNING-KERNEL.md` (read in full).
**Empirical basis:** `longmemeval_s_findings_20260604.md` (OSAM no additive value; pure-OSAM ≈ chance at turn level; cross-encoder is the lever).

## Verdict (lead)

**Selecting OSAM to *evaluate* was not a process mistake — it was a correctly-gated, low-prior bet that
the design document itself flagged as unvalidated and pre-wrote a kill-gate for. The benchmark is that
gate firing exactly as designed.** The real mistakes were downstream of the design: (1) the shipped
**+0.0813 "buyer-packet lift"** treated OSAM as load-bearing *before* the gate passed — a direct violation
of the doc's own §15 instruction; and (2) the prolonged α/threshold tuning on degenerate local corpora.
The lesson is not "don't pick OSAM" — it is "honor your own §14/§15 gate earlier."

## The document predicted this outcome — verbatim

- **§15 Blind Spot (line 1316):** *"OSAM reranker lift is unvalidated in agent context. δ-mem's gains were
  measured with backbone coupling. OSAM as a standalone reranker is a principled adaptation — not a
  benchmarked mechanism. Sprint 2 must produce empirical lift measurements before OSAM is treated as
  load-bearing."*
- **§14 Phase 2a-1 Retrieval Ablation Gate (blocking, line 1233):** *"if `dense_plus_osam` shows no
  measurable Recall@10 lift over `dense_only` → `AssociativeStateEngine` is optional, not load-bearing; do
  not build it into the hot path."*
- **§1.2 / §12 (lines 126, 1142):** OSAM's original readout→text role was **eliminated** as non-portable
  ("no text inverse without backbone coupling"); it was **repurposed** as a cosine reranker — an *adaptation*,
  explicitly not a validated mechanism.

So the architecture (a) knew δ-mem's gains were backbone-coupled, (b) disclosed the standalone-reranker port
was unvalidated, and (c) built the exact ablation gate that would kill it if it showed no lift. My
LongMemEval result *is* the OSAM arm of that gate — run on a real public benchmark with precise evidence,
which is **stronger** than the doc's planned "200+ synthetic queries." OSAM fails the gate. Prescribed
action per the doc: do not build it into the hot path. Confirmed.

## The weakest assumption (what the design under-argued)

The doc calls the reranker port a "principled adaptation" but never argues *why a recency-weighted
associative echo would carry query↔document relevance independent of the dense bi-encoder.* It doesn't.
Mechanistically, the readout `r_t = S q_t` with Hebbian `S = Σ λ^(t-i) v_i k_iᵀ` (and v=k=embedding in the
simplified port) ≈ `Σ_i λ^(t-i) (emb_i·q) emb_i` — a **recency-weighted, query-similarity-weighted sum of
past observations**. Cosine-scoring candidates against it rewards "looks like recently-seen context that was
already query-similar." That is a **recency/association prior, not a relevance judgment**, and it is largely
**redundant with dense cosine** (which already ranks by query similarity) while adding a **recency bias that
is actively harmful when the evidence is an old turn**. This was derivable a priori. The benchmark confirmed
it precisely: pure-OSAM ≈ chance at turn level (no independent relevance signal); redundant-with-dense at
session level; net hurt where evidence is not recent.

So: **not a process mistake, but a low-prior bet** — a sharper read of *why δ-mem works* (backbone coupling,
not the matrix per se) could have deprioritized the standalone-reranker hypothesis before a single tuning cycle.

## What was actually a mistake (and what wasn't)

| Item | Mistake? | Why |
|---|---|---|
| Selecting OSAM as a hypothesis to ablate | **No** | Credible recent source (δ-mem), near-zero build cost (pure math, no training), fenced by a blocking gate. Cheap gated exploration of a credible idea is good engineering. |
| Shipping +0.0813 as a buyer-facing lift | **Yes, if it left the building** | The number was produced on a degenerate corpus (warmup ≡ gold) and treats OSAM as load-bearing before the §14/§15 gate passed — a violation of "must produce empirical lift before load-bearing." How far it actually went externally is not verifiable from this repo; the defect is the claim's provenance regardless of audience. |
| Months of α/threshold/pool tuning on local corpora | **Yes (sequencing)** | Tuning a knob already at its safe maximum on corpora that cannot validly measure value. The gate (public benchmark) should have come first. |
| Not testing PPR yet | **Now tested — also FAILS** | The §14 gate's second arm (`dense_plus_ppr`) was run (n=200, HippoRAG-style spaCy graph): Δ −0.059 MRR, recall flat, earned null at reachability 0.995. Phase 3 / SemanticStore not justified by retrieval lift under spaCy extraction. See `ppr_arm_findings_20260604.md`. (LLM-OpenIE extraction is the one unfalsified refinement, gated on Phase 2a-0, lower-prior than the cross-encoder.) |

## Steel-man: "OSAM *was* the right pick"

δ-mem is recent and credible; external associative memory was a legitimately novel angle; the test cost
almost nothing; the architecture correctly fenced the bet so failure costs only the ablation, not the
product; and you could not *know* dense-MiniLM would be this strong on these benchmarks without measuring.
Crucially, the cross-encoder lever (+0.20 MRR) was identified *because* OSAM was run as the baseline-to-beat.
On this reading, the exploration did its job — the failure of the OSAM bet is a *successful* experiment.

## Scope caveats & missing data

**1. The benchmark tested the *simplified* variant, not the full §8 spec.** The engine run on LongMemEval is
`osam_simplified_hebbian` — `S += β·(emb·embᵀ)`, a raw Hebbian outer product. The doc's §8 spec is richer: a
gated delta rule with fixed projections (`k=L2norm(tanh(W_k·emb))`, `v=W_v·emb`) and an error-correction term
`(v − S·k)kᵀ`. My closed-form `r_t ≈ Σ λ^(t-i)(emb_i·q)emb_i` is exact only for the simplified variant, so the
verdict's mechanism argument is, strictly, about what was run. **Bound on this gap (why it does not change the
direction):** (a) the redundancy critique is largely *mechanism-invariant* — the delta-correction changes
*what is stored* (decorrelated values), but the readout is still `S·q` cosine-scored against candidates, i.e.
an associative echo, not a relevance reader; projections add no query↔doc relevance reading. (b) Direct
evidence the richer variant does not rescue it: **Option B (learned projections) failed both gates at n=125**
(see `state/todo.md`). Therefore the full delta-rule OSAM is the one untested arm, but it is **low-prior and
lower-value than the PPR arm — do not benchmark it** (that would contradict the converged "stop OSAM tuning").

**2. Whether the δ-mem paper ever reported OSAM as a *standalone retrieval* signal** (decoupled from the
backbone). The doc cites only backbone-coupled gains. No standalone-retrieval result ⇒ the reranker-port prior
was low and the doc knew it (strengthens "low-prior bet"); a standalone result ⇒ better-founded. Not
resolvable from this repo (would need arXiv 2605.12357) — the *external* version of caveat #1.

## Bottom line

The architecture was honest about OSAM (§15) and disciplined about gating it (§14). The process around it
was not: a buyer claim and a tuning campaign ran *ahead* of the gate. Selecting OSAM wasn't the mistake;
acting on it before the evidence was. **Update:** the doc's other gated arm — PPR / graph retrieval — has now
also been run and **also fails** (`ppr_arm_findings_20260604.md`). So *both* differentiated retrieval
mechanisms the architecture bet on (OSAM, PPR) fail the §14 gate on LongMemEval-S, while a plain
cross-encoder clears it (+0.20 MRR). The empirically-supported path is retrieve-then-rerank with a
cross-encoder, not OSAM and not (spaCy-graph) PPR. The one unfalsified graph refinement is LLM-OpenIE
extraction, which must first clear Phase 2a-0 and is lower-prior than the cross-encoder already in hand.
