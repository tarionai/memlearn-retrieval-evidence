# Enterprise Data Adapter

**Status: validated implementation** — the full chain has been run against a real
external cloud database (hosted Neon PostgreSQL) under a frozen evaluation contract,
and reproduces bit-identically. This is *not* a deployed product, a customer pilot, or
a production retrieval default. Read the **Scope and standing prohibitions** section
before citing any number here.

A skeptical reviewer with no Neon credentials and no write access can, in ≤15 minutes:
read the ingestion contract, follow one frozen evaluation report, inspect one **PROMOTED**
and one **BLOCKED** governed-output decision plus the append-only audit record, and
reproduce the chain end-to-end against their **own** Postgres with one command.

---

## The chain

```
  ┌─────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
  │   Neon      │   │ ingestion  │   │ retrieval  │   │ evaluation │   │  governed  │   │   audit    │
  │  source     │──▶│  contract  │──▶│  pipeline  │──▶│    run     │──▶│   output   │──▶│   record   │
  │ (real, ext) │   │ (+lineage) │   │ (5 arms)   │   │  (frozen)  │   │   (gate)   │   │(append-only)│
  └─────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘
   20k profiles      typed schema      FTS / dense      run 1554d517     PROMOTED /       JSONL, two
   hosted Neon PG    + source_hash     RRF hybrid /     evaluation_      BLOCKED by        UTC clocks
                     + load report     cross-encoder    status=complete  deterministic     (event_at,
                                       reranked         qrels=frozen     rule              recorded_at)
```

The first five links already existed in the `mem-learn` candidate-search benchmark. The
**governed output** and **audit record** (the two rightmost links) are the net-new
deterministic terminal of this proof: a pure rule that turns the eval contract's
citability condition into an executable verdict, and an append-only record of every
governance run.

| # | Link | Implemented by | Evidence here |
|---|---|---|---|
| 1 | **Neon source** (real, external) | `load_neon.py` — transactional loader: migrations → upsert → pgvector embeddings (mean-pool chunking) → HNSW indexes → load report | `evidence/neon_load_report.json` |
| 2 | **Ingestion contract** | typed schema + `source_hash` content addressing + PII-redaction count + lineage fields (pg/pgvector versions, counts, index names, `git_commit`, UTC timestamps) | `evidence/neon_load_report.json` |
| 3 | **Retrieval pipeline** | PostgreSQL native FTS (`fts`, `fts_idf8`), pgvector dense, RRF hybrid, and cross-encoder reranked arms | `evidence/candidate_search_eval_report.json` → `retrieval_variants` |
| 4 | **Evaluation run** | frozen `EVAL_CONTRACT.md` (label rubric, metric math fixed before results, `claim_boundaries`, citability gate); MRR / Recall@k / NDCG@k / P@10 / pool-recall@50 | `evidence/candidate_search_eval_report.json` (run `1554d517`) |
| 5 | **Governed output** | `enterprise_adapter/gate.py` — pure deterministic rule over the frozen report | `evidence/governed_output_promoted.json`, `evidence/governed_output_blocked.json` |
| 6 | **Audit record** | `enterprise_adapter/audit.py` — append-only JSONL writer | `evidence/audit_log.jsonl` |

---

## The governance gate (net-new)

The eval contract states the citability rule in prose:

> A run is only citable if `evaluation_status == complete` and `qrels_status == frozen`.
> — `EVAL_CONTRACT.md §7`

`gate.py` makes that rule executable. `evaluate_governance(report, requested_claims=None)`
is a pure function — no agent, no inference, no I/O beyond reading the report — that
returns a typed `GovernanceDecision`:

- **PROMOTED** — the run passed the citability rule and no forbidden claim was requested.
  The decision carries the *allowed-claims* set (every `claim_boundary` whose status begins
  with `Demonstrable`) and the run's lineage (`eval_run_id`, `git_commit`, `dataset_version`,
  `embedding_model_id`).
- **BLOCKED** — a single named condition failed: incomplete evaluation, unfrozen qrels, or an
  attempt to cite a `Not validated` / `False; do not claim` claim.

**The BLOCKED case is the point, not the PROMOTED one.** A rule that only ever says "yes" is
decorative. The frozen `governed_output_blocked.json` shows the gate rejecting an attempt to
cite *"The system improves recruiter satisfaction in production"* — a claim the contract marks
`Not validated` — against the very same real, frozen, citable run. Same input, same verdict,
every time. That is "candidate before promotion; deterministic before agentic" made concrete.

### What a PROMOTED decision looks like (run `1554d517`)

```json
{
  "status": "PROMOTED",
  "eval_run_id": "1554d517",
  "git_commit": "28c16a9",
  "dataset_version": "djinni_v0",
  "embedding_model_id": "all-MiniLM-L6-v2",
  "allowed_claims": [ "...retrieval over a real anonymized recruitment corpus", "..." ],
  "forbidden_claims": [ "...improves recruiter satisfaction in production", "...predicts hiring success", "..." ],
  "failed_condition": null
}
```

The audit record (`evidence/audit_log.jsonl`) pins each governance run to its source-load
report, the eval run id, the decision, and two UTC timestamps: `event_at` (when the evaluated
run was produced) and `recorded_at` (when the audit line was written). Lines are only ever
appended, never rewritten.

---

## What the numbers say (run `1554d517`)

Per-arm aggregate metrics over the frozen pool, three query families:

| Arm | NDCG@10 | MRR | P@10 |
|---|---|---|---|
| random_baseline | 0.000 | 0.028 | 0.00 |
| fts (PostgreSQL native) | 0.400 | 0.750 | 0.47 |
| dense (pgvector) | 0.692 | 1.000 | 0.77 |
| hybrid (RRF) | 0.620 | 0.833 | 0.73 |
| **dense_reranked (cross-encoder)** | **0.848** | 1.000 | 0.90 |
| hybrid_idf8_reranked | 0.837 | 1.000 | 0.87 |

The cross-encoder reranker is the clearest lift over the dense generator
(NDCG@10 0.692 → 0.848 on this pool). This is consistent with the standing finding
that cross-encoder reranking is the defensible retrieval lever — but read the ceiling below.

---

## Scope and standing prohibitions

This evidence is bounded. The artifact carries its limits affirmatively, not in fine print.

- **Three query families, not a corpus claim.** The frozen pool covers three recruiter
  queries (`platform_engineer_v1`, `qa_auto_v1`, `backend_postgres_v1`). No
  cross-position generalization or "best arm overall" claim is made.
- **Single human reviewer; inter-annotator agreement is N/A.** The report's
  `reviewer_count: 2` is a label-casing artifact (`Ed` and `ed` are the same person);
  **zero** judgment pairs are double-annotated, so Cohen's κ cannot be computed. The
  contract (`§3`) requires κ on a ≥20% double-judged sample before any cross-position claim —
  that bar is not met, so no such claim is made.
- **Reranker result is a lift *signal*, not a production default.** It is a per-arm
  comparison on one frozen pool (contract `§6`, Phase-1/2 allowed claims). It is **not** a
  validated retrieval default and does **not** predict production behaviour.
- **Recall is pool-relative.** Denominators are judged-relevant candidates in the frozen
  pool, never the corpus. **No corpus-level recall is reported** — the pool is not an
  exhaustive relevance census.
- **Standing prohibitions (contract §6, verbatim):** no corpus-level recall; no
  "PostgreSQL native FTS is BM25" (it is `to_tsquery` + `ts_rank_cd`, never BM25); no
  production-efficacy, fairness, or hiring-success claim — all listed `Not validated` in the
  report's `claim_boundaries` and therefore rejected by the gate.
- **Lineage caveat (honest):** `neon_load_report.json` records
  `pgvector_version: "not_installed"` and `index_sizes: "error"` — these are version-probe
  failures, not absences. The HNSW cosine indexes exist and the `dense` arm returns results,
  so pgvector is present and functional; only the introspection query failed.

## Data handling

No raw candidate rows — even anonymized — are published. The evidence here is the contract,
the typed schema and lineage report, **aggregate per-arm metrics only**, the
governed-output / audit records, the adapter code, and a synthetic reproduction pack.
Zero third-party-PII surface. The live Neon connection string never enters any committed
artifact.

---

## Reproduce it yourself (no Neon, no embeddings, ≤15 min)

The reproduction pack runs the full chain — ingest → retrieve → evaluate → govern → audit —
against **your own** Postgres, using fully synthetic profiles (`reproduction/synthetic_corpus.json`,
~30 hand-authored rows, no djinni data) and PostgreSQL native full-text search. The governance
gate and audit writer are the same published modules; only the corpus and retrieval arm are
swapped for a dependency-free local stand-in.

The only external dependency is `psycopg2-binary`; everything else is the Python standard
library plus the published adapter modules.

```bash
# 1. install the one dependency (use a fresh venv if you prefer)
pip install psycopg2-binary

# 2. start any Postgres (throwaway container shown; or point REPRO_PG_CONN at your own)
docker run --rm -d --name repro-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16

# 3. run the chain
python enterprise-adapter/reproduction/reproduce.py

# 4. tear down
docker rm -f repro-pg
```

Expected: an `fts` arm that beats the random baseline, a **PROMOTED** decision (synthetic run
is complete + frozen), and a **BLOCKED** decision (the gate rejects an attempt to cite a
`Not validated` synthetic claim). Outputs land in `reproduction/_output/`. The synthetic run
is a **mechanics demonstration of the chain and the gate**, not a benchmark — its metrics carry
no retrieval-quality claim.

---

## Verify the frozen evidence

```bash
cd enterprise-adapter/evidence && sha256sum -c SHA256SUMS.txt
```

The governed-output examples and audit log are regenerated deterministically by
`enterprise-adapter/freeze_examples.py` (timestamps are fixed constants), so the hashes are
stable across runs.

## Layout

```
enterprise-adapter/
  enterprise_adapter/        # the net-new gate + audit modules (pure, typed, tested)
    decision.py              #   GovernanceDecision, ClaimBoundary
    gate.py                  #   evaluate_governance() — the deterministic citability rule
    audit.py                 #   append-only JSONL audit writer
  evidence/                  # frozen artifacts + SHA256SUMS
    candidate_search_eval_report.json   # run 1554d517, aggregate metrics only
    neon_load_report.json               # ingestion lineage
    governed_output_promoted.json       # PROMOTED decision
    governed_output_blocked.json        # BLOCKED decision (Not-validated claim rejected)
    audit_log.jsonl                     # append-only record of both runs
    SHA256SUMS.txt
  reproduction/              # synthetic, no-Neon, your-own-Postgres reproduction
    synthetic_corpus.json
    repro_metrics.py
    reproduce.py
  tests/                     # gate + audit unit tests (run under the repo's pytest suite)
  freeze_examples.py         # deterministic regenerator for evidence/
```
