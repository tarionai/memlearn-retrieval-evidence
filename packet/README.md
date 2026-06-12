# Frozen Retrieval Benchmark Packet — LongMemEval-S, n=200

Self-contained evidence packet for the retrieval-stage benchmark behind this dashboard.
Read in this order:

| File | What it answers |
|---|---|
| `EVAL_CONTRACT.md` | What protocol was locked, when, and what changed after lock (one documented amendment) |
| `DATASET.md` | What data, what split, how sampled, how evidence is labeled |
| `METRICS.md` | The full 8-variant table: MRR, hit@5, NDCG@10, bootstrap 95% CIs, latency scope |
| `NEGATIVE_FINDINGS.md` | OSAM and PPR rejected, with diagnostics, plus the +0.0813 retraction postmortem |
| `REPRODUCE.md` | Pinned commands for the n=5 smoke run and the full n=200 reproduction |
| `requirements.lock.txt` | The pinned package set (fresh-venv verified 2026-06-12) |
| `ENVIRONMENT.md` | The single machine and model set all numbers were measured on |
| `results/` | Verbatim frozen research artifacts + raw result JSONs + `SHA256SUMS.txt` manifest |

## Integrity statement

The files under `results/` are **verbatim** copies of the working research documents and raw
result JSONs, frozen out of ephemeral working state on 2026-06-12 with a SHA-256 manifest.
**Publish decision: no redactions** — after review, the internal references they contain
(work-package numbers, internal design-doc section citations, the pre-gate buyer-claim
postmortem) were judged part of the integrity story, not a leak risk, and are published as-is.
The packet pages above transcribe numbers from these files and from the dashboard's single
source of truth; they never re-derive them.

Every number in this packet is **measured-on-frozen-benchmark** (LongMemEval-S, n=200,
seed=99, single published environment). Negative findings are final — no rehabilitation
framing. Verify any file against `results/SHA256SUMS.txt`.
