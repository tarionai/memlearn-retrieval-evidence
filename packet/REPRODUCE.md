# Reproduction

**Status:** verified, with stated scope · Smoke-tested in a **fresh venv** on the publishing
machine on 2026-06-12: clean install from `requirements.lock.txt`, then the n=5 run
end-to-end. Two missing transitive pins surfaced in that test (`psycopg2-binary`, `pgvector`)
and were added to the lock file — what is published is the post-fix, verified set. Scope of
that verification: package install and harness execution were exercised cold; the dataset
was already on disk and the two transformer models were in the local Hugging Face cache.
The download step itself was verified against the HF repo's file listing (both filenames in
step 2 exist there verbatim), not by re-downloading.

## Prerequisites

- Python 3.10.x (results were produced on 3.10.11) — CPU only, no GPU needed.
- ~2 GB disk: the dataset (~280 MB), two small transformer models (auto-downloaded from
  Hugging Face on first run), and the pinned packages.
- Paths below are relative to the repository's `apps/mem-learn/` directory, which is also the
  required working directory (the harness resolves `data/` relative to it).

## 1. Environment

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Unix: source .venv/bin/activate
pip install -r requirements.lock.txt
```

`requirements.lock.txt` is in this packet — versions are pinned to the set that produced the
n=200 results.

## 2. Data (two files, from the public HF dataset)

Download from [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
into `data/`:

```python
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs("data", exist_ok=True)
for f in ("longmemeval_s_cleaned.json", "longmemeval_oracle.json"):
    shutil.copyfile(
        hf_hub_download("xiaowu0162/longmemeval-cleaned", f, repo_type="dataset"),
        f"data/{f}")
```

Both files are required: the S split provides the haystacks; the oracle split provides the
per-turn `has_answer` flags used for precise round-level evidence (see `DATASET.md`).

**Known quirk:** `datasets.load_dataset()` in non-streaming mode fails on this dataset
(pyarrow JSON `block_size` OverflowError on very large records). The harness deliberately
reads the downloaded JSON with stdlib `json.load` instead — do not "fix" this by switching
to `load_dataset()`.

## 3. Smoke mode (~minutes) — verify the harness runs

```
python src/validation_mvp/run_longmemeval_retrieval.py --n 5 --hybrid
```

Expected: per-record progress lines, then a `== SUMMARY (n=5, ...) ==` block covering
chance, dense, OSAM, lexical (BM25), hybrid-RRF, cross-encoder, and hybrid+cross-encoder,
and a result JSON written under `state/intermediate/`.

**The n=5 numbers will NOT match the published table — by design.** At n=5 the estimates
are noise (this packet documents a sign flip between n=5 and n=200 in
`NEGATIVE_FINDINGS.md` § Small-n warning). Smoke mode verifies the harness runs from these
instructions alone; it does not re-measure anything. Note: smoke mode samples from the full
S split, so it still requires the complete ~280 MB dataset download — it saves compute time,
not download size.

## 4. Full reproduction of the published numbers

```
python src/validation_mvp/run_longmemeval_retrieval.py --n 200 --hybrid
python src/validation_mvp/run_longmemeval_retrieval.py --n 200 --granularity session
python src/validation_mvp/ppr_arm.py --mode full --n 200
```

Defaults already encode the frozen protocol: `--seed 99`, `--pool-k 50`. The first command
reproduces the main 8-variant table (`METRICS.md`); the second the session-level control; the
third the PPR arm (requires the spaCy model from the lock file). Compare against the frozen
JSONs in `results/` (embeddings are cached per record as `.npz`, so re-runs are faster than
first runs).

**Success criterion: each arm's sign and CI-excludes-zero status — not the third decimal.**
The run is seeded end-to-end (subset seed, chance shuffle, bootstrap RNG), so exact point
match is expected only on identical dependency versions; different torch/BLAS builds can
flip near-tie ranks at the margin. A reproduction that gets every verdict in the `METRICS.md`
table right (which lifts validate, which mechanisms fail) has reproduced the result.

## Optional: cross-backend lexical arm

The dashboard's cross-backend table (same pipeline, lexical engine swapped) is outside this
packet's frozen scope, but its PostgreSQL arm is reproducible with the same harness:

```
python src/validation_mvp/run_longmemeval_retrieval.py --n 200 --hybrid --backend postgres
```

Requires a reachable PostgreSQL instance via the `MEMLEARN_PG_CONN` environment variable.
Reference artifact (not frozen in this packet): `state/intermediate/longmemeval_s_hybrid_results_n200_postgres.json`,
sha256 `8235869279e4…`, run 2026-06-06.

## What reproduction does and does not establish

Running these commands re-derives the published result from the public dataset on your
hardware, judged by the success criterion above: every verdict in the `METRICS.md` table
(which lifts validate, which mechanisms fail) should reproduce; point estimates should land
close but may differ at the margin on different dependency builds. Latency figures are
environment-bound (`ENVIRONMENT.md`) and will differ on your machine.
