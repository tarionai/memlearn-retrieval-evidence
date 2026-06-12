# memlearn-retrieval-evidence

**Status: validated implementation — benchmark harness + frozen evidence packet.**

A curated public excerpt of a private memory-retrieval system: its typed contracts, the
unit tests for those contracts, and a runnable LongMemEval-S benchmark harness that
reproduces a frozen evaluation packet (bootstrap 95% CIs, two negative findings). It is
deliberately small. It is not the product — the memory kernel itself stays private — and
it makes no claim beyond what the packet measures.

Companion surface: the same packet is served at
[relevance.tarion.ai](https://relevance.tarion.ai/#benchmark-packet); the SHA-256 hashes
in [`MANIFEST.md`](MANIFEST.md) tie both copies to identical bytes.

## What this repository proves — and does not prove

**Proves (every number measured-on-frozen-benchmark, LongMemEval-S, n=200, seed=99):**

- A locked evaluation protocol was run honestly: dense baseline MRR 0.491; cross-encoder
  rerank Δ +0.202 (95% CI [+0.144, +0.255]); hybrid BM25+dense+cross-encoder Δ +0.225
  (95% CI [+0.166, +0.281]) — the best configuration, earned by a paired per-query test,
  not max-picking. Full table: [`packet/METRICS.md`](packet/METRICS.md).
- **Two negative findings are reported as final**, not buried: the system's own
  associative-memory rerank (OSAM, Δ −0.022, CI entirely below zero) and a
  personalized-PageRank graph arm (Δ −0.059) both *hurt* retrieval and were rejected.
  Diagnostics: [`packet/NEGATIVE_FINDINGS.md`](packet/NEGATIVE_FINDINGS.md).
- The typed contracts and service slice shipped here are the real modules the harness ran
  against, byte-identical to their private originals (verifiable via `MANIFEST.md`).

**Does not prove:**

- **When the evaluation design was locked.** This repository's git history begins at its
  creation date and cannot corroborate the lock date. The lock-before-run claim is
  documented in [`packet/EVAL_CONTRACT.md`](packet/EVAL_CONTRACT.md) §1 and is calibrated
  to a *private* repository's history (the harness commit of 2026-06-04). Take the
  protocol-lock claim as documented-and-internally-corroborated, not publicly verifiable.
- Anything about the private memory kernel's quality beyond the two service modules
  vendored here.
- Any production, latency-at-scale, or customer-operations claim. Latency figures in the
  packet are scoped to the single measured environment
  ([`packet/ENVIRONMENT.md`](packet/ENVIRONMENT.md)).

## The 15-minute path

1. **(2 min)** [`packet/README.md`](packet/README.md) — the packet index and integrity
   statement.
2. **(4 min)** [`packet/METRICS.md`](packet/METRICS.md) — the eight-variant table with
   CIs, then [`packet/NEGATIVE_FINDINGS.md`](packet/NEGATIVE_FINDINGS.md) — what failed
   and why it was rejected.
3. **(5 min)** [`src/memlearn/primitives.py`](src/memlearn/primitives.py) and
   [`src/memlearn/ports.py`](src/memlearn/ports.py) — the typed contracts (frozen
   dataclasses with invariants in `__post_init__`; `Protocol` ports). Then skim
   [`tests/test_primitives.py`](tests/test_primitives.py) — the invariants are tested,
   not asserted in prose.
4. **(2 min)** [`MANIFEST.md`](MANIFEST.md) — SHA-256 of every file with its origin tag;
   exactly one code file is modified from its private original, and it is named there.
5. **(2 min)** Optionally: `pip install pytest numpy && python -m pytest -q` — the unit
   tests run with no dataset, no model download, no network.

If you want to spend more than 15 minutes: [`packet/REPRODUCE.md`](packet/REPRODUCE.md)
has the pinned commands. The n=5 smoke run works from a clone of this repository alone
(plus the public ~280 MB HuggingFace dataset download it documents).

## Layout

This repository mirrors the layout of the private monorepo directory it was excerpted
from. **The repository root corresponds to the private repo's `apps/mem-learn/`
directory** — where `packet/REPRODUCE.md` says "relative to the repository's
`apps/mem-learn/` directory," read "relative to this repository's root." Every command
in it then works verbatim from the root of a clone.

```
packet/                      frozen evidence packet (verbatim, hash-manifested)
src/memlearn/                typed contracts: primitives.py, ports.py, errors.py
src/memlearn/services/       the two service modules the harness exercises
src/memlearn/adapters/       in-memory adapters + deterministic fakes
src/validation_mvp/          the benchmark harness (self-bootstraps sys.path)
benchmarks/retrieval_ablation/  metrics + bootstrap-CI slice
tests/                       unit tests (no network, no dataset)
```

## Running the tests

```
pip install pytest numpy
python -m pytest -q
```

204 tests pass. One test inside the byte-identical
`tests/unit/test_in_memory_adapters.py` is deselected in `pyproject.toml` (with an
explanatory comment there): it smoke-tests the full memory kernel, which this repository
deliberately does not vendor. The test file itself is unmodified.

## Reproducing the benchmark

Follow [`packet/REPRODUCE.md`](packet/REPRODUCE.md) from the repository root: create a
venv, `pip install -r packet/requirements.lock.txt`, download the two public dataset
files it names, then run the n=5 smoke command. Read its "What reproduction does and
does not establish" section before comparing numbers — the success criterion is each
arm's sign and CI-excludes-zero verdict, not point-estimate match.

Out of scope here: `REPRODUCE.md`'s optional cross-backend PostgreSQL arm references a
result artifact this repository does not freeze; the harness flag exists, but that arm
is not part of this repository's claims.

## What is excluded, and why

This is a curated excerpt, not the product:

- **The memory kernel proper** (admission gate, semantic store, consolidator, learning
  kernel, and the rest of `memlearn/services/`) stays private. The two service modules
  here — `episodic_store.py` and `associative_state_engine.py` — are vendored because
  the harness imports them.
- **All persistent adapters** (PostgreSQL, Snowflake, graph) and the CLI stay private.
  The lock file's `psycopg2-binary`/`pgvector` pins remain because they are frozen
  evidence of the verified install set; no PostgreSQL code ships.
- **One code file is modified from its origin:** `benchmarks/retrieval_ablation/`
  `tournament_base.py` ships as a public slice containing only the `bootstrap_ci`
  function the harness uses (the full module imports the private CLI). Its docstring and
  `MANIFEST.md` disclose this. One test file, `tests/test_services_slice.py`, is a
  disclosed per-class excerpt for the same reason.
- Internal work-package references (`WP-…`, `IMPL_…`) in docstrings are retained
  deliberately — the packet's integrity statement treats internal references as part of
  the provenance story, and redacting them would break byte-identity for no
  confidentiality gain.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Applies to everything in this repository, code
and documentation alike.
