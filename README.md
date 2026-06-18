# memlearn-retrieval-evidence

**Status: validated implementation — benchmark harness + frozen evidence packet.**

> **Latest release:** [`governed-agent-v1`](https://github.com/tarionai/memlearn-retrieval-evidence/releases/tag/governed-agent-v1) — a LangGraph state graph + MCP server over the adapter chain, reusing the same deterministic gate + audit verbatim. See [§ Governed retrieval agent](#governed-retrieval-agent) below.
>
> Prior release: [`enterprise-adapter-v1`](https://github.com/tarionai/memlearn-retrieval-evidence/releases/tag/enterprise-adapter-v1) — the enterprise data adapter, frozen and citable by tag. See [§ Enterprise data adapter](#enterprise-data-adapter).

A curated public excerpt of a private memory-retrieval system: its typed contracts,
contract tests, a frozen LongMemEval-S evidence packet, and a runnable benchmark harness
for independent protocol-level reproduction. The packet includes bootstrap 95% CIs and
two negative findings. It is deliberately small. It is not the product — the memory kernel itself stays private — and it makes no claim beyond what the packet measures.

Companion surface: the same packet is served at
[relevance.tarion.ai](https://relevance.tarion.ai/#benchmark-packet); the SHA-256 hashes
in [`MANIFEST.md`](MANIFEST.md) tie both copies to identical bytes. Project write-up:
[tarion.ai/projects/search-relevance-evidence](https://www.tarion.ai/projects/search-relevance-evidence).

## Key result

Transcribed verbatim from the frozen packet ([`packet/METRICS.md`](packet/METRICS.md));
every number is measured-on-frozen-benchmark (LongMemEval-S, n=200, seed=99):

| Variant | MRR | Δ MRR vs dense | 95% CI (Δ) | Verdict |
|---|---|---|---|---|
| Dense bi-encoder (baseline) | 0.491 | 0.000 | — | baseline |
| Dense + OSAM rerank (α=0.7) | 0.469 | −0.022 | [−0.038, −0.009] | no lift — rejected |
| Dense + PPR (graph) | 0.432 | −0.059 | [−0.116, −0.005] | no lift — rejected |
| Cross-encoder rerank | 0.693 | +0.202 | [+0.144, +0.255] | validated |
| Hybrid + cross-encoder | 0.715 | +0.225 | [+0.166, +0.281] | validated — best config |

The two negative findings are intentional evidence: the harness rejected both of the
system's own differentiated mechanisms. LongMemEval-S is the benchmark's published
"small"-haystack variant (500 records, each ~50 sessions / ~500 turns), not a subset
created for this evaluation; the n=200 seed=99 sample drawn from it is this packet's
choice, documented with its sampling rule in [`packet/DATASET.md`](packet/DATASET.md).

## What this repository proves — and does not prove

**Proves (every number measured-on-frozen-benchmark, LongMemEval-S, n=200, seed=99):**

- Results were generated under the documented evaluation contract: dense baseline MRR 0.491; cross-encoder
  rerank Δ +0.202 (95% CI [+0.144, +0.255]); hybrid BM25+dense+cross-encoder Δ +0.225
  (95% CI [+0.166, +0.281]) — the strongest measured configuration; its margin over
  cross-encoder-alone is supported by a documented paired per-query test (+0.0224,
  95% CI [+0.0047, +0.0468]). Full table: [`packet/METRICS.md`](packet/METRICS.md).
- **Two negative findings are reported as final**, not buried: the system's own
  associative-memory rerank (OSAM, Δ −0.022, CI entirely below zero) and a
  personalized-PageRank graph arm (Δ −0.059) both *hurt* retrieval and were rejected.
  Diagnostics: [`packet/NEGATIVE_FINDINGS.md`](packet/NEGATIVE_FINDINGS.md).
- The typed contracts and service slice shipped here are the modules exercised by the published harness. `MANIFEST.md` records their hashes and origin tags so the public evidence slice can be checked for internal consistency and future changes.

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

**All commands — in this README and in `packet/REPRODUCE.md` — run from the root of a
clone.** (Provenance note: the tree mirrors the private monorepo directory it was
excerpted from, so where the frozen `packet/REPRODUCE.md` says "relative to the
repository's `apps/mem-learn/` directory," that directory corresponds to this
repository's root.)

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

210 tests pass with only `pytest` and `numpy` installed — no model download, no
network. Two groups inside the byte-identical `tests/unit/test_in_memory_adapters.py`
are deselected in `pyproject.toml` (with explanatory comments there): a smoke test of
the full memory kernel, which this repository deliberately does not vendor, and the
`SentenceTransformerEmbedder` tests, which require the real embedding model — those run
under the full benchmark venv (`packet/requirements.lock.txt`) by overriding the
deselect, as documented in `pyproject.toml`. The test file itself is unmodified.

## Reproducing the benchmark

Follow [`packet/REPRODUCE.md`](packet/REPRODUCE.md) from the repository root: create a
venv, `pip install -r packet/requirements.lock.txt`, download the two public dataset
files it names, then run the level of reproduction you have time for. The three levels
establish different things — do not conflate them:

| Command | Purpose | Expected result |
|---|---|---|
| n=5 smoke run (`REPRODUCE.md` §3) | Verify installation and the execution path from these instructions alone | All arms execute and a result JSON is written; the n=5 numbers are statistical noise by design and validate nothing |
| n=200 frozen-protocol rerun (`REPRODUCE.md` §4) | Independent protocol-level reproduction of the published evidence | Every sign and CI-excludes-zero verdict in `packet/METRICS.md` matches; point estimates may differ at the margin across dependency builds |
| `python -m pytest -q` | Validate contracts and deterministic behavior | 210 tests pass, no network (two disclosed deselections — see `pyproject.toml`) |

Read `REPRODUCE.md`'s "What reproduction does and does not establish" section before
comparing numbers.

Out of scope here: `REPRODUCE.md`'s optional cross-backend PostgreSQL arm references a
result artifact this repository does not freeze; the harness flag exists, but that arm
is not part of this repository's claims.

## Enterprise data adapter

A separate, self-contained proof: a real anonymized recruitment corpus on hosted **Neon
PostgreSQL**, taken through a typed ingestion contract → pgvector + cross-encoder
retrieval → a frozen evaluation → a **deterministic governance gate** → an append-only
**audit record**. The gate turns the eval contract's citability rule into an executable
verdict (PROMOTED / BLOCKED) — and ships one of each, frozen and hash-manifested. It is
reproducible against your own Postgres with one command and no Neon access.

→ [`enterprise-adapter/README.md`](enterprise-adapter/README.md). Status: **validated
implementation**; aggregate metrics only, no raw candidate rows, scope and standing
prohibitions stated on the page.

**Release:** [`enterprise-adapter-v1`](https://github.com/tarionai/memlearn-retrieval-evidence/releases/tag/enterprise-adapter-v1) — a frozen, hash-manifested snapshot you can cite by tag (the resume/portfolio link points here).

## Governed retrieval agent

A runnable agent layer over the adapter chain: the same deterministic gate and
append-only audit, reused **verbatim**, wrapped in two artifacts. (1) An explicit
**LangGraph** state graph — typed state, a SQLite checkpointer for durable
execution, a retry policy on the database nodes, and a verdict-conditional **safe
fallback** to a first-class BLOCKED terminal that still writes an audit line;
when the database is unreachable after retries the graph fails closed onto that
same terminal. (2) A real **MCP server** exposing three well-typed tools, with
the audit id in every return and no raw candidate rows or connection string ever
leaked. Two modes share one graph: a cold-reproducible deterministic mode, and
an optional agentic mode where an LLM proposes claims and the deterministic gate
disposes.

→ [`governed-agent/README.md`](governed-agent/README.md). Status: **runnable
artifacts**; deterministic mode is hash-frozen under `governed-agent/evidence/`
(`sha256sum -c SHA256SUMS.txt`) and covered by an 8-test suite (database-backed
tests skip cleanly without Postgres).

**Release:** [`governed-agent-v1`](https://github.com/tarionai/memlearn-retrieval-evidence/releases/tag/governed-agent-v1) — a frozen snapshot you can cite by tag.

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
