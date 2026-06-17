# Governed Retrieval Agent — LangGraph + MCP

A LangGraph state graph and an MCP server wrapped around the **already
cold-verified** [`enterprise-adapter/`](../enterprise-adapter/) chain. The
deterministic citability gate and the append-only audit are imported from the
published package **verbatim** — not reimplemented — so these artifacts inherit
that chain's reproducibility.

> What is new here is only the graph wiring, the optional LLM draft node, and
> the typed MCP tool surface. The retrieval, the gate, and the audit are the
> same functions the reproduction pack ships. Dependency direction is one-way:
> this code depends on the published modules, never the reverse, and
> `reproduction/reproduce.py` is left byte-identical.

---

## Two moves

**Move 1 — LangGraph.** An explicit, typed state graph with durable execution
(SQLite checkpointer), a retry policy on the DB nodes, and a verdict-conditional
**safe fallback**: a BLOCKED verdict is a first-class terminal that still writes
an audit line — never a silent drop. If the database is unreachable after
retries, the graph **fails closed** onto that same BLOCKED terminal.

**Move 2 — MCP.** Three well-typed tools expose the governed chain as a product.
The audit record id is part of every tool's return contract, and no tool ever
returns raw corpus rows or the connection string.

---

## The graph

```
        ┌─────────────────── retry x3 (transient DB) ───────────────────┐
        ▼                                                               │
  START ─▶ ingest ─▶ retrieve ─▶ evaluate ──▶ gate ──▶ emit_promoted ─┐ │
              │          │           │  (agentic)  │                   ├─▶ audit ─▶ END
              │          │           └─▶ draft ────┘                   │
              │          │                         └──▶ emit_blocked ──┘
              │          │                                   ▲
              └──────────┴──── DB down after retries ────────┘  (fail closed)
```

- **ingest / retrieve / evaluate** — wrap `ingest`, `retrieve_fts`,
  `build_report` from the reproduction pack. Each owns its own connection so the
  run state stays serializable and checkpointable. A `RetryPolicy` (3 attempts,
  backoff) covers transient DB failures; on exhaustion the node's error handler
  diverts to `emit_blocked`.
- **draft** *(agentic mode only)* — an LLM drafts a grounded answer from the
  retrieved candidates and **proposes** claims. The proposed claims are merged
  into the gate's input.
- **gate** — `evaluate_governance(report, requested_claims)`, **published module,
  verbatim**. The deterministic verdict.
- **emit_promoted / emit_blocked** — the verdict-conditional branch. BLOCKED
  names the `failed_condition`.
- **audit** — `append_audit_record(...)`, **published module, verbatim**. Runs
  on **both** paths.

## Two modes, one graph

| Mode | LLM | API key | Determinism |
|---|---|---|---|
| `deterministic` (default) | none | none | fully cold-reproducible; runs in CI |
| `agentic` | Claude draft node | `ANTHROPIC_API_KEY` | the draft is non-deterministic, but **the gate's verdict over the draft's claims is still deterministic** |

**LLM proposes, gate disposes.** In agentic mode the model can draft and propose
claims, but nothing reaches the client until the *same deterministic gate*
promotes it. The draft model is set in one constant
(`graph.DRAFT_MODEL = "claude-sonnet-4-6"`), never at call sites.

---

## Run it

Deterministic mode (needs a local Postgres — same one the reproduction pack uses):

```bash
docker run --rm -d --name repro-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
pip install -e governed-agent            # or: pip install -e governed-agent[agentic]
python -m governed_agent.run --mode deterministic
docker rm -f repro-pg
```

Expected: one **PROMOTED** run (allowed claims emitted) and one **BLOCKED** run
(an over-reach claim requested), both audited — mirroring `reproduce.py`. If
Postgres is unreachable the run fails closed to BLOCKED with a non-zero exit and
no traceback.

### MCP server

```bash
python -m governed_agent.mcp_server      # stdio transport
```

| Tool | Signature | Purpose |
|---|---|---|
| `run_governed_retrieval` | `(query, requested_claims=[]) -> GovernedResult` | run the graph; verdict + `audit_id` in the return |
| `check_claim_citable` | `(claim) -> CitableResult` | run the gate over one claim (no DB) — candidate-before-promotion in one call |
| `read_audit_log` | `() -> list[AuditLine]` | surface the append-only audit |

Connect it from Claude Desktop, the MCP Inspector, or any MCP client. The
in-process protocol round-trip is exercised in `tests/test_mcp.py`.

---

## Evidence

`evidence/` contains hash-frozen governed outputs, regenerated deterministically
(fixed timestamps) from the published frozen eval report:

```bash
python governed-agent/freeze_governed_examples.py
cd governed-agent/evidence && sha256sum -c SHA256SUMS.txt
```

- `governed_run_promoted.json` — PROMOTED verdict + allowed-claims set
- `governed_run_blocked.json` — BLOCKED verdict naming the failed condition
- `audit_log.jsonl` — the append-only record of both

## Tests

```bash
pip install -e governed-agent[dev]
pytest governed-agent -q
```

Eight tests: gate/audit reuse-by-identity, happy-path PROMOTED, conditional
BLOCKED, **checkpoint recovery** (resume without re-ingesting), **DB-down
fail-closed** + retry count, and the MCP round-trip. DB-backed tests skip cleanly
when no Postgres is reachable.

---

## Scope (honest)

- The synthetic corpus exercises the chain **mechanics and the deterministic
  gate** — it is not a retrieval benchmark. The dense / cross-encoder arm matrix
  is the frozen Neon evidence under [`../enterprise-adapter/evidence/`](../enterprise-adapter/evidence/).
- Deterministic mode is what is hash-frozen. The live-DB graph run has
  per-run timestamps and a git commit, so it is demonstrated by the CLI and the
  tests rather than frozen.
- Not closed here: production traffic, on-call, traffic shaping, cloud infra.
  Wiring live dense retrieval into the graph is a possible later extension, not
  part of this build.
