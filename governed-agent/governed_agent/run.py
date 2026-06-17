"""CLI entry point: run the governed-retrieval graph end-to-end.

    python -m governed_agent.run --mode deterministic

Mirrors the reproduction pack's output contract: ingest the synthetic corpus,
retrieve over native FTS, evaluate, then emit one PROMOTED run (no over-reach
claim) and one BLOCKED run (the over-reach claim requested). Both write an
audit line. Exit code is 0 only when the gate produced exactly that pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from . import chain
from .graph import build_graph

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
OUT_DIR = Path(__file__).resolve().parent.parent / "_output"


def corpus_run_id(corpus: dict) -> str:
    """Stable run id from the corpus digest — the checkpoint thread key."""
    return hashlib.sha256(json.dumps(corpus, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def _invoke(app, base_state: dict, thread_id: str) -> dict:
    return app.invoke(base_state, {"configurable": {"thread_id": thread_id}})


def run_once(dsn: str, *, mode: str, audit_path: Path, checkpoint_path: Path) -> dict:
    """Run the PROMOTED and BLOCKED demonstration pair. Returns a summary dict."""
    corpus = chain.load_synthetic_corpus()
    run_id = corpus_run_id(corpus)
    base = {
        "dsn": dsn,
        "corpus": corpus,
        "query": "platform engineer kubernetes terraform module authoring helm argocd",
        "top_k": 20,
        "mode": mode,
        "audit_path": str(audit_path),
        "source_load_report_ref": "synthetic_v0",
    }
    sqlite_conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    try:
        app = build_graph(SqliteSaver(sqlite_conn))
        promoted = _invoke(app, {**base, "requested_claims": []}, f"{run_id}-promoted")
        blocked = _invoke(
            app, {**base, "requested_claims": [chain.OVERREACH_CLAIM]}, f"{run_id}-blocked"
        )
    finally:
        sqlite_conn.close()
    return {"promoted": promoted["result"], "blocked": blocked["result"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the governed-retrieval graph.")
    parser.add_argument("--mode", choices=["deterministic", "agentic"], default="deterministic")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="Postgres DSN (or REPRO_PG_CONN-style URL)")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(exist_ok=True)
    audit_path = OUT_DIR / "audit_log.jsonl"
    if audit_path.exists():
        audit_path.unlink()  # fresh demo log; append-only semantics are tested separately
    checkpoint_path = OUT_DIR / "checkpoints.sqlite"
    checkpoint_path.unlink(missing_ok=True)  # each CLI run is independent; recovery is proven in tests

    summary = run_once(
        args.dsn, mode=args.mode, audit_path=audit_path, checkpoint_path=checkpoint_path
    )
    promoted, blocked = summary["promoted"], summary["blocked"]

    print(f"[mode={args.mode}]")
    print(f"  PROMOTED: {promoted['decision']} ({len(promoted['allowed_claims'])} allowed claims)")
    print(f"  BLOCKED:  {blocked['decision']} -> {blocked['failed_condition']}")
    print(f"  audit log: {audit_path} ({len(chain.read_audit_log(audit_path))} records)")

    ok = promoted["decision"] == "PROMOTED" and blocked["decision"] == "BLOCKED"
    if not ok:
        print("\nFAILED: gate did not produce the expected PROMOTED + BLOCKED pair "
              "(is Postgres reachable? the chain fails closed to BLOCKED when it is not).")
    else:
        print("\nChain reproduced through the LangGraph graph: PROMOTED and BLOCKED, both audited. OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
