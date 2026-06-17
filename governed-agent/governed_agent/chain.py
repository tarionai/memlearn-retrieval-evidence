"""Single import surface over the published enterprise-adapter chain.

The graph and the MCP server call exactly one copy of the chain logic: the
functions published in the cold-verified reproduction pack. Nothing here is
rewritten. We inject the adapter root and the reproduction directory onto
sys.path (the same convention the repo itself uses — it is intentionally not an
installable package) and re-export the verified functions.

Dependency direction is one-way: this new code depends on the stable published
modules, never the reverse. `reproduce.py` is left byte-identical so its
cold-verified SHA stays valid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg2

_ADAPTER_ROOT = Path(__file__).resolve().parents[2] / "enterprise-adapter"
_REPRO_DIR = _ADAPTER_ROOT / "reproduction"
for _candidate in (_ADAPTER_ROOT, _REPRO_DIR):
    _entry = str(_candidate)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# Published gate + audit — reused verbatim, never reimplemented.
from enterprise_adapter import (  # noqa: E402
    CITABILITY_RULE,
    AuditRecord,
    GovernanceDecision,
    append_audit_record,
    evaluate_governance,
    read_audit_log,
)

# Published chain functions from the reproduction module (one copy of truth).
from reproduce import (  # noqa: E402
    DEFAULT_CONN,
    OVERREACH_CLAIM,
    SCHEMA,
    build_report,
    ingest,
    retrieve_fts,
)
from repro_metrics import aggregate, query_metrics  # noqa: E402

SYNTHETIC_CORPUS_PATH = _REPRO_DIR / "synthetic_corpus.json"
# The published, cold-verified frozen eval report. check_claim_citable runs the
# gate over this with no database — the "candidate before promotion" demo.
FROZEN_REPORT_PATH = _ADAPTER_ROOT / "evidence" / "candidate_search_eval_report.json"


def load_frozen_report() -> dict:
    """Load the published frozen eval report the gate was cold-verified against."""
    return json.loads(FROZEN_REPORT_PATH.read_text(encoding="utf-8"))


def connect(dsn: str):
    """Open a psycopg2 connection. The single declared DB-I/O boundary."""
    return psycopg2.connect(dsn, connect_timeout=15)


def load_synthetic_corpus() -> dict:
    """Load the frozen synthetic corpus the reproduction pack ships."""
    return json.loads(SYNTHETIC_CORPUS_PATH.read_text(encoding="utf-8"))


__all__ = [
    "CITABILITY_RULE",
    "AuditRecord",
    "GovernanceDecision",
    "append_audit_record",
    "evaluate_governance",
    "read_audit_log",
    "build_report",
    "ingest",
    "retrieve_fts",
    "aggregate",
    "query_metrics",
    "connect",
    "load_synthetic_corpus",
    "load_frozen_report",
    "DEFAULT_CONN",
    "OVERREACH_CLAIM",
    "SCHEMA",
    "SYNTHETIC_CORPUS_PATH",
    "FROZEN_REPORT_PATH",
]
