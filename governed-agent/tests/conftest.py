"""Shared fixtures. DB-backed tests skip cleanly when no Postgres is reachable,
so the suite is safe to run in CI without a database; the DB-down fail-closed
test and the gate-reuse tests always run."""
from __future__ import annotations

import os

import pytest

from governed_agent import chain

LIVE_DSN = os.environ.get("REPRO_PG_CONN", "postgresql://postgres:postgres@localhost:5432/postgres")
DEAD_DSN = "postgresql://postgres:postgres@localhost:1/postgres"


@pytest.fixture(scope="session")
def live_dsn() -> str:
    try:
        conn = chain.connect(LIVE_DSN)
        conn.close()
    except Exception:  # noqa: BLE001
        pytest.skip("no Postgres reachable for DB-backed tests")
    return LIVE_DSN


@pytest.fixture
def base_state(tmp_path):
    return {
        "corpus": chain.load_synthetic_corpus(),
        "query": "platform engineer kubernetes terraform module authoring helm argocd",
        "top_k": 20,
        "mode": "deterministic",
        "requested_claims": [],
        "audit_path": str(tmp_path / "audit.jsonl"),
        "source_load_report_ref": "synthetic_v0",
    }
