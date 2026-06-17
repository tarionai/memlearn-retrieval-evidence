"""Phase 1 acceptance tests for the LangGraph governed-run graph.

Proven by execution, not asserted: happy-path PROMOTED, verdict-conditional
BLOCKED, durable checkpoint recovery, and fail-closed fallback when the DB is
down. The gate and audit are the published modules — verified by identity.
"""
from __future__ import annotations

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import enterprise_adapter
from governed_agent import chain, graph
from governed_agent.graph import build_graph
from conftest import DEAD_DSN


def _invoke(app, state, thread_id="t-default"):
    return app.invoke(state, {"configurable": {"thread_id": thread_id}})


# --- reuse-verbatim proof: these are the published modules, not copies ---

def test_gate_and_audit_are_the_published_modules():
    assert chain.evaluate_governance is enterprise_adapter.evaluate_governance
    assert chain.append_audit_record is enterprise_adapter.append_audit_record
    assert graph.chain.evaluate_governance is enterprise_adapter.evaluate_governance


# --- (a) happy path -> PROMOTED + exactly one audit line ---

def test_happy_path_promoted_with_audit(live_dsn, base_state):
    app = build_graph()
    out = _invoke(app, {**base_state, "dsn": live_dsn})
    assert out["result"]["decision"] == "PROMOTED"
    assert len(out["result"]["allowed_claims"]) >= 1
    assert out["result"]["failed_condition"] is None
    records = chain.read_audit_log(base_state["audit_path"])
    assert len(records) == 1
    assert records[0]["decision"]["status"] == "PROMOTED"


# --- (b) a Not-validated claim is requested -> BLOCKED naming the condition ---

def test_not_validated_claim_blocked(live_dsn, base_state):
    app = build_graph()
    out = _invoke(app, {**base_state, "dsn": live_dsn, "requested_claims": [chain.OVERREACH_CLAIM]})
    assert out["result"]["decision"] == "BLOCKED"
    assert "not citable" in out["result"]["failed_condition"]
    assert chain.OVERREACH_CLAIM in out["result"]["failed_condition"]
    assert out["result"]["allowed_claims"] == []
    records = chain.read_audit_log(base_state["audit_path"])
    assert records[0]["decision"]["status"] == "BLOCKED"


# --- (c) durable execution: interrupt after evaluate, resume without re-ingest ---

def test_checkpoint_recovery_does_not_reingest(live_dsn, base_state, tmp_path, monkeypatch):
    calls = {"n": 0}
    real_ingest = chain.ingest

    def counting_ingest(conn, corpus):
        calls["n"] += 1
        return real_ingest(conn, corpus)

    monkeypatch.setattr(graph.chain, "ingest", counting_ingest)

    cpfile = tmp_path / "cp.sqlite"
    cfg = {"configurable": {"thread_id": "recover-1"}}
    state = {**base_state, "dsn": live_dsn}

    conn1 = sqlite3.connect(str(cpfile), check_same_thread=False)
    app1 = build_graph(SqliteSaver(conn1), interrupt_after=["evaluate"])
    paused = app1.invoke(state, cfg)
    conn1.close()  # simulate process death after evaluate
    assert "report" in paused and "result" not in paused
    assert calls["n"] == 1

    conn2 = sqlite3.connect(str(cpfile), check_same_thread=False)
    app2 = build_graph(SqliteSaver(conn2))
    resumed = app2.invoke(None, cfg)  # None input = resume from checkpoint
    conn2.close()
    assert resumed["result"]["decision"] == "PROMOTED"
    assert calls["n"] == 1  # ingest never re-ran


# --- (d) DB down after retries -> fail closed to BLOCKED, audited, no crash ---

def test_db_down_fails_closed_to_blocked(base_state):
    app = build_graph()
    out = app.invoke({**base_state, "dsn": DEAD_DSN})
    assert out["result"]["decision"] == "BLOCKED"
    assert "OperationalError" in out["result"]["failed_condition"] or "failed" in out["result"]["failed_condition"]
    # Fail-closed verdict carries no real lineage — nothing was evaluated.
    assert out["result"]["lineage"]["eval_run_id"] == "unknown"
    records = chain.read_audit_log(base_state["audit_path"])
    assert len(records) == 1
    assert records[0]["decision"]["status"] == "BLOCKED"


def test_db_down_retries_three_times(base_state, monkeypatch):
    attempts = {"n": 0}
    real_connect = chain.connect

    def counting_connect(dsn):
        attempts["n"] += 1
        return real_connect(dsn)

    monkeypatch.setattr(graph.chain, "connect", counting_connect)
    app = build_graph()
    app.invoke({**base_state, "dsn": DEAD_DSN})
    assert attempts["n"] == 3  # RetryPolicy max_attempts on the ingest node
