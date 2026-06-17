"""LangGraph state graph over the governed-retrieval chain.

Every node wraps a verified function from the published chain. The only
control flow that is *not* a straight line is the part that matters: after the
deterministic gate produces a verdict, a conditional edge branches —
PROMOTED emits the allowed-claims set, BLOCKED routes to a first-class
fallback node that emits a safe message and still writes an audit line. A
governance failure is never a silent drop, and an infrastructure failure
(DB down after retries) fails closed onto the same BLOCKED terminal.

Two modes, one graph:
  - "deterministic" (default): ingest -> retrieve -> evaluate -> gate -> emit -> audit.
    No LLM, no API key, cold-reproducible.
  - "agentic": inserts a draft node before the gate. An LLM drafts a grounded
    answer and proposes claims; the proposed claims must clear the same
    deterministic gate before anything is emitted. LLM proposes, gate disposes.
"""
from __future__ import annotations

import os
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

import psycopg2

from . import chain

# Model for the agentic draft node. Set once here, never at call sites.
DRAFT_MODEL = "claude-sonnet-4-6"
DRAFT_MODEL_CHEAP = "claude-haiku-4-5-20251001"

# Transient DB failures worth retrying; a dead port surfaces as OperationalError.
_TRANSIENT_DB_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)
_DB_RETRY = RetryPolicy(
    max_attempts=3, initial_interval=0.5, backoff_factor=2.0, retry_on=_TRANSIENT_DB_ERRORS
)

# Side channel so the per-node error_handler can name the real DB error. A node
# cannot both return state and re-raise to trigger a retry, so the message is
# stashed here on the way out and read back when retries are exhausted.
_LAST_DB_ERROR: dict[str, str] = {}


class GovernedRunState(TypedDict, total=False):
    """Explicit, serializable run state — no live connections, so it checkpoints."""

    dsn: str
    corpus: dict
    query: str
    top_k: int
    requested_claims: list[str]
    proposed_claims: list[str]
    mode: str
    source_load_report_ref: str
    audit_path: str
    retrieved_ids: list[str]
    report: dict
    draft: str | None
    decision: dict
    result: dict
    audit_ref: str | None
    error: str | None


# --- DB nodes (wrap verified chain functions; each owns its own connection) ---

def _record_db_error(node: str, exc: Exception) -> None:
    _LAST_DB_ERROR[node] = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:200]


def ingest_node(state: GovernedRunState) -> dict:
    try:
        conn = chain.connect(state["dsn"])
        try:
            chain.ingest(conn, state["corpus"])
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - re-raised after recording for retry
        _record_db_error("ingest", exc)
        raise
    return {}


def retrieve_node(state: GovernedRunState) -> dict:
    try:
        conn = chain.connect(state["dsn"])
        try:
            ids = chain.retrieve_fts(conn, state["query"], state.get("top_k", 20))
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        _record_db_error("retrieve", exc)
        raise
    return {"retrieved_ids": ids}


def evaluate_node(state: GovernedRunState) -> dict:
    try:
        conn = chain.connect(state["dsn"])
        try:
            report = chain.build_report(conn, state["corpus"])
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        _record_db_error("evaluate", exc)
        raise
    return {"report": report}


def _db_error_handler(node: str):
    """Fail-closed: on retry exhaustion, divert in-graph to the BLOCKED terminal."""

    def handler(state: GovernedRunState) -> Command:
        message = _LAST_DB_ERROR.get(node, f"{node} failed after retries (transient DB failure)")
        return Command(goto="emit_blocked", update={"error": message})

    return handler


# --- Agentic draft node: LLM proposes, gate disposes ---

def draft_node(state: GovernedRunState) -> dict:
    """Draft a grounded answer from retrieved candidates and propose claims.

    Imported lazily so deterministic mode never needs the anthropic SDK or a key.
    The proposed claims are merged into the gate's requested_claims — they only
    survive to the client if the deterministic gate promotes them.
    """
    from anthropic import Anthropic  # noqa: PLC0415 - optional dependency

    report = state.get("report", {})
    candidate_ids = state.get("retrieved_ids", [])
    boundaries = report.get("claim_boundaries", [])
    allowed = [b["claim"] for b in boundaries if b.get("status", "").startswith("Demonstrable")]

    client = Anthropic()
    model = os.environ.get("GOVERNED_AGENT_DRAFT_MODEL", DRAFT_MODEL)
    prompt = (
        "You are drafting a grounded summary of a retrieval run. Use ONLY the "
        "facts below. Then list, one per line prefixed with 'CLAIM:', the claims "
        "you assert. Do not invent claims beyond what the evidence supports.\n\n"
        f"Retrieved candidate ids: {candidate_ids[:20]}\n"
        f"Citable claim boundaries: {allowed}\n"
    )
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    proposed = [
        line.split("CLAIM:", 1)[1].strip()
        for line in text.splitlines()
        if "CLAIM:" in line
    ]
    return {"draft": text, "proposed_claims": proposed}


# --- Gate + emit + audit (gate and audit are published modules, verbatim) ---

def gate_node(state: GovernedRunState) -> dict:
    requested = list(state.get("requested_claims") or []) + list(state.get("proposed_claims") or [])
    decision = chain.evaluate_governance(state["report"], requested_claims=requested)
    return {"decision": decision.to_dict()}


def _lineage(decision: dict) -> dict:
    return {
        "eval_run_id": decision["eval_run_id"],
        "git_commit": decision["git_commit"],
        "dataset_version": decision["dataset_version"],
        "embedding_model_id": decision["embedding_model_id"],
    }


def emit_promoted_node(state: GovernedRunState) -> dict:
    decision = state["decision"]
    return {
        "result": {
            "decision": "PROMOTED",
            "allowed_claims": decision["allowed_claims"],
            "forbidden_claims": decision["forbidden_claims"],
            "failed_condition": None,
            "lineage": _lineage(decision),
            "draft": state.get("draft"),
        }
    }


def _synthetic_blocked(error: str) -> dict:
    """Fail-closed verdict when the chain never reached the gate (e.g. DB down)."""
    return chain.GovernanceDecision(
        status="BLOCKED",
        eval_run_id="unknown",
        git_commit="unknown",
        dataset_version="unknown",
        embedding_model_id="unknown",
        failed_condition=error,
    ).to_dict()


def emit_blocked_node(state: GovernedRunState) -> dict:
    decision = state.get("decision")
    if not decision:
        decision = _synthetic_blocked(state.get("error") or "unknown failure")
    return {
        "decision": decision,
        "result": {
            "decision": "BLOCKED",
            "allowed_claims": [],
            "forbidden_claims": decision["forbidden_claims"],
            "failed_condition": decision["failed_condition"],
            "lineage": _lineage(decision),
            "draft": state.get("draft"),
        },
    }


def _decision_from_dict(decision: dict) -> chain.GovernanceDecision:
    return chain.GovernanceDecision(
        status=decision["status"],
        eval_run_id=decision["eval_run_id"],
        git_commit=decision["git_commit"],
        dataset_version=decision["dataset_version"],
        embedding_model_id=decision["embedding_model_id"],
        allowed_claims=tuple(decision["allowed_claims"]),
        forbidden_claims=tuple(decision["forbidden_claims"]),
        failed_condition=decision["failed_condition"],
    )


def audit_node(state: GovernedRunState) -> dict:
    """Append-only audit on BOTH paths. Published module, verbatim."""
    decision = _decision_from_dict(state["decision"])
    report = state.get("report") or {}
    event_at = report.get("run_timestamp")
    if event_at is None:
        from datetime import datetime, timezone

        event_at = datetime.now(timezone.utc).isoformat()
    record = chain.append_audit_record(
        state["audit_path"],
        decision,
        source_load_report_ref=state.get("source_load_report_ref", "synthetic_v0"),
        event_at=event_at,
    )
    return {"audit_ref": record.recorded_at}


# --- Routing ---

def _after_evaluate(state: GovernedRunState) -> str:
    return "draft" if state.get("mode") == "agentic" else "gate"


def _after_gate(state: GovernedRunState) -> str:
    decision = state.get("decision") or {}
    return "emit_promoted" if decision.get("status") == "PROMOTED" else "emit_blocked"


def build_graph(checkpointer=None, interrupt_after=None):
    """Compile the governed-run graph. Pass a checkpointer for durable execution.

    interrupt_after names nodes to pause after (used to prove crash recovery:
    pause after "evaluate", then resume on the same thread_id without re-running
    the earlier nodes).
    """
    builder = StateGraph(GovernedRunState)

    builder.add_node("ingest", ingest_node, retry_policy=_DB_RETRY, error_handler=_db_error_handler("ingest"))
    builder.add_node("retrieve", retrieve_node, retry_policy=_DB_RETRY, error_handler=_db_error_handler("retrieve"))
    builder.add_node("evaluate", evaluate_node, retry_policy=_DB_RETRY, error_handler=_db_error_handler("evaluate"))
    builder.add_node("draft", draft_node)
    builder.add_node("gate", gate_node)
    builder.add_node("emit_promoted", emit_promoted_node)
    builder.add_node("emit_blocked", emit_blocked_node)
    builder.add_node("audit", audit_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "retrieve")
    builder.add_edge("retrieve", "evaluate")
    builder.add_conditional_edges("evaluate", _after_evaluate, {"draft": "draft", "gate": "gate"})
    builder.add_edge("draft", "gate")
    builder.add_conditional_edges(
        "gate", _after_gate, {"emit_promoted": "emit_promoted", "emit_blocked": "emit_blocked"}
    )
    builder.add_edge("emit_promoted", "audit")
    builder.add_edge("emit_blocked", "audit")
    builder.add_edge("audit", END)

    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after or [])
