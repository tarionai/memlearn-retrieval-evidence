"""MCP server — the governed chain as a product surface.

Three well-typed tools over the same gate + audit. The tool surface is the
product: every tool returns typed output, the audit record id is part of the
return contract, and no tool ever leaks raw corpus rows or the DSN — the same
data-handling discipline the GovernanceDecision enforces.

    python -m governed_agent.mcp_server        # stdio transport

  - run_governed_retrieval : run the full graph; returns the verdict + audit id
  - check_claim_citable    : run the gate over one claim (no DB) — candidate-before-promotion
  - read_audit_log         : surface the append-only audit
"""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from . import chain
from .run import corpus_run_id
from .graph import build_graph

DSN = os.environ.get("GOVERNED_AGENT_DSN", os.environ.get("REPRO_PG_CONN", chain.DEFAULT_CONN))
AUDIT_PATH = Path(__file__).resolve().parent.parent / "_output" / "mcp_audit_log.jsonl"

mcp = FastMCP("governed-retrieval")


class Lineage(BaseModel):
    eval_run_id: str
    git_commit: str
    dataset_version: str
    embedding_model_id: str


class GovernedResult(BaseModel):
    """Verdict over a governed retrieval run. No raw rows, no DSN — by contract."""

    decision: str = Field(description="PROMOTED or BLOCKED")
    allowed_claims: list[str]
    forbidden_claims: list[str]
    failed_condition: str | None
    lineage: Lineage
    retrieved_count: int = Field(description="Number of candidate ids retrieved (ids withheld)")
    metrics: dict = Field(default_factory=dict, description="Aggregate FTS retrieval metrics")
    audit_id: str | None = Field(description="recorded_at of the appended audit line")


class CitableResult(BaseModel):
    claim: str
    citable: bool
    reason: str
    rule: str


class AuditLine(BaseModel):
    recorded_at: str
    event_at: str
    decision_status: str
    eval_run_id: str
    failed_condition: str | None


@mcp.tool()
def run_governed_retrieval(query: str, requested_claims: list[str] = []) -> GovernedResult:
    """Run the governed retrieval graph for a query and return the gate verdict.

    Requested claims are checked against the deterministic citability gate; any
    'Not validated' claim blocks the run. The audit id is part of the return.
    """
    corpus = chain.load_synthetic_corpus()
    AUDIT_PATH.parent.mkdir(exist_ok=True)
    state = {
        "dsn": DSN,
        "corpus": corpus,
        "query": query,
        "top_k": 20,
        "mode": "deterministic",
        "requested_claims": requested_claims,
        "audit_path": str(AUDIT_PATH),
        "source_load_report_ref": "synthetic_v0",
    }
    app = build_graph()
    out = app.invoke(state, {"configurable": {"thread_id": f"mcp-{corpus_run_id(corpus)}"}})
    result = out["result"]
    report = out.get("report") or {}
    metrics = report.get("metrics_by_variant", {}).get("fts", {})
    return GovernedResult(
        decision=result["decision"],
        allowed_claims=result["allowed_claims"],
        forbidden_claims=result["forbidden_claims"],
        failed_condition=result["failed_condition"],
        lineage=Lineage(**result["lineage"]),
        retrieved_count=len(out.get("retrieved_ids", [])),
        metrics=metrics,
        audit_id=out.get("audit_ref"),
    )


@mcp.tool()
def check_claim_citable(claim: str) -> CitableResult:
    """Is this claim citable under the frozen evidence? Runs the gate, no database.

    The candidate-before-promotion rule as a single call: a claim marked
    'Not validated' (or worse) in the frozen report is rejected, with the
    failed rule named.
    """
    report = chain.load_frozen_report()
    decision = chain.evaluate_governance(report, requested_claims=[claim])
    citable = decision.is_promoted
    reason = "passes the citability gate" if citable else (decision.failed_condition or "blocked")
    return CitableResult(claim=claim, citable=citable, reason=reason, rule=chain.CITABILITY_RULE)


@mcp.tool()
def read_audit_log() -> list[AuditLine]:
    """Return the append-only audit log for this server's governed runs."""
    if not AUDIT_PATH.exists():
        return []
    lines = []
    for record in chain.read_audit_log(AUDIT_PATH):
        decision = record.get("decision", {})
        lines.append(
            AuditLine(
                recorded_at=record["recorded_at"],
                event_at=record["event_at"],
                decision_status=decision.get("status", "unknown"),
                eval_run_id=decision.get("eval_run_id", "unknown"),
                failed_condition=decision.get("failed_condition"),
            )
        )
    return lines


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
