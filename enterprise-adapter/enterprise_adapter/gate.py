"""Governed-output gate — a pure deterministic rule over a frozen eval report.

The rule is the eval contract's citability condition, made executable:

    A run is citable iff evaluation_status == "complete"
                     and qrels_status == "frozen".

A run that passes, with no forbidden claim requested, is PROMOTED with its
allowed-claims set. Any failure — unfrozen qrels, incomplete evaluation, or an
attempt to cite a "Not validated" claim — is BLOCKED with the failed condition
named. No agent, no inference: the same input always yields the same verdict.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .decision import BLOCKED, PROMOTED, ClaimBoundary, GovernanceDecision

CITABILITY_RULE = "evaluation_status == 'complete' AND qrels_status == 'frozen'"


def load_report(path: str | Path) -> dict:
    """Read a frozen eval report from disk. The only I/O in this module."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _classify_claims(report: dict) -> tuple[list[ClaimBoundary], list[ClaimBoundary]]:
    boundaries = [ClaimBoundary(b["claim"], b["status"]) for b in report.get("claim_boundaries", [])]
    allowed = [b for b in boundaries if b.is_allowed]
    forbidden = [b for b in boundaries if not b.is_allowed]
    return allowed, forbidden


def _lineage(report: dict) -> dict:
    return {
        "eval_run_id": str(report.get("run_id", "unknown")),
        "git_commit": str(report.get("git_commit", "unknown")),
        "dataset_version": str(report.get("dataset_version", "unknown")),
        "embedding_model_id": str(report.get("embedding_model_id", "unknown")),
    }


def _blocked(report: dict, failed_condition: str, forbidden: Sequence[ClaimBoundary]) -> GovernanceDecision:
    return GovernanceDecision(
        status=BLOCKED,
        failed_condition=failed_condition,
        forbidden_claims=tuple(b.claim for b in forbidden),
        **_lineage(report),
    )


def evaluate_governance(
    report: dict, requested_claims: Sequence[str] | None = None
) -> GovernanceDecision:
    """Pure verdict over a frozen eval report. Never mutates the report."""
    allowed, forbidden = _classify_claims(report)

    evaluation_status = report.get("evaluation_status")
    if evaluation_status != "complete":
        return _blocked(report, f"evaluation_status == '{evaluation_status}' (required: 'complete')", forbidden)

    qrels_status = report.get("qrels_status")
    if qrels_status != "frozen":
        return _blocked(report, f"qrels_status == '{qrels_status}' (required: 'frozen')", forbidden)

    forbidden_text = {b.claim for b in forbidden}
    for requested in requested_claims or []:
        if requested in forbidden_text:
            return _blocked(report, f"requested claim is not citable: {requested!r}", forbidden)

    return GovernanceDecision(
        status=PROMOTED,
        allowed_claims=tuple(b.claim for b in allowed),
        forbidden_claims=tuple(b.claim for b in forbidden),
        **_lineage(report),
    )


def evaluate_report_file(
    path: str | Path, requested_claims: Sequence[str] | None = None
) -> GovernanceDecision:
    """Convenience: load a report from disk and evaluate it."""
    return evaluate_governance(load_report(path), requested_claims)
