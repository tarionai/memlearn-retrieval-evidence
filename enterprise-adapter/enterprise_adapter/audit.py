"""Append-only audit record for one adapter chain run.

Mirrors the memlearn snowflake-export / state-mutation-log discipline: records
are only ever appended as JSONL lines, never rewritten in place. Each line
captures the full chain — source-load lineage, retrieval/eval run ids, the
GovernanceDecision, and two timestamps:

    event_at    — business time: when the evaluated run was produced (UTC).
    recorded_at — wall-clock time: when this audit line was written (UTC).

The clock read is confined to append_audit_record (the boundary). build_audit_record
is pure: same inputs always produce the same record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .decision import GovernanceDecision

RECORD_KIND = "adapter_chain_run"


@dataclass(frozen=True)
class AuditRecord:
    """One immutable line in the chain audit log."""

    record_kind: str
    event_at: str
    recorded_at: str
    source_load_report_ref: str
    eval_run_id: str
    git_commit: str
    dataset_version: str
    embedding_model_id: str
    decision: dict

    def to_dict(self) -> dict:
        return {
            "record_kind": self.record_kind,
            "event_at": self.event_at,
            "recorded_at": self.recorded_at,
            "source_load_report_ref": self.source_load_report_ref,
            "eval_run_id": self.eval_run_id,
            "git_commit": self.git_commit,
            "dataset_version": self.dataset_version,
            "embedding_model_id": self.embedding_model_id,
            "decision": self.decision,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def build_audit_record(
    decision: GovernanceDecision,
    *,
    source_load_report_ref: str,
    event_at: str,
    recorded_at: str,
) -> AuditRecord:
    """Pure constructor — no clock, no I/O."""
    return AuditRecord(
        record_kind=RECORD_KIND,
        event_at=event_at,
        recorded_at=recorded_at,
        source_load_report_ref=source_load_report_ref,
        eval_run_id=decision.eval_run_id,
        git_commit=decision.git_commit,
        dataset_version=decision.dataset_version,
        embedding_model_id=decision.embedding_model_id,
        decision=decision.to_dict(),
    )


def append_audit_record(
    path: str | Path,
    decision: GovernanceDecision,
    *,
    source_load_report_ref: str,
    event_at: str,
    recorded_at: str | None = None,
) -> AuditRecord:
    """Append one record as a JSONL line. Never rewrites prior lines."""
    if recorded_at is None:
        recorded_at = datetime.now(timezone.utc).isoformat()
    record = build_audit_record(
        decision,
        source_load_report_ref=source_load_report_ref,
        event_at=event_at,
        recorded_at=recorded_at,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json_line() + "\n")
    return record


def read_audit_log(path: str | Path) -> list[dict]:
    """Read all audit lines in append order."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
