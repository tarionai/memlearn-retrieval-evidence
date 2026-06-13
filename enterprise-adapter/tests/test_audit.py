"""Audit-writer tests — append-only discipline.

Append two runs; assert ordering is preserved and the first line is byte-for-byte
immutable after the second append.
"""
from __future__ import annotations

from enterprise_adapter import append_audit_record, evaluate_governance, read_audit_log
from enterprise_adapter.audit import RECORD_KIND, build_audit_record

_REPORT = {
    "evaluation_status": "complete",
    "qrels_status": "frozen",
    "run_id": "1554d517",
    "git_commit": "28c16a9",
    "dataset_version": "djinni_v0",
    "embedding_model_id": "all-MiniLM-L6-v2",
    "run_timestamp": "2026-06-09T11:32:18.452931+00:00",
    "claim_boundaries": [{"claim": "retrieval over a real corpus", "status": "Demonstrable"}],
}


def test_append_two_runs_preserves_order_and_immutability(tmp_path):
    log = tmp_path / "audit_log.jsonl"
    decision = evaluate_governance(_REPORT)

    append_audit_record(
        log, decision,
        source_load_report_ref="evidence/neon_load_report.json",
        event_at=_REPORT["run_timestamp"],
        recorded_at="2026-06-13T10:00:00+00:00",
    )
    first_line_after_one = log.read_text(encoding="utf-8").splitlines()[0]

    append_audit_record(
        log, decision,
        source_load_report_ref="evidence/neon_load_report.json",
        event_at=_REPORT["run_timestamp"],
        recorded_at="2026-06-13T11:00:00+00:00",
    )

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # Prior line unchanged by the later append.
    assert lines[0] == first_line_after_one
    # Ordering reflects append order via recorded_at.
    records = read_audit_log(log)
    assert records[0]["recorded_at"] == "2026-06-13T10:00:00+00:00"
    assert records[1]["recorded_at"] == "2026-06-13T11:00:00+00:00"
    assert records[0]["record_kind"] == RECORD_KIND


def test_audit_record_carries_decision_and_lineage(tmp_path):
    log = tmp_path / "audit_log.jsonl"
    decision = evaluate_governance(_REPORT)
    record = append_audit_record(
        log, decision,
        source_load_report_ref="evidence/neon_load_report.json",
        event_at=_REPORT["run_timestamp"],
        recorded_at="2026-06-13T10:00:00+00:00",
    )
    assert record.eval_run_id == "1554d517"
    assert record.dataset_version == "djinni_v0"
    assert record.decision["status"] == "PROMOTED"
    assert record.event_at == _REPORT["run_timestamp"]


def test_build_record_is_pure():
    decision = evaluate_governance(_REPORT)
    args = dict(source_load_report_ref="ref", event_at="2026-06-09T11:32:18+00:00", recorded_at="2026-06-13T10:00:00+00:00")
    one = build_audit_record(decision, **args)
    two = build_audit_record(decision, **args)
    assert one.to_json_line() == two.to_json_line()


def test_default_recorded_at_is_stamped(tmp_path):
    log = tmp_path / "audit_log.jsonl"
    decision = evaluate_governance(_REPORT)
    record = append_audit_record(
        log, decision,
        source_load_report_ref="ref",
        event_at=_REPORT["run_timestamp"],
    )
    # A UTC ISO timestamp was stamped at the boundary.
    assert record.recorded_at.endswith("+00:00")
