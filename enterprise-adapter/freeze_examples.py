"""Generate the frozen governed-output examples + audit log, deterministically.

Re-running this against the vendored eval report reproduces byte-identical
JSON (timestamps are fixed constants, not clock reads), so the SHA256SUMS in
this directory stay stable. Run from the repo root:

    python enterprise-adapter/freeze_examples.py

Outputs (under enterprise-adapter/evidence/):
    governed_output_promoted.json   PROMOTED — frozen complete run, allowed claims
    governed_output_blocked.json    BLOCKED  — a 'Not validated' claim was requested
    audit_log.jsonl                 append-only record of both governance runs
"""
from __future__ import annotations

import json
from pathlib import Path

from enterprise_adapter import append_audit_record, evaluate_governance, load_report

EVIDENCE = Path(__file__).resolve().parent / "evidence"
EVAL_REPORT = EVIDENCE / "candidate_search_eval_report.json"
LOAD_REPORT_REF = "enterprise-adapter/evidence/neon_load_report.json"

# Fixed freeze timestamps — constants so the frozen artifacts are reproducible.
FREEZE_PROMOTED_AT = "2026-06-13T00:00:00+00:00"
FREEZE_BLOCKED_AT = "2026-06-13T00:00:01+00:00"

# An over-reach claim a skeptic might try to cite. Verbatim from claim_boundaries,
# status "Not validated" — the gate must reject it.
OVERREACH_CLAIM = "The system improves recruiter satisfaction in production"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    report = load_report(EVAL_REPORT)
    event_at = report["run_timestamp"]

    promoted = evaluate_governance(report)
    assert promoted.is_promoted, "vendored eval report must promote"
    _write_json(EVIDENCE / "governed_output_promoted.json", promoted.to_dict())

    blocked = evaluate_governance(report, requested_claims=[OVERREACH_CLAIM])
    assert not blocked.is_promoted, "over-reach claim must block"
    _write_json(EVIDENCE / "governed_output_blocked.json", blocked.to_dict())

    log = EVIDENCE / "audit_log.jsonl"
    if log.exists():
        log.unlink()  # regenerate cleanly; append-only semantics are tested separately
    append_audit_record(
        log, promoted, source_load_report_ref=LOAD_REPORT_REF,
        event_at=event_at, recorded_at=FREEZE_PROMOTED_AT,
    )
    append_audit_record(
        log, blocked, source_load_report_ref=LOAD_REPORT_REF,
        event_at=event_at, recorded_at=FREEZE_BLOCKED_AT,
    )

    print(f"PROMOTED: run {promoted.eval_run_id}, {len(promoted.allowed_claims)} allowed claims")
    print(f"BLOCKED:  {blocked.failed_condition}")
    print(f"audit log: {log}")


if __name__ == "__main__":
    main()
