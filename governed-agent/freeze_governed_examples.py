"""Freeze the governed-agent evidence, deterministically.

Re-running this reproduces byte-identical JSON + audit lines (timestamps are
fixed constants, not clock reads), so SHA256SUMS.txt stays stable — the same
discipline as the adapter's freeze_examples.py.

The frozen artifacts are the graph's terminal governed outputs (the emit-node
payloads) and the append-only audit, computed over the published, cold-verified
frozen eval report. Deterministic mode is what gets frozen; the live-DB graph
run (timestamps + git commit vary) is demonstrated by the CLI and the tests.

    python governed-agent/freeze_governed_examples.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from governed_agent import chain
from governed_agent.graph import emit_blocked_node, emit_promoted_node

EVIDENCE = Path(__file__).resolve().parent / "evidence"
LOAD_REPORT_REF = "enterprise-adapter/evidence/neon_load_report.json"

# Fixed freeze timestamps — constants so the frozen artifacts are reproducible.
FREEZE_PROMOTED_AT = "2026-06-17T00:00:00+00:00"
FREEZE_BLOCKED_AT = "2026-06-17T00:00:01+00:00"

FROZEN_FILES = (
    "governed_run_promoted.json",
    "governed_run_blocked.json",
    "audit_log.jsonl",
)


def _overreach_claim(report: dict) -> str:
    return next(b["claim"] for b in report["claim_boundaries"] if b["status"] == "Not validated")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _normalize_lf(path: Path) -> None:
    """Force LF line endings so hashes are stable across platforms. The published
    audit writer uses text-mode open (CRLF on Windows); normalize to match Linux."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    path.write_bytes(data)


def _write_sha256sums() -> None:
    lines = []
    for name in FROZEN_FILES:
        _normalize_lf(EVIDENCE / name)
        digest = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (EVIDENCE / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    EVIDENCE.mkdir(exist_ok=True)
    report = chain.load_frozen_report()
    event_at = report["run_timestamp"]
    overreach = _overreach_claim(report)

    promoted = chain.evaluate_governance(report)
    assert promoted.is_promoted, "frozen report must promote"
    blocked = chain.evaluate_governance(report, requested_claims=[overreach])
    assert not blocked.is_promoted, "over-reach claim must block"

    # Reuse the graph's emit-node logic verbatim — these are the graph outputs.
    _write_json(EVIDENCE / "governed_run_promoted.json", emit_promoted_node({"decision": promoted.to_dict()})["result"])
    _write_json(EVIDENCE / "governed_run_blocked.json", emit_blocked_node({"decision": blocked.to_dict()})["result"])

    log = EVIDENCE / "audit_log.jsonl"
    if log.exists():
        log.unlink()
    chain.append_audit_record(log, promoted, source_load_report_ref=LOAD_REPORT_REF, event_at=event_at, recorded_at=FREEZE_PROMOTED_AT)
    chain.append_audit_record(log, blocked, source_load_report_ref=LOAD_REPORT_REF, event_at=event_at, recorded_at=FREEZE_BLOCKED_AT)

    _write_sha256sums()
    print(f"PROMOTED: {len(promoted.allowed_claims)} allowed claims")
    print(f"BLOCKED:  {blocked.failed_condition}")
    print(f"froze {len(FROZEN_FILES)} files + SHA256SUMS.txt under {EVIDENCE}")


if __name__ == "__main__":
    main()
