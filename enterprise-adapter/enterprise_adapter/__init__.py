"""Enterprise data adapter — governed-output gate and append-only audit record.

Public surface:
    evaluate_governance(report, requested_claims=None) -> GovernanceDecision
    GovernanceDecision, ClaimBoundary
    AuditRecord, append_audit_record
"""
from .audit import AuditRecord, append_audit_record, read_audit_log
from .decision import ClaimBoundary, GovernanceDecision
from .gate import CITABILITY_RULE, evaluate_governance, load_report

__all__ = [
    "evaluate_governance",
    "load_report",
    "CITABILITY_RULE",
    "GovernanceDecision",
    "ClaimBoundary",
    "AuditRecord",
    "append_audit_record",
    "read_audit_log",
]
