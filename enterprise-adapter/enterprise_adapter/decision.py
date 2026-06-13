"""Typed governed-output primitives.

A GovernanceDecision is the terminal artifact of the adapter chain: a
deterministic verdict over a frozen evaluation report. It never carries raw
corpus rows, connection strings, or reviewer PII — only lineage identifiers
and the citability verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROMOTED = "PROMOTED"
BLOCKED = "BLOCKED"

# Claim-boundary status vocabulary, mirrored verbatim from the frozen eval
# contract. A claim is citable only if its status begins with "Demonstrable".
_FORBIDDEN_EXACT = frozenset({"Not validated"})


def claim_is_allowed(status: str) -> bool:
    """A claim may be cited only when its boundary status is a Demonstrable kind.

    'Not validated' and 'False; do not claim' are never citable.
    """
    if status in _FORBIDDEN_EXACT:
        return False
    if status.startswith("False"):
        return False
    return status.startswith("Demonstrable")


@dataclass(frozen=True)
class ClaimBoundary:
    """One claim and its frozen citability status."""

    claim: str
    status: str

    @property
    def is_allowed(self) -> bool:
        return claim_is_allowed(self.status)

    def to_dict(self) -> dict:
        return {"claim": self.claim, "status": self.status, "is_allowed": self.is_allowed}


@dataclass(frozen=True)
class GovernanceDecision:
    """Deterministic verdict over one frozen evaluation report.

    status is PROMOTED only when the citability rule holds and no forbidden
    claim was requested. failed_condition names the single rule that failed
    when BLOCKED, and is None when PROMOTED.
    """

    status: str
    eval_run_id: str
    git_commit: str
    dataset_version: str
    embedding_model_id: str
    allowed_claims: tuple = field(default_factory=tuple)
    forbidden_claims: tuple = field(default_factory=tuple)
    failed_condition: str | None = None

    @property
    def is_promoted(self) -> bool:
        return self.status == PROMOTED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "eval_run_id": self.eval_run_id,
            "git_commit": self.git_commit,
            "dataset_version": self.dataset_version,
            "embedding_model_id": self.embedding_model_id,
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "failed_condition": self.failed_condition,
        }
