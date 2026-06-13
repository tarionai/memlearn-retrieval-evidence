"""Gate tests — the deterministic citability + claim-boundary rule.

One PROMOTED case (frozen, complete run), and two BLOCKED cases (unfrozen
qrels; a 'Not validated' claim requested). The BLOCKED cases are the proof the
gate does work — a run that fails the frozen condition is rejected by rule.
"""
from __future__ import annotations

import pytest

from enterprise_adapter import GovernanceDecision, evaluate_governance
from enterprise_adapter.decision import BLOCKED, PROMOTED, claim_is_allowed

_CLAIM_BOUNDARIES = [
    {"claim": "retrieval over a real anonymized corpus", "status": "Demonstrable"},
    {"claim": "qrels support bounded offline eval", "status": "Demonstrable after annotation"},
    {"claim": "improves recruiter satisfaction in production", "status": "Not validated"},
    {"claim": "PostgreSQL native FTS is BM25", "status": "False; do not claim"},
]


def _complete_report() -> dict:
    return {
        "evaluation_status": "complete",
        "qrels_status": "frozen",
        "run_id": "1554d517",
        "git_commit": "28c16a9",
        "dataset_version": "djinni_v0",
        "embedding_model_id": "all-MiniLM-L6-v2",
        "claim_boundaries": list(_CLAIM_BOUNDARIES),
    }


def test_promoted_on_complete_frozen_run():
    decision = evaluate_governance(_complete_report())
    assert decision.status == PROMOTED
    assert decision.is_promoted
    assert decision.failed_condition is None
    assert decision.eval_run_id == "1554d517"
    assert "retrieval over a real anonymized corpus" in decision.allowed_claims
    assert "improves recruiter satisfaction in production" in decision.forbidden_claims
    # No forbidden claim leaks into the allowed set.
    assert "PostgreSQL native FTS is BM25" not in decision.allowed_claims


def test_blocked_on_unfrozen_qrels():
    report = _complete_report()
    report["qrels_status"] = "not_frozen"
    decision = evaluate_governance(report)
    assert decision.status == BLOCKED
    assert not decision.is_promoted
    assert "qrels_status" in decision.failed_condition
    assert "not_frozen" in decision.failed_condition
    assert decision.allowed_claims == ()


def test_blocked_on_incomplete_evaluation():
    report = _complete_report()
    report["evaluation_status"] = "blocked_pending_human_qrels"
    decision = evaluate_governance(report)
    assert decision.status == BLOCKED
    assert "evaluation_status" in decision.failed_condition


def test_blocked_when_not_validated_claim_requested():
    decision = evaluate_governance(
        _complete_report(),
        requested_claims=["improves recruiter satisfaction in production"],
    )
    assert decision.status == BLOCKED
    assert "not citable" in decision.failed_condition
    assert "recruiter satisfaction" in decision.failed_condition


def test_promoted_when_only_allowed_claims_requested():
    decision = evaluate_governance(
        _complete_report(),
        requested_claims=["retrieval over a real anonymized corpus"],
    )
    assert decision.status == PROMOTED


def test_claim_classification():
    assert claim_is_allowed("Demonstrable")
    assert claim_is_allowed("Demonstrable after annotation")
    assert claim_is_allowed("Demonstrable after inspection")
    assert not claim_is_allowed("Not validated")
    assert not claim_is_allowed("False; do not claim")


def test_decision_is_immutable():
    decision = evaluate_governance(_complete_report())
    # Python 3.10 frozen dataclass raises AttributeError (not FrozenInstanceError).
    with pytest.raises(AttributeError):
        decision.status = "TAMPERED"


def test_decision_to_dict_roundtrip():
    decision = evaluate_governance(_complete_report())
    payload = decision.to_dict()
    assert payload["status"] == PROMOTED
    assert payload["failed_condition"] is None
    assert isinstance(payload["allowed_claims"], list)


def test_evaluate_does_not_mutate_report():
    report = _complete_report()
    before = dict(report)
    evaluate_governance(report, requested_claims=["improves recruiter satisfaction in production"])
    assert report == before
