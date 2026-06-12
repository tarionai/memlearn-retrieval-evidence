"""
Primitive validation test suite for WP-02.

Tests every type in memlearn.primitives and memlearn.errors against the
invariants stated in IMPL_WP02_primitive_type_definitions.md §6 (N-04).

Runtime: Python 3.10 — frozen dataclass mutation raises AttributeError,
not FrozenInstanceError (which is 3.11+).
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest

from memlearn.errors import (
    BudgetExceeded,
    EmbeddingDimensionMismatch,
    InvalidLaneId,
    MemLearnError,
    StaleEmbeddingModel,
)
from memlearn.services.associative_state_engine import AssociativeState
from memlearn.primitives import (
    AdmissionConfig,
    AgeDriftPolicy,
    ConsolidationReport,
    DeltaClampPolicy,
    EmbeddingModelRef,
    ExplicitKeyPolicy,
    ExtractionResult,
    ForgetPolicy,
    ForgetPolicyKind,
    ForgetReport,
    ForgettingAuditReport,
    ImportanceDecayPolicy,
    InterferencePrunePolicy,
    LaneId,
    LearningReport,
    MemoryContext,
    MemoryRecord,
    MemorySource,
    RetrievalEffectivenessReport,
    SnapshotHandle,
    TaskScopeExpiryPolicy,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _model_ref(dim: int = 4) -> EmbeddingModelRef:
    return EmbeddingModelRef(model_id="test/model", version="1.0.0", dimension=dim)


def _embedding(dim: int = 4) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _record(**overrides) -> MemoryRecord:
    now = _utcnow()
    defaults = dict(
        id=uuid4(),
        content="test content",
        embedding=_embedding(),
        embedding_model=_model_ref(),
        timestamp=now,
        valid_from=now,
        valid_until=None,
        causal_parent_id=None,
        importance=0.5,
        surprise=0.5,
        source="agent_output",
        lane_id="general",
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


# ── EmbeddingModelRef ─────────────────────────────────────────────────────────


class TestEmbeddingModelRef:
    def test_construction(self):
        ref = EmbeddingModelRef(model_id="BAAI/bge-m3", version="1.5.0", dimension=1024)
        assert ref.model_id == "BAAI/bge-m3"
        assert ref.version == "1.5.0"
        assert ref.dimension == 1024

    def test_frozen(self):
        ref = _model_ref()
        with pytest.raises(AttributeError):
            ref.dimension = 99

    def test_dimension_zero_raises(self):
        with pytest.raises(ValueError, match="dimension must be > 0"):
            EmbeddingModelRef(model_id="m", version="1", dimension=0)

    def test_dimension_negative_raises(self):
        with pytest.raises(ValueError, match="dimension must be > 0"):
            EmbeddingModelRef(model_id="m", version="1", dimension=-1)

    def test_model_id_is_str(self):
        ref = _model_ref()
        assert isinstance(ref.model_id, str)

    def test_version_is_str(self):
        ref = _model_ref()
        assert isinstance(ref.version, str)


# ── MemoryRecord ──────────────────────────────────────────────────────────────


class TestMemoryRecord:
    def test_construction(self):
        record = _record()
        assert record.importance == 0.5
        assert record.surprise == 0.5
        assert record.valid_until is None

    def test_frozen(self):
        record = _record()
        with pytest.raises(AttributeError):
            record.importance = 0.9

    def test_importance_below_zero_raises(self):
        with pytest.raises(ValueError, match="importance"):
            _record(importance=-0.01)

    def test_importance_above_one_raises(self):
        with pytest.raises(ValueError, match="importance"):
            _record(importance=1.01)

    def test_importance_at_zero_valid(self):
        r = _record(importance=0.0)
        assert r.importance == 0.0

    def test_importance_at_one_valid(self):
        r = _record(importance=1.0)
        assert r.importance == 1.0

    def test_surprise_below_zero_raises(self):
        with pytest.raises(ValueError, match="surprise"):
            _record(surprise=-0.01)

    def test_surprise_above_one_raises(self):
        with pytest.raises(ValueError, match="surprise"):
            _record(surprise=1.01)

    def test_surprise_at_zero_valid(self):
        r = _record(surprise=0.0)
        assert r.surprise == 0.0

    def test_surprise_at_one_valid(self):
        r = _record(surprise=1.0)
        assert r.surprise == 1.0

    def test_valid_until_none(self):
        """valid_until=None represents a currently-valid record."""
        r = _record(valid_until=None)
        assert r.valid_until is None

    def test_valid_until_set(self):
        now = _utcnow()
        r = _record(valid_until=now)
        assert r.valid_until == now

    def test_causal_parent_id_none(self):
        r = _record(causal_parent_id=None)
        assert r.causal_parent_id is None

    def test_causal_parent_id_uuid(self):
        uid = uuid4()
        r = _record(causal_parent_id=uid)
        assert r.causal_parent_id == uid

    def test_embedding_is_ndarray(self):
        r = _record()
        assert isinstance(r.embedding, np.ndarray)

    def test_embedding_holds_dimension_length_vector(self):
        dim = 8
        r = _record(embedding=_embedding(dim), embedding_model=_model_ref(dim))
        assert r.embedding.shape == (dim,)

    def test_id_is_uuid(self):
        r = _record()
        assert isinstance(r.id, UUID)


# ── AssociativeState ──────────────────────────────────────────────────────────


class TestAssociativeState:
    def _state(self, r: int = 4) -> AssociativeState:
        return AssociativeState(
            lane_id="facts",
            S=np.zeros((r, r), dtype=np.float64),
            retention=np.ones(r, dtype=np.float64),
            write_strength=np.ones(r, dtype=np.float64),
            step=0,
            embedding_model=_model_ref(r),
        )

    def test_construction(self):
        s = self._state()
        assert s.lane_id == "facts"
        assert s.step == 0

    def test_mutable_field_reassignment(self):
        """AssociativeState is mutable — field reassignment must succeed."""
        s = self._state()
        s.step = 42
        assert s.step == 42

    def test_mutable_lane_id_reassignment(self):
        s = self._state()
        s.lane_id = "events"
        assert s.lane_id == "events"

    def test_inplace_ndarray_mutation(self):
        """In-place ndarray mutation must work (mutable dataclass)."""
        s = self._state(r=4)
        s.S[0, 0] = 99.0
        assert s.S[0, 0] == 99.0

    def test_S_shape(self):
        for r in (2, 4, 8):
            s = self._state(r)
            assert s.S.shape == (r, r)


# ── ExtractionResult ──────────────────────────────────────────────────────────


class TestExtractionResult:
    def test_construction(self):
        er = ExtractionResult(
            entities=[{"id": "e1", "label": "Alice", "type": "person", "properties": {}}],
            relations=[{"source_id": "e1", "target_id": "e2", "relation": "knows", "weight": 0.9}],
        )
        assert len(er.entities) == 1
        assert len(er.relations) == 1

    def test_frozen(self):
        er = ExtractionResult(entities=[], relations=[])
        with pytest.raises(AttributeError):
            er.entities = []

    def test_empty_construction(self):
        er = ExtractionResult(entities=[], relations=[])
        assert er.entities == []
        assert er.relations == []


# ── AdmissionConfig ───────────────────────────────────────────────────────────


class TestAdmissionConfig:
    def test_defaults(self):
        cfg = AdmissionConfig()
        assert cfg.surprise_threshold_percentile == 0.90
        assert cfg.importance_floor == 0.05
        assert cfg.rolling_window_size == 100
        assert cfg.gate_enabled is True

    def test_frozen(self):
        cfg = AdmissionConfig()
        with pytest.raises(AttributeError):
            cfg.gate_enabled = False

    def test_custom_values(self):
        cfg = AdmissionConfig(
            surprise_threshold_percentile=0.75,
            importance_floor=0.10,
            rolling_window_size=50,
            gate_enabled=False,
        )
        assert cfg.surprise_threshold_percentile == 0.75
        assert cfg.gate_enabled is False


# ── DeltaClampPolicy ──────────────────────────────────────────────────────────


class TestDeltaClampPolicy:
    def test_defaults(self):
        pol = DeltaClampPolicy()
        assert pol.max_abs_delta_per_node == 0.1
        assert pol.max_total_delta_per_pass == 0.5
        assert pol.min_weight == 0.0
        assert pol.max_weight == 1.0

    def test_frozen(self):
        pol = DeltaClampPolicy()
        with pytest.raises(AttributeError):
            pol.min_weight = 0.5

    def test_min_weight_greater_than_max_weight_raises(self):
        with pytest.raises(ValueError, match="min_weight"):
            DeltaClampPolicy(min_weight=0.8, max_weight=0.2)

    def test_max_abs_delta_zero_raises(self):
        with pytest.raises(ValueError, match="max_abs_delta_per_node"):
            DeltaClampPolicy(max_abs_delta_per_node=0.0)

    def test_max_abs_delta_negative_raises(self):
        with pytest.raises(ValueError, match="max_abs_delta_per_node"):
            DeltaClampPolicy(max_abs_delta_per_node=-0.1)

    def test_max_total_delta_zero_raises(self):
        with pytest.raises(ValueError, match="max_total_delta_per_pass"):
            DeltaClampPolicy(max_total_delta_per_pass=0.0)

    def test_equal_min_max_weight_valid(self):
        pol = DeltaClampPolicy(min_weight=0.5, max_weight=0.5)
        assert pol.min_weight == pol.max_weight


# ── MemoryContext ─────────────────────────────────────────────────────────────


class TestMemoryContext:
    def _context(self, token_count: int = 50, token_budget: int = 100) -> MemoryContext:
        return MemoryContext(
            context_id=uuid4(),
            recent_episodes=[],
            retrieved_episodes=[],
            semantic_facts=[],
            conflict_flags=[],
            token_count=token_count,
            token_budget=token_budget,
        )

    def test_construction(self):
        ctx = self._context()
        assert ctx.token_count == 50
        assert ctx.token_budget == 100

    def test_frozen(self):
        ctx = self._context()
        with pytest.raises(AttributeError):
            ctx.token_count = 10

    def test_token_count_exceeds_budget_raises(self):
        with pytest.raises(ValueError, match="token_count"):
            self._context(token_count=101, token_budget=100)

    def test_token_count_equals_budget_valid(self):
        ctx = self._context(token_count=100, token_budget=100)
        assert ctx.token_count == ctx.token_budget

    def test_context_id_is_uuid(self):
        ctx = self._context()
        assert isinstance(ctx.context_id, UUID)

    def test_episodes_are_lists(self):
        ctx = self._context()
        assert isinstance(ctx.recent_episodes, list)
        assert isinstance(ctx.retrieved_episodes, list)


# ── Report Types ──────────────────────────────────────────────────────────────


class TestConsolidationReport:
    def test_construction(self):
        r = ConsolidationReport(
            promoted=5, pruned=2, conflicts_flagged=1, conflicts_staled=0, elapsed_ms=12.5
        )
        assert r.promoted == 5
        assert r.elapsed_ms == 12.5

    def test_frozen(self):
        r = ConsolidationReport(
            promoted=1, pruned=0, conflicts_flagged=0, conflicts_staled=0, elapsed_ms=1.0
        )
        with pytest.raises(AttributeError):
            r.promoted = 99


class TestLearningReport:
    def test_construction(self):
        r = LearningReport(
            nodes_updated=10,
            nodes_rejected=2,
            nodes_clamped=1,
            interference_score=0.05,
            total_delta_mass=0.3,
        )
        assert r.nodes_updated == 10
        assert r.total_delta_mass == 0.3

    def test_frozen(self):
        r = LearningReport(
            nodes_updated=0, nodes_rejected=0, nodes_clamped=0,
            interference_score=0.0, total_delta_mass=0.0
        )
        with pytest.raises(AttributeError):
            r.nodes_updated = 5


class TestRetrievalEffectivenessReport:
    def test_construction(self):
        r = RetrievalEffectivenessReport(
            total_contexts=100,
            total_records_served=500,
            total_hits=80,
            hit_rate=0.8,
            per_lane_hit_rates={"facts": 0.9, "general": 0.7},
            conflict_accumulation=3,
            stale_conflict_count=1,
        )
        assert r.hit_rate == 0.8
        assert r.per_lane_hit_rates["facts"] == 0.9

    def test_frozen(self):
        r = RetrievalEffectivenessReport(
            total_contexts=0, total_records_served=0, total_hits=0,
            hit_rate=0.0, per_lane_hit_rates={}, conflict_accumulation=0, stale_conflict_count=0
        )
        with pytest.raises(AttributeError):
            r.hit_rate = 1.0


class TestForgetReport:
    def test_construction(self):
        r = ForgetReport(
            policy_kind=ForgetPolicyKind.AGE_DRIFT,
            records_removed=7,
            lanes_reset=["facts"],
            elapsed_ms=3.2,
        )
        assert r.policy_kind == ForgetPolicyKind.AGE_DRIFT
        assert r.records_removed == 7

    def test_frozen(self):
        r = ForgetReport(
            policy_kind=ForgetPolicyKind.EXPLICIT_KEY,
            records_removed=0,
            lanes_reset=[],
            elapsed_ms=0.0,
        )
        with pytest.raises(AttributeError):
            r.records_removed = 5


class TestSnapshotHandle:
    def test_construction(self):
        now = _utcnow()
        s = SnapshotHandle(tag="v1", timestamp=now, node_count=42, edge_count=100)
        assert s.tag == "v1"
        assert s.node_count == 42

    def test_frozen(self):
        s = SnapshotHandle(tag="v1", timestamp=_utcnow(), node_count=0, edge_count=0)
        with pytest.raises(AttributeError):
            s.tag = "v2"


class TestForgettingAuditReport:
    def test_construction(self):
        r = ForgettingAuditReport(
            per_task_modification_counts={"task_a": 5},
            interference_budget_usage={"task_a": 0.3},
            retention_ratio=0.85,
            acquisition_count=20,
            rollback_count=1,
        )
        assert r.retention_ratio == 0.85
        assert r.rollback_count == 1

    def test_frozen(self):
        r = ForgettingAuditReport(
            per_task_modification_counts={},
            interference_budget_usage={},
            retention_ratio=0.0,
            acquisition_count=0,
            rollback_count=0,
        )
        with pytest.raises(AttributeError):
            r.retention_ratio = 1.0


# ── Enums ─────────────────────────────────────────────────────────────────────


class TestLaneId:
    def test_member_count(self):
        assert len(LaneId) == 5

    def test_values(self):
        assert LaneId.FACTS == "facts"
        assert LaneId.PREFERENCES == "preferences"
        assert LaneId.TASK_PROGRESS == "task_progress"
        assert LaneId.EVENTS == "events"
        assert LaneId.GENERAL == "general"

    def test_str_subclass(self):
        assert isinstance(LaneId.FACTS, str)


class TestMemorySource:
    def test_member_count(self):
        assert len(MemorySource) == 3

    def test_values(self):
        assert MemorySource.AGENT_OUTPUT == "agent_output"
        assert MemorySource.USER_INPUT == "user_input"
        assert MemorySource.TOOL_RESULT == "tool_result"

    def test_str_subclass(self):
        assert isinstance(MemorySource.AGENT_OUTPUT, str)


class TestForgetPolicyKind:
    def test_member_count(self):
        assert len(ForgetPolicyKind) == 5

    def test_values(self):
        assert ForgetPolicyKind.AGE_DRIFT == "age_drift"
        assert ForgetPolicyKind.IMPORTANCE_DECAY == "importance_decay"
        assert ForgetPolicyKind.TASK_SCOPE_EXPIRY == "task_scope_expiry"
        assert ForgetPolicyKind.EXPLICIT_KEY == "explicit_key"
        assert ForgetPolicyKind.INTERFERENCE_PRUNE == "interference_prune"


# ── ForgetPolicy Types ────────────────────────────────────────────────────────


class TestAgeDriftPolicy:
    def test_construction(self):
        p = AgeDriftPolicy(decay_lambda=0.05, importance_floor=0.1)
        assert p.decay_lambda == 0.05
        assert p.importance_floor == 0.1

    def test_kind_auto_set(self):
        p = AgeDriftPolicy(decay_lambda=0.1, importance_floor=0.05)
        assert p.kind == ForgetPolicyKind.AGE_DRIFT

    def test_frozen(self):
        p = AgeDriftPolicy(decay_lambda=0.1, importance_floor=0.05)
        with pytest.raises(AttributeError):
            p.decay_lambda = 0.9


class TestImportanceDecayPolicy:
    def test_construction(self):
        p = ImportanceDecayPolicy(unretrieved_cycles_threshold=10)
        assert p.unretrieved_cycles_threshold == 10

    def test_kind_auto_set(self):
        p = ImportanceDecayPolicy(unretrieved_cycles_threshold=5)
        assert p.kind == ForgetPolicyKind.IMPORTANCE_DECAY

    def test_frozen(self):
        p = ImportanceDecayPolicy(unretrieved_cycles_threshold=10)
        with pytest.raises(AttributeError):
            p.unretrieved_cycles_threshold = 99


class TestTaskScopeExpiryPolicy:
    def test_construction(self):
        p = TaskScopeExpiryPolicy(task_tag="task_123")
        assert p.task_tag == "task_123"

    def test_kind_auto_set(self):
        p = TaskScopeExpiryPolicy(task_tag="t")
        assert p.kind == ForgetPolicyKind.TASK_SCOPE_EXPIRY

    def test_frozen(self):
        p = TaskScopeExpiryPolicy(task_tag="t")
        with pytest.raises(AttributeError):
            p.task_tag = "other"


class TestExplicitKeyPolicy:
    def test_construction(self):
        ids = [uuid4(), uuid4()]
        p = ExplicitKeyPolicy(target_ids=ids)
        assert p.target_ids == ids
        assert len(p.target_ids) == 2

    def test_kind_auto_set(self):
        p = ExplicitKeyPolicy(target_ids=[])
        assert p.kind == ForgetPolicyKind.EXPLICIT_KEY

    def test_frozen(self):
        p = ExplicitKeyPolicy(target_ids=[])
        with pytest.raises(AttributeError):
            p.target_ids = [uuid4()]


class TestInterferencePrunePolicy:
    def test_construction(self):
        p = InterferencePrunePolicy(task_tag="task_abc", confidence_floor=0.6)
        assert p.task_tag == "task_abc"
        assert p.confidence_floor == 0.6

    def test_kind_auto_set(self):
        p = InterferencePrunePolicy(task_tag="t", confidence_floor=0.5)
        assert p.kind == ForgetPolicyKind.INTERFERENCE_PRUNE

    def test_frozen(self):
        p = InterferencePrunePolicy(task_tag="t", confidence_floor=0.5)
        with pytest.raises(AttributeError):
            p.confidence_floor = 0.9


class TestForgetPolicyUnion:
    def test_union_covers_all_five_policy_types(self):
        """ForgetPolicy union includes all five concrete policy types."""
        policies = [
            AgeDriftPolicy(decay_lambda=0.1, importance_floor=0.05),
            ImportanceDecayPolicy(unretrieved_cycles_threshold=5),
            TaskScopeExpiryPolicy(task_tag="t"),
            ExplicitKeyPolicy(target_ids=[]),
            InterferencePrunePolicy(task_tag="t", confidence_floor=0.5),
        ]
        # Each policy is an instance of its concrete type.
        # ForgetPolicy is a type alias, so we verify via isinstance of each concrete class.
        assert isinstance(policies[0], AgeDriftPolicy)
        assert isinstance(policies[1], ImportanceDecayPolicy)
        assert isinstance(policies[2], TaskScopeExpiryPolicy)
        assert isinstance(policies[3], ExplicitKeyPolicy)
        assert isinstance(policies[4], InterferencePrunePolicy)


# ── Error Types ───────────────────────────────────────────────────────────────


class TestMemLearnError:
    def test_is_exception(self):
        err = MemLearnError("base error")
        assert isinstance(err, Exception)

    def test_subclasses_inherit(self):
        assert issubclass(EmbeddingDimensionMismatch, MemLearnError)
        assert issubclass(BudgetExceeded, MemLearnError)
        assert issubclass(StaleEmbeddingModel, MemLearnError)
        assert issubclass(InvalidLaneId, MemLearnError)


class TestEmbeddingDimensionMismatch:
    def test_construction(self):
        err = EmbeddingDimensionMismatch(expected=1024, got=768)
        assert err.expected == 1024
        assert err.got == 768

    def test_message(self):
        err = EmbeddingDimensionMismatch(expected=512, got=256)
        assert "512" in str(err)
        assert "256" in str(err)

    def test_is_mem_learn_error(self):
        err = EmbeddingDimensionMismatch(expected=1, got=2)
        assert isinstance(err, MemLearnError)
        assert isinstance(err, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(MemLearnError):
            raise EmbeddingDimensionMismatch(expected=4, got=8)


class TestBudgetExceeded:
    def test_construction(self):
        err = BudgetExceeded(token_count=200, token_budget=100)
        assert err.token_count == 200
        assert err.token_budget == 100

    def test_message(self):
        err = BudgetExceeded(token_count=150, token_budget=100)
        assert "150" in str(err)
        assert "100" in str(err)

    def test_is_mem_learn_error(self):
        err = BudgetExceeded(token_count=1, token_budget=0)
        assert isinstance(err, MemLearnError)
        assert isinstance(err, Exception)


class TestStaleEmbeddingModel:
    def test_construction(self):
        err = StaleEmbeddingModel(expected="bge-m3", got="ada-002")
        assert err.expected == "bge-m3"
        assert err.got == "ada-002"

    def test_message(self):
        err = StaleEmbeddingModel(expected="bge-m3", got="ada-002")
        assert "bge-m3" in str(err)
        assert "ada-002" in str(err)

    def test_is_mem_learn_error(self):
        err = StaleEmbeddingModel(expected="a", got="b")
        assert isinstance(err, MemLearnError)
        assert isinstance(err, Exception)


class TestInvalidLaneId:
    def test_construction(self):
        err = InvalidLaneId(lane_id="bogus_lane")
        assert err.lane_id == "bogus_lane"

    def test_message(self):
        err = InvalidLaneId(lane_id="bogus_lane")
        assert "bogus_lane" in str(err)

    def test_is_mem_learn_error(self):
        err = InvalidLaneId(lane_id="x")
        assert isinstance(err, MemLearnError)
        assert isinstance(err, Exception)
