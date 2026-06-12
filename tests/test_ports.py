"""Unit tests for memlearn.ports — no Postgres connection required.

Tests are structural: frozen dataclass behavior, enum values, runtime_checkable
isinstance semantics, and Protocol method arities. No adapter implementations
are constructed or connected to a database.
"""
import inspect
import pytest
from uuid import UUID, uuid4

from memlearn.ports import (
    GraphStoreAdapter,
    KVStoreAdapter,
    MergeStrategy,
    SearchResult,
    VectorStoreAdapter,
)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_fields_accessible(self):
        sr = SearchResult(id=uuid4(), score=0.42, payload={"k": "v"})
        assert isinstance(sr.id, UUID)
        assert sr.score == pytest.approx(0.42)
        assert sr.payload == {"k": "v"}

    def test_frozen_raises_attribute_error_on_mutation(self):
        """Python 3.10 frozen dataclass raises AttributeError, not FrozenInstanceError."""
        sr = SearchResult(id=uuid4(), score=0.1, payload={})
        with pytest.raises(AttributeError):
            sr.score = 0.9  # type: ignore[misc]

    def test_frozen_raises_on_id_mutation(self):
        sr = SearchResult(id=uuid4(), score=0.0, payload={})
        with pytest.raises(AttributeError):
            sr.id = uuid4()  # type: ignore[misc]

    def test_frozen_raises_on_payload_mutation(self):
        sr = SearchResult(id=uuid4(), score=0.0, payload={})
        with pytest.raises(AttributeError):
            sr.payload = {"other": True}  # type: ignore[misc]

    def test_equality_same_fields(self):
        uid = uuid4()
        a = SearchResult(id=uid, score=0.5, payload={"x": 1})
        b = SearchResult(id=uid, score=0.5, payload={"x": 1})
        assert a == b

    def test_inequality_different_score(self):
        uid = uuid4()
        a = SearchResult(id=uid, score=0.5, payload={})
        b = SearchResult(id=uid, score=0.9, payload={})
        assert a != b

    def test_inequality_different_id(self):
        a = SearchResult(id=uuid4(), score=0.5, payload={})
        b = SearchResult(id=uuid4(), score=0.5, payload={})
        assert a != b


# ---------------------------------------------------------------------------
# MergeStrategy
# ---------------------------------------------------------------------------

class TestMergeStrategy:
    def test_keep_a_value(self):
        assert MergeStrategy.KEEP_A == "keep_a"

    def test_keep_b_value(self):
        assert MergeStrategy.KEEP_B == "keep_b"

    def test_merge_value(self):
        assert MergeStrategy.MERGE == "merge"

    def test_no_supersede(self):
        values = {m.value for m in MergeStrategy}
        assert "supersede" not in values

    def test_str_enum_round_trip_keep_a(self):
        assert MergeStrategy("keep_a") == MergeStrategy.KEEP_A

    def test_str_enum_round_trip_keep_b(self):
        assert MergeStrategy("keep_b") == MergeStrategy.KEEP_B

    def test_str_enum_round_trip_merge(self):
        assert MergeStrategy("merge") == MergeStrategy.MERGE

    def test_exactly_three_members(self):
        assert len(list(MergeStrategy)) == 3

    def test_is_str_subclass(self):
        assert isinstance(MergeStrategy.KEEP_A, str)


# ---------------------------------------------------------------------------
# VectorStoreAdapter — runtime_checkable isinstance
# ---------------------------------------------------------------------------

class TestVectorStoreAdapterProtocol:
    def test_isinstance_true_when_all_methods_present(self):
        import numpy as np
        from typing import List, Optional

        class MinimalVector:
            def upsert(self, id: UUID, embedding: np.ndarray, payload: dict) -> None: ...
            def search(
                self,
                embedding: np.ndarray,
                k: int,
                lane_id: Optional[str] = None,
            ) -> List[SearchResult]: ...
            def delete(self, id: UUID) -> None: ...
            def update_valid_until(self, id: UUID, valid_until) -> None: ...
            def recent_by_timestamp(self, n: int, lane_id=None): ...

        assert isinstance(MinimalVector(), VectorStoreAdapter)

    def test_isinstance_false_missing_upsert(self):
        class NoUpsert:
            def search(self, embedding, k, lane_id=None): ...
            def delete(self, id): ...

        assert not isinstance(NoUpsert(), VectorStoreAdapter)

    def test_isinstance_false_missing_search(self):
        class NoSearch:
            def upsert(self, id, embedding, payload): ...
            def delete(self, id): ...

        assert not isinstance(NoSearch(), VectorStoreAdapter)

    def test_isinstance_false_missing_delete(self):
        class NoDelete:
            def upsert(self, id, embedding, payload): ...
            def search(self, embedding, k, lane_id=None): ...

        assert not isinstance(NoDelete(), VectorStoreAdapter)

    def test_upsert_arity(self):
        """upsert must accept (self, id, embedding, payload, lane_id) — 4 params beyond self (Decision D3)."""
        sig = inspect.signature(VectorStoreAdapter.upsert)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["id", "embedding", "payload", "lane_id"]

    def test_search_arity(self):
        """search must accept (self, embedding, k) at minimum — 2 required params."""
        sig = inspect.signature(VectorStoreAdapter.search)
        params = [p for p in sig.parameters if p != "self"]
        assert params[0] == "embedding"
        assert params[1] == "k"

    def test_search_has_lane_id_optional(self):
        sig = inspect.signature(VectorStoreAdapter.search)
        assert "lane_id" in sig.parameters
        assert sig.parameters["lane_id"].default is None


# ---------------------------------------------------------------------------
# GraphStoreAdapter — runtime_checkable isinstance
# ---------------------------------------------------------------------------

class TestGraphStoreAdapterProtocol:
    def _make_full(self):
        class FullGraph:
            def add_node(self, id, label, properties): ...
            def add_edge(self, src, dst, relation, weight): ...
            def ppr_search(self, seed_nodes, alpha, k): ...
            def merge_nodes(self, a, b, strategy): ...
            def delete_node(self, id): ...
            def apply_weight_deltas(self, node_deltas, edge_deltas, policy): ...
            def update_node_properties(self, id, properties): ...
            def get_all_node_weights(self): ...

        return FullGraph()

    def test_isinstance_true_all_methods(self):
        assert isinstance(self._make_full(), GraphStoreAdapter)

    def test_isinstance_false_missing_ppr_search(self):
        class NoPPR:
            def add_node(self, id, label, properties): ...
            def add_edge(self, src, dst, relation, weight): ...
            def merge_nodes(self, a, b, strategy): ...
            def delete_node(self, id): ...
            def apply_weight_deltas(self, node_deltas, edge_deltas, policy): ...

        assert not isinstance(NoPPR(), GraphStoreAdapter)

    def test_isinstance_false_missing_delete_node(self):
        class NoDelete:
            def add_node(self, id, label, properties): ...
            def add_edge(self, src, dst, relation, weight): ...
            def ppr_search(self, seed_nodes, alpha, k): ...
            def merge_nodes(self, a, b, strategy): ...
            def apply_weight_deltas(self, node_deltas, edge_deltas, policy): ...

        assert not isinstance(NoDelete(), GraphStoreAdapter)

    def test_isinstance_false_missing_apply_weight_deltas(self):
        class NoDeltas:
            def add_node(self, id, label, properties): ...
            def add_edge(self, src, dst, relation, weight): ...
            def ppr_search(self, seed_nodes, alpha, k): ...
            def merge_nodes(self, a, b, strategy): ...
            def delete_node(self, id): ...

        assert not isinstance(NoDeltas(), GraphStoreAdapter)

    def test_merge_nodes_arity(self):
        sig = inspect.signature(GraphStoreAdapter.merge_nodes)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["a", "b", "strategy"]

    def test_ppr_search_arity(self):
        sig = inspect.signature(GraphStoreAdapter.ppr_search)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["seed_nodes", "alpha", "k"]

    def test_apply_weight_deltas_arity(self):
        sig = inspect.signature(GraphStoreAdapter.apply_weight_deltas)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["node_deltas", "edge_deltas", "policy"]


# ---------------------------------------------------------------------------
# KVStoreAdapter — runtime_checkable isinstance
# ---------------------------------------------------------------------------

class TestKVStoreAdapterProtocol:
    def _make_full(self):
        class FullKV:
            def set(self, key, value): ...
            def get(self, key): ...
            def delete(self, key): ...
            def scan_prefix(self, prefix): ...

        return FullKV()

    def test_isinstance_true_all_methods(self):
        assert isinstance(self._make_full(), KVStoreAdapter)

    def test_isinstance_false_missing_set(self):
        class NoSet:
            def get(self, key): ...
            def delete(self, key): ...
            def scan_prefix(self, prefix): ...

        assert not isinstance(NoSet(), KVStoreAdapter)

    def test_isinstance_false_missing_delete(self):
        class NoDelete:
            def set(self, key, value): ...
            def get(self, key): ...
            def scan_prefix(self, prefix): ...

        assert not isinstance(NoDelete(), KVStoreAdapter)

    def test_isinstance_false_missing_scan_prefix(self):
        class NoScan:
            def set(self, key, value): ...
            def get(self, key): ...
            def delete(self, key): ...

        assert not isinstance(NoScan(), KVStoreAdapter)

    def test_scan_prefix_arity(self):
        sig = inspect.signature(KVStoreAdapter.scan_prefix)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["prefix"]

    def test_set_arity(self):
        sig = inspect.signature(KVStoreAdapter.set)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["key", "value"]

    def test_get_arity(self):
        sig = inspect.signature(KVStoreAdapter.get)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["key"]

    def test_delete_arity(self):
        sig = inspect.signature(KVStoreAdapter.delete)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["key"]
