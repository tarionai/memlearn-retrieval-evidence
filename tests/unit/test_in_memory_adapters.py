"""Unit tests for in-memory adapter implementations.

Each test class corresponds to one adapter. Tests are pure Python — no network,
no database, no external services.
"""
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest
import socket

from memlearn.adapters.in_memory.fake_llm_port import FakeLLMPort
from memlearn.adapters.in_memory.fake_tokenizer import FakeTokenizer
from memlearn.adapters.in_memory.kv_store import InMemoryKVStore


class TestInMemoryKVStore:
    def test_kv_set_get_roundtrip(self) -> None:
        store = InMemoryKVStore()
        store.set("alpha", b"hello")
        assert store.get("alpha") == b"hello"

    def test_kv_get_missing_returns_none(self) -> None:
        store = InMemoryKVStore()
        assert store.get("does_not_exist") is None

    def test_kv_delete_removes_entry(self) -> None:
        store = InMemoryKVStore()
        store.set("beta", b"world")
        store.delete("beta")
        assert store.get("beta") is None

    def test_kv_delete_idempotent(self) -> None:
        store = InMemoryKVStore()
        # Deleting a key that was never set must not raise.
        store.delete("ghost")
        store.delete("ghost")

    def test_kv_scan_prefix_filters_correctly(self) -> None:
        store = InMemoryKVStore()
        store.set("user:1", b"alice")
        store.set("user:2", b"bob")
        store.set("task:1", b"write")
        result = dict(store.scan_prefix("user:"))
        assert result == {"user:1": b"alice", "user:2": b"bob"}

    def test_kv_scan_prefix_empty_string_returns_all(self) -> None:
        store = InMemoryKVStore()
        store.set("a", b"1")
        store.set("b", b"2")
        store.set("c", b"3")
        result = dict(store.scan_prefix(""))
        assert result == {"a": b"1", "b": b"2", "c": b"3"}


class TestFakeTokenizer:
    def test_tokenizer_empty_string(self) -> None:
        tok = FakeTokenizer()
        assert tok.count_tokens("") == 0

    def test_tokenizer_single_word(self) -> None:
        tok = FakeTokenizer()
        assert tok.count_tokens("hello") == 1

    def test_tokenizer_multi_word(self) -> None:
        tok = FakeTokenizer()
        assert tok.count_tokens("a b c") == 3

    def test_tokenizer_is_deterministic(self) -> None:
        tok = FakeTokenizer()
        text = "the quick brown fox"
        assert tok.count_tokens(text) == tok.count_tokens(text)


class TestFakeLLMPort:
    def test_llm_returns_configured_response(self) -> None:
        llm = FakeLLMPort(response="hello world")
        assert llm.complete("any prompt", max_tokens=50) == "hello world"

    def test_llm_call_count_increments(self) -> None:
        llm = FakeLLMPort()
        llm.complete("first", max_tokens=10)
        llm.complete("second", max_tokens=10)
        assert llm.call_count == 2

    def test_llm_default_response(self) -> None:
        llm = FakeLLMPort()
        assert llm.complete("prompt", max_tokens=10) == "stub_response"

    def test_llm_no_network(self) -> None:
        # Structural test: constructing and calling FakeLLMPort must not
        # require socket access. No assertion beyond no exception raised.
        _ = socket  # imported but not used to block access — presence only
        llm = FakeLLMPort()
        llm.complete("test", max_tokens=10)


class TestFakeEmbedding:
    _MODEL = None  # initialised in helpers below

    @staticmethod
    def _model(dim: int = 16):
        from memlearn.primitives import EmbeddingModelRef
        return EmbeddingModelRef(model_id="fake/model", version="0.0.1", dimension=dim)

    @staticmethod
    def _embedder(dim: int = 16):
        from memlearn.adapters.in_memory.fake_embedding import FakeEmbedding
        return FakeEmbedding(TestFakeEmbedding._model(dim))

    def test_embedding_deterministic(self) -> None:
        emb = self._embedder()
        text = "the quick brown fox"
        np.testing.assert_array_equal(emb.embed(text), emb.embed(text))

    def test_embedding_different_texts_differ(self) -> None:
        emb = self._embedder()
        v1 = emb.embed("hello")
        v2 = emb.embed("world")
        assert not np.array_equal(v1, v2)

    def test_embedding_shape_and_dtype(self) -> None:
        dim = 32
        emb = self._embedder(dim)
        result = emb.embed("shape test")
        assert result.shape == (dim,)
        assert result.dtype == np.float32

    def test_embedding_normalised(self) -> None:
        emb = self._embedder()
        result = emb.embed("normalisation check")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_embedding_empty_string(self) -> None:
        dim = 16
        emb = self._embedder(dim)
        result = emb.embed("")
        assert result.shape == (dim,)
        assert np.all(result == 0.0)


class TestInMemoryVectorStore:
    _VALID_FROM = "2024-01-01T00:00:00+00:00"

    @staticmethod
    def _store():
        from memlearn.adapters.in_memory.vector_store import InMemoryVectorStore
        return InMemoryVectorStore()

    @staticmethod
    def _emb():
        from memlearn.adapters.in_memory.fake_embedding import FakeEmbedding
        from memlearn.primitives import EmbeddingModelRef
        return FakeEmbedding(EmbeddingModelRef("fake/model", "0.0.1", dimension=16))

    def test_vector_upsert_search_roundtrip(self) -> None:
        store = self._store()
        emb = self._emb()
        uid = uuid4()
        vec = emb.embed("roundtrip text")
        store.upsert(uid, vec, {"valid_from": self._VALID_FROM})
        results = store.search(vec, k=1)
        assert len(results) == 1
        assert results[0].id == uid
        assert results[0].payload["valid_from"] == self._VALID_FROM
        assert results[0].payload["lane_id"] == "default"
        assert results[0].payload["valid_until"] is None

    def test_vector_search_returns_k_results(self) -> None:
        store = self._store()
        emb = self._emb()
        uids = [uuid4() for _ in range(5)]
        vecs = [emb.embed(f"text {i}") for i in range(5)]
        for uid, vec in zip(uids, vecs):
            store.upsert(uid, vec, {"valid_from": self._VALID_FROM})
        results = store.search(vecs[0], k=3)
        assert len(results) == 3

    def test_vector_search_lane_filter(self) -> None:
        store = self._store()
        emb = self._emb()
        uid_a = uuid4()
        uid_b = uuid4()
        vec_a = emb.embed("lane A text")
        vec_b = emb.embed("lane B text")
        store.upsert(uid_a, vec_a, {"valid_from": self._VALID_FROM}, lane_id="A")
        store.upsert(uid_b, vec_b, {"valid_from": self._VALID_FROM}, lane_id="B")
        results = store.search(vec_a, k=10, lane_id="A")
        returned_ids = [r.id for r in results]
        assert uid_a in returned_ids
        assert uid_b not in returned_ids

    def test_vector_delete_removes_from_search(self) -> None:
        store = self._store()
        emb = self._emb()
        uid = uuid4()
        vec = emb.embed("to be deleted")
        store.upsert(uid, vec, {"valid_from": self._VALID_FROM})
        store.delete(uid)
        results = store.search(vec, k=10)
        assert uid not in [r.id for r in results]

    def test_vector_upsert_overwrites(self) -> None:
        store = self._store()
        emb = self._emb()
        uid = uuid4()
        vec = emb.embed("overwrite test")
        store.upsert(uid, vec, {"valid_from": self._VALID_FROM, "text": "old"})
        store.upsert(uid, vec, {"valid_from": self._VALID_FROM, "text": "new"})
        results = store.search(vec, k=1)
        assert len(results) == 1
        assert results[0].payload["text"] == "new"

    def test_vector_update_valid_until(self) -> None:
        store = self._store()
        emb = self._emb()
        uid = uuid4()
        vec = emb.embed("valid_until test")
        store.upsert(uid, vec, {"valid_from": self._VALID_FROM})
        dt = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        store.update_valid_until(uid, dt)
        results = store.search(vec, k=1)
        assert results[0].payload["valid_until"] == dt.isoformat()

    def test_vector_recent_by_timestamp_order(self) -> None:
        store = self._store()
        emb = self._emb()
        uids = [uuid4() for _ in range(3)]
        for i, uid in enumerate(uids):
            vec = emb.embed(f"recent {i}")
            store.upsert(uid, vec, {"valid_from": self._VALID_FROM})
            time.sleep(0.001)
        results = store.recent_by_timestamp(3)
        assert [r.id for r in results] == list(reversed(uids))

    def test_vector_recent_by_timestamp_lane_filter(self) -> None:
        store = self._store()
        emb = self._emb()
        uid_x1 = uuid4()
        uid_x2 = uuid4()
        uid_y = uuid4()
        store.upsert(uid_x1, emb.embed("x1"), {"valid_from": self._VALID_FROM}, lane_id="X")
        store.upsert(uid_x2, emb.embed("x2"), {"valid_from": self._VALID_FROM}, lane_id="X")
        store.upsert(uid_y, emb.embed("y1"), {"valid_from": self._VALID_FROM}, lane_id="Y")
        results = store.recent_by_timestamp(10, lane_id="X")
        returned_ids = [r.id for r in results]
        assert uid_x1 in returned_ids
        assert uid_x2 in returned_ids
        assert uid_y not in returned_ids


class TestInMemoryGraphStore:
    @staticmethod
    def _store():
        from memlearn.adapters.in_memory.graph_store import InMemoryGraphStore
        return InMemoryGraphStore()

    @staticmethod
    def _policy():
        from memlearn.primitives import DeltaClampPolicy
        return DeltaClampPolicy()

    def test_graph_add_node_initial_weight_zero(self) -> None:
        store = self._store()
        uid = uuid4()
        store.add_node(uid, "concept", {})
        assert store.get_all_node_weights()[uid] == 0.0

    def test_graph_add_edge_appears_in_ppr(self) -> None:
        store = self._store()
        a, b = uuid4(), uuid4()
        store.add_node(a, "A", {})
        store.add_node(b, "B", {})
        store.add_edge(a, b, "related", 1.0)
        results = store.ppr_search([a], alpha=0.85, k=5)
        assert b in results

    def test_graph_apply_weight_deltas_increases_weight(self) -> None:
        store = self._store()
        uid = uuid4()
        store.add_node(uid, "node", {})
        store.apply_weight_deltas({uid: 0.3}, {}, self._policy())
        assert store.get_all_node_weights()[uid] == pytest.approx(0.3)

    def test_graph_apply_weight_deltas_clamps_max(self) -> None:
        store = self._store()
        uid = uuid4()
        policy = self._policy()
        store.add_node(uid, "node", {})
        store.apply_weight_deltas({uid: 0.8}, {}, policy)
        store.apply_weight_deltas({uid: 0.8}, {}, policy)
        assert store.get_all_node_weights()[uid] == pytest.approx(policy.max_weight)

    def test_graph_apply_weight_deltas_clamps_min(self) -> None:
        store = self._store()
        uid = uuid4()
        policy = self._policy()
        store.add_node(uid, "node", {})
        store.apply_weight_deltas({uid: -0.5}, {}, policy)
        assert store.get_all_node_weights()[uid] == pytest.approx(policy.min_weight)

    def test_graph_delete_node_removes_from_weights(self) -> None:
        store = self._store()
        uid = uuid4()
        store.add_node(uid, "ephemeral", {})
        store.delete_node(uid)
        assert uid not in store.get_all_node_weights()

    def test_graph_ppr_empty_returns_empty(self) -> None:
        store = self._store()
        assert store.ppr_search([uuid4()], alpha=0.85, k=5) == []

    def test_graph_ppr_connected_returns_results(self) -> None:
        store = self._store()
        a, b, c = uuid4(), uuid4(), uuid4()
        store.add_node(a, "A", {})
        store.add_node(b, "B", {})
        store.add_node(c, "C", {})
        store.add_edge(a, b, "r", 1.0)
        store.add_edge(b, c, "r", 1.0)
        results = store.ppr_search([a], alpha=0.85, k=5)
        assert len(results) > 0
        assert a not in results

    def test_graph_merge_keep_a_removes_b(self) -> None:
        from memlearn.ports import MergeStrategy
        store = self._store()
        a, b = uuid4(), uuid4()
        store.add_node(a, "A", {})
        store.add_node(b, "B", {})
        store.merge_nodes(a, b, MergeStrategy.KEEP_A)
        assert b not in store.get_all_node_weights()

    def test_graph_update_node_properties(self) -> None:
        store = self._store()
        uid = uuid4()
        store.add_node(uid, "L", {"x": 1})
        store.update_node_properties(uid, {"y": 2})
        _label, props = store._nodes[uid]
        assert props["x"] == 1
        assert props["y"] == 2


class TestFakeEntityExtractor:
    @staticmethod
    def _result() -> "ExtractionResult":
        from memlearn.primitives import ExtractionResult
        return ExtractionResult(
            entities=[{"id": "e1", "label": "Cat", "type": "animal", "properties": {}}],
            relations=[{"source_id": "e1", "target_id": "e1", "relation": "self", "weight": 1.0}],
        )

    @staticmethod
    def _extractor(fixtures=None):
        from memlearn.adapters.in_memory.fake_entity_extractor import FakeEntityExtractor
        return FakeEntityExtractor(fixtures=fixtures)

    def test_extractor_no_fixtures_returns_empty(self) -> None:
        from memlearn.primitives import ExtractionResult
        ext = self._extractor()
        result = ext.extract("any text")
        assert result == ExtractionResult(entities=[], relations=[])

    def test_extractor_matching_fixture_returns_value(self) -> None:
        fixture = self._result()
        ext = self._extractor(fixtures={"hello": fixture})
        assert ext.extract("hello") is fixture

    def test_extractor_non_matching_key_returns_empty(self) -> None:
        from memlearn.primitives import ExtractionResult
        ext = self._extractor(fixtures={"hello": self._result()})
        assert ext.extract("world") == ExtractionResult(entities=[], relations=[])

    def test_extractor_call_count_increments(self) -> None:
        ext = self._extractor()
        ext.extract("first")
        ext.extract("second")
        assert ext.call_count == 2


class TestSentenceTransformerEmbedder:
    @staticmethod
    def _embedder():
        from memlearn.adapters.in_memory.sentence_transformer_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()

    def test_embedding_shape_and_dtype(self) -> None:
        emb = self._embedder()
        result = emb.embed("account dispute letter received")
        assert result.shape == (384,)
        assert result.dtype == np.float32

    def test_embedding_normalised(self) -> None:
        emb = self._embedder()
        result = emb.embed("payment processing failed")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_embedding_deterministic(self) -> None:
        emb = self._embedder()
        text = "fraud investigation initiated"
        np.testing.assert_array_equal(emb.embed(text), emb.embed(text))

    def test_embedding_different_texts_differ(self) -> None:
        emb = self._embedder()
        v1 = emb.embed("account dispute")
        v2 = emb.embed("payment processing")
        assert not np.array_equal(v1, v2)

    def test_embedding_empty_string(self) -> None:
        emb = self._embedder()
        result = emb.embed("")
        assert result.shape == (384,)
        assert np.all(result == 0.0)

    def test_embedding_model_ref(self) -> None:
        emb = self._embedder()
        assert emb.embedding_model.dimension == 384
        assert emb.embedding_model.model_id == "all-MiniLM-L6-v2"

    def test_isinstance_protocol(self) -> None:
        from memlearn.ports import EmbedderPort
        from memlearn.adapters.in_memory.sentence_transformer_embedder import SentenceTransformerEmbedder
        assert isinstance(SentenceTransformerEmbedder(), EmbedderPort)


class TestProtocolCompliance:
    """isinstance checks confirming every in-memory adapter satisfies its Protocol."""

    @staticmethod
    def _model_ref():
        from memlearn.primitives import EmbeddingModelRef
        return EmbeddingModelRef(model_id="fake", version="0.0.1", dimension=4)

    def test_kv_isinstance_protocol(self) -> None:
        from memlearn.ports import KVStoreAdapter
        from memlearn.adapters.in_memory.kv_store import InMemoryKVStore
        assert isinstance(InMemoryKVStore(), KVStoreAdapter)

    def test_vector_store_isinstance_protocol(self) -> None:
        from memlearn.ports import VectorStoreAdapter
        from memlearn.adapters.in_memory.vector_store import InMemoryVectorStore
        assert isinstance(InMemoryVectorStore(), VectorStoreAdapter)

    def test_graph_store_isinstance_protocol(self) -> None:
        from memlearn.ports import GraphStoreAdapter
        from memlearn.adapters.in_memory.graph_store import InMemoryGraphStore
        assert isinstance(InMemoryGraphStore(), GraphStoreAdapter)

    def test_embedding_isinstance_protocol(self) -> None:
        from memlearn.ports import EmbedderPort
        from memlearn.adapters.in_memory.fake_embedding import FakeEmbedding
        assert isinstance(FakeEmbedding(self._model_ref()), EmbedderPort)

    def test_tokenizer_isinstance_protocol(self) -> None:
        from memlearn.ports import TokenizerAdapter
        from memlearn.adapters.in_memory.fake_tokenizer import FakeTokenizer
        assert isinstance(FakeTokenizer(), TokenizerAdapter)

    def test_llm_port_isinstance_protocol(self) -> None:
        from memlearn.ports import LLMPort
        from memlearn.adapters.in_memory.fake_llm_port import FakeLLMPort
        assert isinstance(FakeLLMPort(), LLMPort)

    def test_extractor_isinstance_protocol(self) -> None:
        from memlearn.ports import EntityExtractorPort
        from memlearn.adapters.in_memory.fake_entity_extractor import FakeEntityExtractor
        assert isinstance(FakeEntityExtractor(), EntityExtractorPort)

    def test_memory_kernel_in_memory_smoke(self) -> None:
        from memlearn.primitives import (
            AdmissionConfig, DeltaClampPolicy, EmbeddingModelRef,
            MemoryContext, MemoryRecord,
        )
        from memlearn.adapters.in_memory.fake_embedding import FakeEmbedding
        from memlearn.adapters.in_memory.fake_entity_extractor import FakeEntityExtractor
        from memlearn.adapters.in_memory.fake_llm_port import FakeLLMPort
        from memlearn.adapters.in_memory.fake_tokenizer import FakeTokenizer
        from memlearn.adapters.in_memory.graph_store import InMemoryGraphStore
        from memlearn.adapters.in_memory.kv_store import InMemoryKVStore
        from memlearn.adapters.in_memory.vector_store import InMemoryVectorStore
        from memlearn.services.admission_gate import AdmissionGate
        from memlearn.services.episodic_store import EpisodicStore
        from memlearn.services.semantic_store import SemanticStore
        from memlearn.services.associative_state_engine import AssociativeStateEngine
        from memlearn.services.conflict_flagger import ConflictFlagger
        from memlearn.services.retrieval_hit_tracker import RetrievalHitTracker
        from memlearn.services.consolidator import Consolidator
        from memlearn.services.memory_kernel import MemoryKernel

        model_ref = EmbeddingModelRef(model_id="fake", version="0.0.1", dimension=4)
        embedder = FakeEmbedding(model_ref)
        kv = InMemoryKVStore()
        graph = InMemoryGraphStore()
        semantic = SemanticStore(graph, DeltaClampPolicy())
        flagger = ConflictFlagger(graph, kv)
        kernel = MemoryKernel(
            gate=AdmissionGate(AdmissionConfig(gate_enabled=False)),
            episodic=EpisodicStore(InMemoryVectorStore()),
            semantic=semantic,
            associative=AssociativeStateEngine(),
            tokenizer=FakeTokenizer(),
            embedder=embedder,
            entity_extractor=FakeEntityExtractor(),
            flagger=flagger,
            tracker=RetrievalHitTracker(kv),
            consolidator=Consolidator(semantic, flagger, FakeLLMPort()),
        )
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            id=uuid4(),
            content="hello world",
            embedding=embedder.embed("hello world"),
            embedding_model=model_ref,
            timestamp=now,
            valid_from=now,
            valid_until=None,
            causal_parent_id=None,
            importance=0.5,
            surprise=0.5,
            source="user_input",
            lane_id="general",
        )
        assert kernel.observe(record) is True
        result = kernel.retrieve("hello", k=5, lane_id="general", token_budget=100)
        assert isinstance(result, MemoryContext)
