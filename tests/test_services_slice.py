"""Disclosed excerpt of the private ``tests/unit/test_services.py``.

PROVENANCE DISCLOSURE — this file is a curated excerpt, not byte-identical
to its origin. The original module also tests ``MemoryKernel`` (the memory
kernel proper) and the retrieval-ablation runner, neither of which ships in
this repository. The five test classes below exercise only the two service
modules this repository vendors — ``EpisodicStore`` and
``AssociativeStateEngine`` — and are copied verbatim per class, together
with the module-level helpers they use. See MANIFEST.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from memlearn.ports import SearchResult
from memlearn.primitives import EmbeddingModelRef, MemoryRecord
from memlearn.services.associative_state_engine import AssociativeStateEngine
from memlearn.services.episodic_store import EpisodicStore

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 4
_MODEL = EmbeddingModelRef(model_id="stub", version="0.1.0", dimension=_DIM)


def _make_record(
    embedding: np.ndarray | None = None,
    lane_id: str = "general",
    valid_until: datetime | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    emb = embedding if embedding is not None else np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return MemoryRecord(
        id=uuid4(),
        content="unit test content",
        embedding=emb,
        embedding_model=_MODEL,
        timestamp=now,
        valid_from=now,
        valid_until=valid_until,
        causal_parent_id=None,
        importance=0.5,
        surprise=0.5,
        source="user_input",
        lane_id=lane_id,
    )


def _search_result_with_embedding(
    emb: np.ndarray,
    lane_id: str = "general",
    valid_until_past: bool = False,
) -> SearchResult:
    """Build a fully-populated SearchResult carrying _embedding in payload."""
    uid = uuid4()
    now = datetime.now(timezone.utc)
    valid_until_str = (now - timedelta(hours=1)).isoformat() if valid_until_past else None
    return SearchResult(
        id=uid,
        score=0.1,
        payload={
            "id": str(uid),
            "content": f"content {uid}",
            "embedding_model": {"model_id": "stub", "version": "0.1.0", "dimension": _DIM},
            "timestamp": now.isoformat(),
            "valid_from": now.isoformat(),
            "valid_until": valid_until_str,
            "causal_parent_id": None,
            "importance": 0.5,
            "surprise": 0.5,
            "source": "test",
            "lane_id": lane_id,
            "_embedding": emb.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# test_retrieve_raw_applies_validity_filter
# ---------------------------------------------------------------------------

class TestRetrieveRaw:
    def test_retrieve_raw_applies_validity_filter(self):
        """retrieve_raw() excludes expired records (same validity rule as retrieve())."""
        expired_emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        valid_emb = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)

        expired_result = _search_result_with_embedding(expired_emb, valid_until_past=True)
        valid_result = _search_result_with_embedding(valid_emb)

        adapter = MagicMock()
        adapter.search.return_value = [expired_result, valid_result]

        store = EpisodicStore(adapter)
        raw = store.retrieve_raw(np.zeros(_DIM, dtype=np.float32), k=5)

        assert len(raw) == 1
        assert raw[0].id == valid_result.id

    def test_retrieve_raw_returns_search_results_not_records(self):
        """retrieve_raw() returns SearchResult objects, not MemoryRecord objects."""
        valid_result = _search_result_with_embedding(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

        adapter = MagicMock()
        adapter.search.return_value = [valid_result]

        store = EpisodicStore(adapter)
        raw = store.retrieve_raw(np.zeros(_DIM, dtype=np.float32), k=5)

        assert len(raw) == 1
        assert isinstance(raw[0], SearchResult)

    def test_retrieve_raw_overfetches_2k(self):
        """retrieve_raw() passes k*2 to the adapter."""
        adapter = MagicMock()
        adapter.search.return_value = []

        store = EpisodicStore(adapter)
        store.retrieve_raw(np.zeros(_DIM, dtype=np.float32), k=7)

        call_kwargs = adapter.search.call_args
        passed_k = call_kwargs.kwargs.get("k") or call_kwargs.args[1]
        assert passed_k == 14


# ---------------------------------------------------------------------------
# test_payload_embedding_present_after_store
# ---------------------------------------------------------------------------

class TestPayloadEmbedding:
    def test_payload_embedding_present_after_store(self):
        """store() persists _embedding in the JSONB payload for OSAM reranking."""
        adapter = MagicMock()
        store = EpisodicStore(adapter)

        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        record = _make_record(embedding=emb)
        store.store(record)

        adapter.upsert.assert_called_once()
        payload = adapter.upsert.call_args.kwargs["payload"]
        assert "_embedding" in payload
        assert payload["_embedding"] == pytest.approx(emb.tolist())

    def test_payload_embedding_does_not_duplicate_raw_embedding_key(self):
        """store() must not leave the raw 'embedding' key in the payload (it was popped)."""
        adapter = MagicMock()
        store = EpisodicStore(adapter)

        record = _make_record()
        store.store(record)

        payload = adapter.upsert.call_args.kwargs["payload"]
        assert "embedding" not in payload


class TestAlphaBlending:
    """Validates alpha-blended score fusion invariants from IMPL_WP10r N1/N2.

    Uses orthogonal unit-vector embeddings so OSAM scores are exactly 0.0 or 1.0 —
    no approximation required to verify ordering.
    """

    def _make_candidate(self, emb: np.ndarray, score: float) -> SearchResult:
        uid = uuid4()
        now = datetime.now(timezone.utc)
        return SearchResult(
            id=uid,
            score=score,
            payload={
                "id": str(uid),
                "content": f"content {uid}",
                "embedding_model": {"model_id": "stub", "version": "0.1.0", "dimension": _DIM},
                "timestamp": now.isoformat(),
                "valid_from": now.isoformat(),
                "valid_until": None,
                "causal_parent_id": None,
                "importance": 0.5,
                "surprise": 0.5,
                "source": "test",
                "lane_id": "lane",
                "_embedding": emb.tolist(),
            },
        )

    def test_alpha_1_equals_dense_order(self):
        """rerank(alpha=1.0) returns candidates sorted by candidate.score descending.

        OSAM is biased toward emb_a (low dense score), but alpha=1.0 ignores OSAM.
        Dense winner: result_b (score=0.9) must be first.
        """
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        result_a = self._make_candidate(emb_a, score=0.1)
        result_b = self._make_candidate(emb_b, score=0.9)

        engine = AssociativeStateEngine()
        for _ in range(10):
            engine.update("lane", emb_a, _MODEL)  # OSAM biased toward result_a

        out = engine.rerank("lane", [result_a, result_b], query, alpha=1.0)

        assert out[0].id == result_b.id, "alpha=1.0 must return dense-best (score=0.9) first"
        assert out[1].id == result_a.id

    def test_alpha_0_equals_pure_osam(self):
        """rerank(alpha=0.0) sorts by OSAM cosine only — matches pre-WP-10r behavior.

        Dense winner: result_b (score=0.9). OSAM winner: result_a (aligned with readout).
        alpha=0.0 must promote result_a first despite its lower dense score.
        """
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        result_a = self._make_candidate(emb_a, score=0.1)
        result_b = self._make_candidate(emb_b, score=0.9)

        engine = AssociativeStateEngine()
        for _ in range(10):
            engine.update("lane", emb_a, _MODEL)

        out = engine.rerank("lane", [result_b, result_a], query, alpha=0.0)

        assert out[0].id == result_a.id, "alpha=0.0 must return OSAM-best (aligned with readout) first"
        assert out[1].id == result_b.id

    def test_default_alpha_07_blends_scores(self):
        """Default alpha=0.7 produces ordering that differs from both pure-dense and pure-OSAM.

        Three candidates: a(dense=0.9, osam=0), b(dense=0.5, osam=0), c(dense=0.1, osam=1).
        Blended scores: a=0.63, c=0.37, b=0.35  ->  order [a, c, b].
        Pure dense:  [a, b, c]
        Pure OSAM:   [c, a, b]
        """
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        emb_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        query = emb_c  # readout will favour emb_c after OSAM warm-up

        result_a = self._make_candidate(emb_a, score=0.9)
        result_b = self._make_candidate(emb_b, score=0.5)
        result_c = self._make_candidate(emb_c, score=0.1)
        candidates = [result_a, result_b, result_c]

        engine = AssociativeStateEngine()
        for _ in range(10):
            engine.update("lane", emb_c, _MODEL)  # OSAM biased toward result_c

        dense_order = [r.id for r in engine.rerank("lane", candidates, query, alpha=1.0)]
        osam_order = [r.id for r in engine.rerank("lane", candidates, query, alpha=0.0)]
        blended_order = [r.id for r in engine.rerank("lane", candidates, query)]

        assert blended_order != dense_order, "alpha=0.7 blended order must differ from alpha=1.0"
        assert blended_order != osam_order, "alpha=0.7 blended order must differ from alpha=0.0"

        # Exact expected order: a(0.63) > c(0.37) > b(0.35)
        assert blended_order == [result_a.id, result_c.id, result_b.id]


class TestResetLane:
    """Validates reset_lane() invariants from IMPL_WP10r N1/N2."""

    def test_reset_lane_removes_state(self):
        """reset_lane() drops the lane state; subsequent rerank returns candidates unchanged."""
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        engine = AssociativeStateEngine()
        engine.update("test_lane", emb, _MODEL)
        assert "test_lane" in engine._states

        engine.reset_lane("test_lane")
        assert "test_lane" not in engine._states

        candidates = [_search_result_with_embedding(emb)]
        out = engine.rerank("test_lane", candidates, query)
        assert out is candidates  # no state -> guard returns same list object unchanged

    def test_reset_lane_nonexistent_noop(self):
        """reset_lane() on an absent lane_id raises no error."""
        engine = AssociativeStateEngine()
        engine.reset_lane("nonexistent_lane")  # must not raise


class TestNormalizeFlag:
    """Validates normalize=True per-query min-max normalization — WP-10y N3 Condition B.

    Uses L2-distance-style scores (lower = more similar) to confirm that
    normalize=True correctly inverts the direction and blends with OSAM cosine.
    """

    def _make_candidate(self, emb: np.ndarray, score: float) -> SearchResult:
        uid = uuid4()
        now = datetime.now(timezone.utc)
        return SearchResult(
            id=uid,
            score=score,
            payload={
                "id": str(uid),
                "content": f"content {uid}",
                "embedding_model": {"model_id": "stub", "version": "0.1.0", "dimension": _DIM},
                "timestamp": now.isoformat(),
                "valid_from": now.isoformat(),
                "valid_until": None,
                "causal_parent_id": None,
                "importance": 0.5,
                "surprise": 0.5,
                "source": "test",
                "lane_id": "lane",
                "_embedding": emb.tolist(),
            },
        )

    def test_normalize_true_inverts_l2_direction(self):
        """With normalize=True and alpha=1.0, lower L2 distance wins (not higher score).

        result_a: score=0.2 (close), result_b: score=1.5 (far).
        OSAM is neutral (zero updates). normalize=True must put result_a first.
        """
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        result_a = self._make_candidate(emb_a, score=0.2)
        result_b = self._make_candidate(emb_b, score=1.5)

        engine = AssociativeStateEngine()
        engine.update("lane", emb_a, _MODEL)

        out = engine.rerank("lane", [result_b, result_a], query, alpha=1.0, normalize=True)

        assert out[0].id == result_a.id, (
            "normalize=True alpha=1.0: smaller L2 distance (0.2) must rank above larger (1.5)"
        )
        assert out[1].id == result_b.id

    def test_normalize_true_equal_dense_uses_osam(self):
        """With normalize=True, equal L2 distances make OSAM the tiebreaker.

        result_a and result_b share score=1.0. OSAM is biased toward emb_c (query direction).
        result_a's embedding is aligned with emb_c; result_b is orthogonal.
        At alpha=0.5, result_a must win via normalized OSAM score.
        """
        emb_a = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        emb_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        query = emb_c

        result_a = self._make_candidate(emb_a, score=1.0)
        result_b = self._make_candidate(emb_b, score=1.0)

        engine = AssociativeStateEngine()
        for _ in range(10):
            engine.update("lane", emb_c, _MODEL)

        out = engine.rerank("lane", [result_b, result_a], query, alpha=0.5, normalize=True)

        assert out[0].id == result_a.id, (
            "normalize=True equal dense scores: OSAM-aligned candidate must win"
        )

    def test_normalize_false_and_true_agree_when_constant_dense(self):
        """normalize=True and normalize=False produce same order when all dense scores equal.

        When dense scores are identical, normalization is a no-op — OSAM decides in both.
        """
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        query = emb_a

        result_a = self._make_candidate(emb_a, score=0.5)
        result_b = self._make_candidate(emb_b, score=0.5)

        engine = AssociativeStateEngine()
        for _ in range(10):
            engine.update("lane", emb_a, _MODEL)

        out_false = engine.rerank("lane", [result_b, result_a], query, alpha=0.5, normalize=False)
        out_true = engine.rerank("lane", [result_b, result_a], query, alpha=0.5, normalize=True)

        assert [r.id for r in out_false] == [r.id for r in out_true], (
            "normalize flag must not change ordering when dense scores are equal"
        )
