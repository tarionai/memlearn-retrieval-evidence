"""M-layer episodic memory service — VectorStoreAdapter wrapper with validity filtering."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import numpy as np

from memlearn.ports import SearchResult, VectorStoreAdapter
from memlearn.primitives import EmbeddingModelRef, MemoryRecord


class EpisodicStore:
    """M-layer episodic memory service.

    Wraps VectorStoreAdapter. Owns validity filtering logic — VectorStoreAdapter
    returns all matching vectors; EpisodicStore discards expired records in Python.

    Validity rule (canonical §5 MemoryRecord contract):
        valid_until is None → record is currently valid.
        valid_until is a datetime → record is valid only if valid_until > now().

    Over-fetches by 2x to compensate for filtered-out expired records.
    """

    def __init__(self, adapter: VectorStoreAdapter) -> None:
        self._adapter = adapter

    def store(self, record: MemoryRecord) -> None:
        """Upsert record into the vector store.

        Extracts valid_from, valid_until, lane_id into top-level adapter fields.
        Serializes datetime fields to ISO 8601 strings and UUID fields to str
        (dataclasses.asdict() does NOT convert UUID to str — L4).
        """
        payload = dataclasses.asdict(record)

        # Embedding is stored as the vector column AND cached in payload for OSAM reranking.
        # payload["_embedding"] is consumed by AssociativeStateEngine.rerank() (N-02).
        emb = payload.pop("embedding", None)
        if emb is not None:
            payload["_embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        # UUID fields — asdict() leaves these as UUID objects; json.dumps() rejects them.
        for field_name in ("id", "causal_parent_id"):
            value = payload.get(field_name)
            if value is not None:
                payload[field_name] = str(value)

        # datetime fields — serialize to ISO 8601 for JSON roundtrip.
        for field_name in ("timestamp", "valid_from", "valid_until"):
            value = payload.get(field_name)
            if isinstance(value, datetime):
                payload[field_name] = value.isoformat()

        self._adapter.upsert(
            id=record.id,
            embedding=record.embedding,
            payload=payload,
            lane_id=record.lane_id,
        )

    def retrieve(
        self,
        embedding: np.ndarray,
        k: int,
        lane_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Retrieve k nearest valid records. Filters expired records in Python."""
        raw_results = self._adapter.search(embedding, k=k * 2, lane_id=lane_id)
        now = datetime.now(timezone.utc)
        valid = [r for r in raw_results if self._is_valid(r, now)]
        return [self._to_record(r) for r in valid[:k]]

    def retrieve_raw(
        self,
        embedding: np.ndarray,
        k: int,
        lane_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Return validity-filtered raw SearchResults for OSAM reranking.

        Fetches 2k candidates, filters expired records via _is_valid(), returns up to k.
        Candidates carry payload["_embedding"] when store() was called after N-02.
        Does NOT call _to_record() — callers convert after optional reranking.
        """
        raw_results = self._adapter.search(embedding, k=k * 2, lane_id=lane_id)
        now = datetime.now(timezone.utc)
        valid = [r for r in raw_results if self._is_valid(r, now)]
        return valid[:k]

    def recent(self, lane_id: str, n: int) -> List[MemoryRecord]:
        """Return the n most recent valid records in lane. 2x overfetch to compensate for expired records."""
        raw = self._adapter.recent_by_timestamp(n * 2, lane_id)
        now = datetime.now(timezone.utc)
        valid = [r for r in raw if self._is_valid(r, now)]
        return [self._to_record(r) for r in valid[:n]]

    def supersede(self, id: UUID, valid_until: datetime) -> None:
        """Mark record as superseded by setting valid_until. Infrastructure only — not wired into integrate()."""
        self._adapter.update_valid_until(id, valid_until)

    def prune(self, id: UUID) -> None:
        """Hard-delete a record from the vector store."""
        self._adapter.delete(id)

    @staticmethod
    def _is_valid(result: SearchResult, now: datetime) -> bool:
        valid_until_str = result.payload.get("valid_until")
        if valid_until_str is None:
            return True
        valid_until = datetime.fromisoformat(valid_until_str)
        if valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=timezone.utc)
        return valid_until > now

    @staticmethod
    def _to_record(result: SearchResult) -> MemoryRecord:
        """Reconstruct MemoryRecord from SearchResult payload.

        Embedding is NOT stored in JSONB — reconstructed as a zero vector of the
        correct dimension (WP-04 Assumption A1; WP-05 can add re-fetch if reranking
        requires the actual vector).
        """
        payload = result.payload
        emr = EmbeddingModelRef(**payload["embedding_model"])
        zero_embedding = np.zeros(emr.dimension, dtype=np.float32)

        def _parse_dt(value: Optional[str]) -> Optional[datetime]:
            if value is None:
                return None
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        causal_str = payload.get("causal_parent_id")

        return MemoryRecord(
            id=UUID(payload["id"]) if isinstance(payload["id"], str) else payload["id"],
            content=payload["content"],
            embedding=zero_embedding,
            embedding_model=emr,
            timestamp=_parse_dt(payload["timestamp"]),
            valid_from=_parse_dt(payload["valid_from"]),
            valid_until=_parse_dt(payload.get("valid_until")),
            causal_parent_id=UUID(causal_str) if causal_str else None,
            importance=float(payload["importance"]),
            surprise=float(payload["surprise"]),
            source=payload["source"],
            lane_id=payload["lane_id"],
        )
