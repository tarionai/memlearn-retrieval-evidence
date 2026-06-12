"""InMemoryVectorStore — in-memory VectorStoreAdapter for deterministic testing.

Pure Python, no network, no database, no filesystem reads.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

import numpy as np

from memlearn.ports import SearchResult


class InMemoryVectorStore:
    """In-memory implementation of VectorStoreAdapter.

    Internal state per record: embedding, payload, lane_id, valid_until,
    recorded_at (UTC wall-clock at upsert time).

    Not thread-safe — test-only use.
    """

    def __init__(self) -> None:
        self._records: Dict[UUID, dict] = {}

    def upsert(
        self,
        id: UUID,
        embedding: np.ndarray,
        payload: dict,
        lane_id: str = "default",
    ) -> None:
        """Insert or replace the record for id. recorded_at is reset to UTC now."""
        self._records[id] = {
            "embedding": embedding.copy(),
            "payload": dict(payload),
            "lane_id": lane_id,
            "valid_until": None,
            "recorded_at": datetime.now(tz=timezone.utc),
        }

    def search(
        self,
        embedding: np.ndarray,
        k: int,
        lane_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Return at most k records ordered by ascending L2 distance to embedding."""
        candidates = []
        for uid, rec in self._records.items():
            if lane_id is not None and rec["lane_id"] != lane_id:
                continue
            dist = float(np.linalg.norm(rec["embedding"] - embedding))
            candidates.append((dist, uid, rec))

        candidates.sort(key=lambda t: t[0])
        return [
            SearchResult(id=uid, score=dist, payload=self._build_payload(rec))
            for dist, uid, rec in candidates[:k]
        ]

    def delete(self, id: UUID) -> None:
        """Remove the record for id. No-op if absent."""
        self._records.pop(id, None)

    def update_valid_until(
        self,
        id: UUID,
        valid_until: Optional[datetime],
    ) -> None:
        """Set valid_until on the stored record.

        Does not filter at search time — filtering is EpisodicStore's responsibility.
        """
        if id in self._records:
            self._records[id]["valid_until"] = valid_until

    def recent_by_timestamp(
        self,
        n: int,
        lane_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Return n most recently upserted records, ordered by recorded_at DESC.

        score is 0.0 for all results — no distance metric applies here.
        """
        candidates = [
            (uid, rec)
            for uid, rec in self._records.items()
            if lane_id is None or rec["lane_id"] == lane_id
        ]
        candidates.sort(key=lambda t: t[1]["recorded_at"], reverse=True)
        return [
            SearchResult(id=uid, score=0.0, payload=self._build_payload(rec))
            for uid, rec in candidates[:n]
        ]

    def _build_payload(self, rec: dict) -> dict:
        """Return a copy of the stored payload merged with lane/temporal fields.

        EpisodicStore._to_record() requires valid_from, valid_until, lane_id
        to be present in every returned payload (invariant 8).
        valid_from is expected in the stored payload (caller's responsibility).
        valid_until and lane_id are injected here from the internal record.
        """
        built = dict(rec["payload"])
        built["lane_id"] = rec["lane_id"]
        vu = rec["valid_until"]
        built["valid_until"] = vu.isoformat() if vu is not None else None
        return built
