"""M-layer OSAM (Online Sequence Associative Memory) manager — per-lane S-matrix updates."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from memlearn.ports import SearchResult
from memlearn.primitives import EmbeddingModelRef


@dataclass
class AssociativeState:
    """
    Online state matrix S ∈ ℝ^(r×r).
    Source: δ-mem OSAM, ported to pure-external operation.
    One state per explicitly-named lane. Lanes are never inferred.

    ROLE: RETRIEVAL RERANKER only.
    Readout r_t = S_{t-1} q_t is cosine-scored against retrieved
    candidate embeddings to rerank them. Not decoded to text.
    """

    lane_id: str
    S: np.ndarray               # shape (r, r)
    retention: np.ndarray       # λ, shape (r,)
    write_strength: np.ndarray  # β, shape (r,)
    step: int
    embedding_model: EmbeddingModelRef    # must match ingestion encoder


class AssociativeStateEngine:
    """Per-session OSAM manager.

    One AssociativeState per lane. States are initialized on first update for that
    lane and are never persisted — ephemeral per MemoryKernel session (Decision D7).

    Update rule (canonical §8):
        S_t = λ ⊙ S_{t-1} + β ⊗ (v_t v_t^T)
          where ⊙ = element-wise broadcast (retention per row),
                ⊗ = element-wise broadcast (write_strength per row).

    Readout:
        r_t = S_{t-1} q_t

    Warm-up window: effective memory horizon is 1/(1-λ) ≈ 10 records at default λ=0.9.
    Records updated more than ~10 steps before a query contribute <37% of their original
    weight. Callers MUST NOT warm OSAM with more than ~1/(1-λ) records before any query.
    For batch contexts, call reset_lane(lane_id) before per-query warm-up.

    Score fusion: rerank(alpha) blends dense score and OSAM readout cosine.
        alpha=0.0  -> pure OSAM (OSAM signal fully overrides dense; diagnostic only)
        alpha=1.0  -> pure dense (OSAM readout ignored; use as correctness baseline)
        alpha=0.7  -> canonical default (70% dense / 30% OSAM; production-recommended)
                   Source: EXTERNAL_MEMORY_LEARNING_KERNEL_CANONICAL.md §8
    Always use rerank(alpha > 0) in production contexts with mixed-topic sessions.

    Candidates without _embedding in payload are scored 0.0 for the OSAM component
    and fall to the bottom under pure-OSAM (WP-04 limitation — see IMPL_WP04).
    """

    def __init__(self) -> None:
        self._states: Dict[str, AssociativeState] = {}

    def update(
        self, lane_id: str, embedding: np.ndarray, model: EmbeddingModelRef
    ) -> None:
        """Update the lane's S-matrix with the new embedding observation."""
        state = self._get_or_init(lane_id, embedding, model)
        # S_t = λ ⊙ S_{t-1} + β ⊗ (v v^T)
        outer = np.outer(embedding, embedding)
        new_S = state.retention[:, None] * state.S + state.write_strength[:, None] * outer
        updated = dataclasses.replace(state, S=new_S, step=state.step + 1)
        self._states[lane_id] = updated

    def rerank(
        self,
        lane_id: str,
        candidates: List[SearchResult],
        query_embedding: np.ndarray,
        alpha: float = 0.7,
        normalize: bool = False,
    ) -> List[SearchResult]:
        """Rerank candidates using alpha-blended dense and OSAM scores.

        normalize=False (default):
            final_score = alpha * candidate.score + (1 - alpha) * osam_cosine_score
            Scores sorted descending. Compatible with FakeEmbedding (constant L2 distances).

        normalize=True (recommended with real embeddings):
            Dense score is an L2 distance (lower = more similar). Per-query min-max
            normalization inverts and scales it to [0, 1] so higher = more similar.
            OSAM cosine is similarly min-max scaled to [0, 1] per query.
            final_score = alpha * norm_dense + (1 - alpha) * norm_osam, sorted descending.
            Use normalize=True whenever candidate.score is an L2 distance from a real
            embedding model — this is required for the blend ratio to reflect the stated
            alpha weight.

        alpha=1.0 -> pure dense order
        alpha=0.0 -> pure OSAM order (diagnostic only)
        alpha=0.7 -> canonical default (70% dense / 30% OSAM)
                  Source: EXTERNAL_MEMORY_LEARNING_KERNEL_CANONICAL.md §8

        Candidates without payload["_embedding"] receive osam_cosine_score=0.0.
        If no OSAM state exists for lane_id, candidates are returned unchanged.
        If readout norm < 1e-10 (degenerate), candidates are returned unchanged.
        """
        if lane_id not in self._states or not candidates:
            return candidates

        state = self._states[lane_id]
        readout = state.S @ query_embedding  # shape (r,)
        norm_readout = np.linalg.norm(readout)
        if norm_readout < 1e-10:
            return candidates  # degenerate readout — no rerank

        def _osam_score(candidate: SearchResult) -> float:
            emb = candidate.payload.get("_embedding")
            if emb is None:
                return 0.0
            emb_arr = np.array(emb, dtype=np.float32)
            norm_emb = np.linalg.norm(emb_arr)
            if norm_emb < 1e-10:
                return 0.0
            return float(np.dot(readout, emb_arr) / (norm_readout * norm_emb))

        if not normalize:
            def _final_score(c: SearchResult) -> float:
                return alpha * c.score + (1.0 - alpha) * _osam_score(c)
            return sorted(candidates, key=_final_score, reverse=True)

        # normalize=True: per-query min-max normalization, invert L2 distance direction.
        raw_dense = np.array([c.score for c in candidates], dtype=np.float64)
        raw_osam = np.array([_osam_score(c) for c in candidates], dtype=np.float64)

        d_min, d_max = raw_dense.min(), raw_dense.max()
        o_min, o_max = raw_osam.min(), raw_osam.max()

        # Invert L2 distance so higher normalized score = more similar.
        if d_max > d_min:
            norm_dense = (d_max - raw_dense) / (d_max - d_min)
        else:
            norm_dense = np.ones_like(raw_dense)

        if o_max > o_min:
            norm_osam = (raw_osam - o_min) / (o_max - o_min)
        else:
            norm_osam = np.ones_like(raw_osam)

        final_scores = alpha * norm_dense + (1.0 - alpha) * norm_osam
        order = np.argsort(final_scores)[::-1]
        return [candidates[i] for i in order]

    def reset_lane(self, lane_id: str) -> None:
        """Discard the accumulated S-matrix for lane_id.

        Used for per-query warm-up isolation in benchmarking.
        No-op if lane_id is not in _states.
        """
        self._states.pop(lane_id, None)

    def _get_or_init(
        self, lane_id: str, embedding: np.ndarray, model: EmbeddingModelRef
    ) -> AssociativeState:
        if lane_id not in self._states:
            r = embedding.shape[0]
            self._states[lane_id] = AssociativeState(
                lane_id=lane_id,
                S=np.zeros((r, r), dtype=np.float32),
                retention=np.ones(r, dtype=np.float32) * 0.9,      # λ default
                write_strength=np.ones(r, dtype=np.float32) * 0.1,  # β default
                step=0,
                embedding_model=model,
            )
        return self._states[lane_id]
