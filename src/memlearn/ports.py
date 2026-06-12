"""C-layer Protocol contracts for external storage adapters.

These are typing.Protocol interfaces. They define the contract the kernel
depends on, not the implementation. Concrete adapters live in memlearn.adapters.

Base signatures from EXTERNAL_MEMORY_LEARNING_KERNEL_CANONICAL.md §9.
Extended with §10 (weight mutations) and §11 (forget operations) per Decision D7.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterator, List, Optional, Protocol, Tuple, runtime_checkable
from uuid import UUID

import numpy as np

from memlearn.primitives import DeltaClampPolicy, EmbeddingModelRef, ExtractionResult


@dataclass(frozen=True)
class SearchResult:
    """Return value from VectorStoreAdapter.search(). score = L2 distance."""

    id: UUID
    score: float
    payload: dict


class MergeStrategy(str, Enum):
    """Graph topology merge choice for GraphStoreAdapter.merge_nodes()."""

    KEEP_A = "keep_a"
    KEEP_B = "keep_b"
    MERGE = "merge"


@runtime_checkable
class VectorStoreAdapter(Protocol):
    def upsert(
        self,
        id: UUID,
        embedding: np.ndarray,
        payload: dict,
        lane_id: str = "default",  # Decision D3 — backward-compatible; isinstance checks presence not signature
    ) -> None: ...
    def search(
        self,
        embedding: np.ndarray,
        k: int,
        lane_id: Optional[str] = None,  # canonical §8 — 5-lane taxonomy
    ) -> List[SearchResult]: ...
    def delete(self, id: UUID) -> None: ...
    def update_valid_until(self, id: UUID, valid_until: Optional[datetime]) -> None: ...
    def recent_by_timestamp(self, n: int, lane_id: Optional[str] = None) -> List[SearchResult]: ...


@runtime_checkable
class GraphStoreAdapter(Protocol):
    def add_node(self, id: UUID, label: str, properties: dict) -> None: ...
    def add_edge(self, src: UUID, dst: UUID, relation: str, weight: float) -> None: ...
    def ppr_search(self, seed_nodes: List[UUID], alpha: float, k: int) -> List[UUID]: ...
    def merge_nodes(self, a: UUID, b: UUID, strategy: MergeStrategy) -> UUID: ...
    def delete_node(self, id: UUID) -> None: ...  # canonical §11 ForgetPolicy.Hard
    def apply_weight_deltas(
        self,
        node_deltas: dict[UUID, float],
        edge_deltas: dict[tuple[UUID, UUID, str], float],
        policy: DeltaClampPolicy,
    ) -> None: ...
    def update_node_properties(self, id: UUID, properties: dict) -> None: ...
    def get_all_node_weights(self) -> dict[UUID, float]: ...


@runtime_checkable
class LLMPort(Protocol):
    """Route all kernel LLM calls through this port. No direct client instantiation in M-layer services."""

    def complete(self, prompt: str, *, max_tokens: int, system: str = "") -> str: ...


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Token counting for token_budget enforcement in MemoryKernel.retrieve()."""

    def count_tokens(self, text: str) -> int: ...


@runtime_checkable
class EntityExtractorPort(Protocol):
    """Entity/relation extraction contract. Consumed by integrate() (WP-05 fully implements)."""

    def extract(self, text: str) -> ExtractionResult: ...


@runtime_checkable
class EmbedderPort(Protocol):
    """Dense text embedding contract.

    Added in WP-04 Decision D2 — not in canonical §5 but required by
    MemoryKernel.observe() when embedding is not pre-computed.
    """

    def embed(self, text: str) -> np.ndarray: ...


@runtime_checkable
class AssociativeRankingCandidate(Protocol):
    """Minimal interface for a mechanism that updates associative state and reranks candidates.

    MemoryKernel depends on this Protocol, not on AssociativeStateEngine directly.
    Structural subtyping: AssociativeStateEngine satisfies this protocol unchanged.
    """

    def update(
        self, lane_id: str, embedding: np.ndarray, model: EmbeddingModelRef
    ) -> None: ...

    def rerank(
        self,
        lane_id: str,
        candidates: List[SearchResult],
        query_embedding: np.ndarray,
    ) -> List[SearchResult]: ...


@runtime_checkable
class KVStoreAdapter(Protocol):
    def set(self, key: str, value: bytes) -> None: ...
    def get(self, key: str) -> Optional[bytes]: ...
    def delete(self, key: str) -> None: ...  # canonical §11 ForgetPolicy.Hard
    def scan_prefix(self, prefix: str) -> Iterator[Tuple[str, bytes]]: ...
