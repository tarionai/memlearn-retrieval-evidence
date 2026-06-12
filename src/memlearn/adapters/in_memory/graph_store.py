"""In-memory GraphStoreAdapter for deterministic testing.

No external dependencies except numpy for PPR power-iteration (Decision D3).
No network, no SQL, no psycopg2.
"""
from __future__ import annotations

from typing import List
from uuid import UUID

import numpy as np

from memlearn.ports import MergeStrategy
from memlearn.primitives import DeltaClampPolicy


class InMemoryGraphStore:
    """Pure-Python GraphStoreAdapter.

    State
    -----
    _nodes   : dict[UUID, tuple[str, dict]]        — (label, properties)
    _weights : dict[UUID, float]                   — node weights, initialised 0.0
    _edges   : list[tuple[UUID, UUID, str, float]] — (src, dst, relation, weight)
    """

    def __init__(self) -> None:
        self._nodes: dict[UUID, tuple[str, dict]] = {}
        self._weights: dict[UUID, float] = {}
        self._edges: list[tuple[UUID, UUID, str, float]] = []

    # ------------------------------------------------------------------
    # Node primitives
    # ------------------------------------------------------------------

    def add_node(self, id: UUID, label: str, properties: dict) -> None:
        self._nodes[id] = (label, dict(properties))
        self._weights[id] = 0.0

    def delete_node(self, id: UUID) -> None:
        """Hard-delete node and all incident edges (ForgetPolicy.Hard §11)."""
        self._nodes.pop(id, None)
        self._weights.pop(id, None)
        self._edges = [
            (src, dst, rel, w)
            for src, dst, rel, w in self._edges
            if src != id and dst != id
        ]

    def update_node_properties(self, id: UUID, properties: dict) -> None:
        """Merge props into stored dict; incoming values win on conflict.

        0-row update (node absent) is silently ignored — matches pg behaviour.
        """
        if id not in self._nodes:
            return
        label, existing = self._nodes[id]
        self._nodes[id] = (label, {**existing, **properties})

    def get_all_node_weights(self) -> dict[UUID, float]:
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Edge primitives
    # ------------------------------------------------------------------

    def add_edge(self, src: UUID, dst: UUID, relation: str, weight: float) -> None:
        self._edges.append((src, dst, relation, weight))

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_nodes(self, a: UUID, b: UUID, strategy: MergeStrategy) -> UUID:
        """Merge graph nodes by topology strategy; return surviving node UUID."""
        if strategy == MergeStrategy.KEEP_A:
            self._reassign_edges(b, a)
            self._nodes.pop(b, None)
            self._weights.pop(b, None)
            return a

        if strategy == MergeStrategy.KEEP_B:
            self._reassign_edges(a, b)
            self._nodes.pop(a, None)
            self._weights.pop(a, None)
            return b

        # MERGE — union properties (B wins conflicts); keep both nodes & edges; return a
        label_a, props_a = self._nodes.get(a, ("", {}))
        _label_b, props_b = self._nodes.get(b, ("", {}))
        self._nodes[a] = (label_a, {**props_a, **props_b})
        return a

    def _reassign_edges(self, from_id: UUID, to_id: UUID) -> None:
        """Redirect every edge pointing to/from from_id so it points to/from to_id."""
        self._edges = [
            (
                to_id if src == from_id else src,
                to_id if dst == from_id else dst,
                rel,
                w,
            )
            for src, dst, rel, w in self._edges
        ]

    # ------------------------------------------------------------------
    # Weight deltas
    # ------------------------------------------------------------------

    def apply_weight_deltas(
        self,
        node_deltas: dict[UUID, float],
        edge_deltas: dict[tuple[UUID, UUID, str], float],
        policy: DeltaClampPolicy,
    ) -> None:
        """Apply bounded weight mutations.

        Node deltas:  new = clamp(old + delta, min_weight, max_weight).
                      Missing node IDs are silently skipped.
        Edge deltas:  find current max weight for (src, dst, relation);
                      clamp(current + delta); append new entry to _edges so
                      PPR (which uses max across parallel edges) sees the update.
        """
        for node_id, delta in node_deltas.items():
            if node_id not in self._weights:
                continue
            new = max(
                policy.min_weight,
                min(policy.max_weight, self._weights[node_id] + delta),
            )
            self._weights[node_id] = new

        for (src, dst, relation), delta in edge_deltas.items():
            current = max(
                (w for s, d, r, w in self._edges if s == src and d == dst and r == relation),
                default=0.0,
            )
            new_weight = max(
                policy.min_weight,
                min(policy.max_weight, current + delta),
            )
            self._edges.append((src, dst, relation, new_weight))

    # ------------------------------------------------------------------
    # PPR search (power-iteration, Decision D3)
    # ------------------------------------------------------------------

    def ppr_search(self, seed_nodes: List[UUID], alpha: float, k: int) -> List[UUID]:
        """Personalised PageRank — 20 iterations of power method.

        Returns top-k UUIDs by final score, excluding seed nodes.
        Returns [] immediately when no edges exist.
        """
        if not self._edges:
            return []

        # Build adjacency: adj[src][dst] = max weight across parallel edges
        adj: dict[UUID, dict[UUID, float]] = {}
        for src, dst, _rel, weight in self._edges:
            row = adj.setdefault(src, {})
            if dst not in row or weight > row[dst]:
                row[dst] = weight

        # Universe of nodes (union of _nodes keys and edge endpoints)
        all_nodes: set[UUID] = set(self._nodes)
        for src, dst_map in adj.items():
            all_nodes.add(src)
            all_nodes.update(dst_map)
        node_list = list(all_nodes)
        idx = {n: i for i, n in enumerate(node_list)}
        n = len(node_list)

        # Row-stochastic transition matrix A where A[i,j] = normalised prob from i to j
        A = np.zeros((n, n), dtype=np.float64)
        for src, dst_map in adj.items():
            total = sum(dst_map.values())
            if total > 0.0:
                i = idx[src]
                for dst, w in dst_map.items():
                    A[i, idx[dst]] = w / total

        # Personalisation vector: uniform over seed nodes present in universe
        p = np.zeros(n, dtype=np.float64)
        valid_seeds = [idx[s] for s in seed_nodes if s in idx]
        if valid_seeds:
            for i in valid_seeds:
                p[i] = 1.0 / len(valid_seeds)

        # Power iteration: v_new[i] = alpha * sum_j(A[j,i] * v[j]) + (1-alpha) * p[i]
        v = p.copy()
        for _ in range(20):
            v = alpha * (A.T @ v) + (1.0 - alpha) * p

        # Return top-k excluding seed set
        seed_set = set(seed_nodes)
        scored = sorted(
            ((v[i], node) for i, node in enumerate(node_list) if node not in seed_set),
            key=lambda x: x[0],
            reverse=True,
        )
        return [node for _, node in scored[:k]]
