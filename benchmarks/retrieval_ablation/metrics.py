"""Pure retrieval metric functions for ablation benchmarking.

All functions are deterministic, stateless, and side-effect-free.
Input types use only stdlib primitives so this module has no memlearn imports.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RetrievalRun:
    """One query result for one retrieval variant."""
    query_id: str
    variant: str
    # IDs of records considered ground-truth relevant for this query.
    relevant_ids: set[str]
    # IDs of retrieved items in ranked order (index 0 = top rank).
    retrieved_ids: list[str]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved set."""
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant item (0.0 if none found)."""
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Binary relevance: 1 if retrieved item is relevant, 0 otherwise.
    Ideal DCG assumes all relevant items at positions 1..min(|relevant|, k).
    """
    def _dcg(ids: list[str], rel: set[str], cutoff: int) -> float:
        return sum(
            (1.0 / math.log2(rank + 2))  # rank is 0-indexed; log2(2)=1 at rank=0
            for rank, item_id in enumerate(ids[:cutoff])
            if item_id in rel
        )

    dcg = _dcg(retrieved, relevant, k)
    ideal_count = min(len(relevant), k)
    idcg = _dcg(list(relevant)[:ideal_count], relevant, ideal_count)
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics(runs: list[RetrievalRun], k: int) -> dict[str, dict[str, float]]:
    """Compute mean recall@k, precision@k, MRR, NDCG@k per variant.

    Returns: {variant_name: {recall_at_k, precision_at_k, mrr, ndcg_at_k}}
    """
    buckets: dict[str, list[RetrievalRun]] = defaultdict(list)
    for run in runs:
        buckets[run.variant].append(run)

    result: dict[str, dict[str, float]] = {}
    for variant, variant_runs in buckets.items():
        n = len(variant_runs)
        result[variant] = {
            "recall_at_k": sum(recall_at_k(r.retrieved_ids, r.relevant_ids, k) for r in variant_runs) / n,
            "precision_at_k": sum(precision_at_k(r.retrieved_ids, r.relevant_ids, k) for r in variant_runs) / n,
            "mrr": sum(reciprocal_rank(r.retrieved_ids, r.relevant_ids) for r in variant_runs) / n,
            "ndcg_at_k": sum(ndcg_at_k(r.retrieved_ids, r.relevant_ids, k) for r in variant_runs) / n,
        }
    return result
