"""Self-contained retrieval metrics for the reproduction pack.

Definitions match the frozen eval contract: MRR is cutoff-free; recall is
pool-relative (denominator = judged-relevant in the frozen qrels, never the
corpus); NDCG IDCG is the best ranking of the full judged-relevant set.
"""
from __future__ import annotations

import math

K_VALUES = (5, 10)


def _dcg(graded_in_rank_order: list[int]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(graded_in_rank_order, start=1))


def query_metrics(retrieved: list[str], grades: dict[str, int]) -> dict:
    """Per-query metrics over the frozen pool. grades: candidate_id -> 0/1/2."""
    relevant = {cid for cid, g in grades.items() if g >= 1}
    ideal_full = sorted((g for g in grades.values() if g > 0), reverse=True)

    mrr = 0.0
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            mrr = 1.0 / rank
            break

    record = {"mrr": round(mrr, 4)}
    for k in K_VALUES:
        topk = retrieved[:k]
        record[f"recall@{k}"] = round(len(set(topk) & relevant) / max(len(relevant), 1), 4)
        idcg = _dcg(ideal_full[:k])
        record[f"ndcg@{k}"] = round(_dcg([grades.get(c, 0) for c in topk]) / idcg, 4) if idcg else 0.0
    record["p@10"] = round(len(set(retrieved[:10]) & relevant) / 10, 4)
    return record


def aggregate(records: list[dict]) -> dict:
    """Mean of each metric over per-query records."""
    if not records:
        return {}
    keys = set().union(*(r.keys() for r in records))
    agg = {k: round(sum(r[k] for r in records if k in r) / sum(1 for r in records if k in r), 4) for k in keys}
    agg["query_count"] = len(records)
    return agg
