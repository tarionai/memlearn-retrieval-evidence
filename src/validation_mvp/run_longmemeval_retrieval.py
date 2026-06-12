"""LongMemEval-S retrieval-stage evaluation (WP-110 Phase P1).

Measures whether external associative memory (OSAM rerank, α=0.7 frozen) adds
retrieval value over a dense bi-encoder on an independently-constructed public
benchmark — with the controls the prior local-corpus session proved necessary:

  * per-question store isolation (questions are independent),
  * non-degeneracy guard (evidence ⊊ history, enforced by construction),
  * dedup of identical embeddings (kills tie-inflation artifacts),
  * a CHANCE baseline (random top-k MRR at the pool's relevant density),
  * λ-recency bucketing (OSAM's ~10-step horizon vs ~500-turn histories),
  * pure-OSAM (α=0) to expose raw signal under the 70/30 blend,
  * a cross-encoder variant to detect benchmark-saturation (no reranking headroom),
  * a dense sanity tripwire (catches inverted ranking).

Retrieval-stage only; no LLM answerer. Reuses metrics + bootstrap from the
existing retrieval_ablation harness — nothing duplicated.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import UUID

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from benchmarks.retrieval_ablation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from benchmarks.retrieval_ablation.tournament_base import bootstrap_ci
from memlearn.adapters.in_memory.sentence_transformer_embedder import SentenceTransformerEmbedder
from memlearn.adapters.in_memory.vector_store import InMemoryVectorStore
from memlearn.primitives import MemoryRecord
from memlearn.services.associative_state_engine import AssociativeStateEngine
from memlearn.services.episodic_store import EpisodicStore
from validation_mvp.bench_datasets.longmemeval_adapter import load_longmemeval_records

_ALPHA = 0.7
_NORMALIZE = True
_KS = (1, 5, 10)
_POOL_K = 50
_RECENCY_BUCKETS = ((0, 10), (11, 50), (51, 200), (201, 10**9))
_CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Embedding (batched + per-record disk cache) ───────────────────────────────


def _embedder():
    embedder = SentenceTransformerEmbedder()
    return embedder, embedder._get_model(), embedder.embedding_model


def _embed_record(record, model, cache_dir: Path):
    """Return (query_vec, turn_matrix). Cached per record_id as .npz."""
    cache = cache_dir / f"{record.record_id}.npz"
    texts = [c.text for c in record.history_chunks]
    if cache.exists():
        data = np.load(cache)
        if data["turns"].shape[0] == len(texts):
            return data["query"].astype(np.float32), data["turns"].astype(np.float32)
    query = model.encode(record.question, normalize_embeddings=True, show_progress_bar=False)
    turns = model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                         batch_size=128) if texts else np.zeros((0, query.shape[0]))
    turns = np.asarray(turns, dtype=np.float32)
    for i, t in enumerate(texts):                      # match production embed(): empty → zeros
        if not t.strip():
            turns[i] = 0.0
    query = np.asarray(query, dtype=np.float32)
    np.savez(cache, query=query, turns=turns)
    return query, turns


# ── Per-question seeding (isolation + OSAM warmup + dedup) ─────────────────────


def _make_record(uid: UUID, chunk, emb, model_ref, now) -> MemoryRecord:
    return MemoryRecord(
        id=uid, content=chunk.text, embedding=emb, embedding_model=model_ref,
        timestamp=now, valid_from=now, valid_until=None, causal_parent_id=None,
        importance=0.5, surprise=0.5, source="benchmark", lane_id="bench",
    )


def _seed_question(record, turn_matrix, model_ref):
    """Fresh store + engine. OSAM sees ALL turns chronologically; the vector store
    holds one doc per unique embedding (dedup). Returns store, engine, relevant,
    n_collapsed, n_docs, docs — where ``docs`` is the ordered list of surviving
    (uid, content) pairs (one per unique embedding), used to build the lexical
    BM25 index over exactly the same deduped doc set the dense store holds, so
    BM25 index position ↔ uid is aligned by construction."""
    store = EpisodicStore(InMemoryVectorStore())
    engine = AssociativeStateEngine()
    now = datetime.now(timezone.utc)
    seen: dict[bytes, str] = {}
    relevant: set[str] = set()
    docs: list[tuple[str, str]] = []
    collapsed = 0
    for chunk, emb in zip(record.history_chunks, turn_matrix):
        engine.update("bench", emb, model_ref)
        key = np.round(emb, 5).tobytes()
        if key in seen:
            collapsed += 1
            if chunk.is_evidence:
                relevant.add(seen[key])
            continue
        uid = UUID(int=chunk.order)
        seen[key] = str(uid)
        store.store(_make_record(uid, chunk, emb, model_ref, now))
        docs.append((str(uid), chunk.text))
        if chunk.is_evidence:
            relevant.add(str(uid))
    return store, engine, relevant, collapsed, len(seen), docs


# ── Scoring ───────────────────────────────────────────────────────────────────


def _ids(results) -> list[str]:
    return [str(r.payload.get("id", "")) for r in results]


# ── Lexical (BM25) + hybrid Reciprocal Rank Fusion ────────────────────────────
#
# Lexical backend = bm25s (already a dependency). The fusion pattern (dense ANN ∪
# lexical top-k → RRF) is backend-agnostic: bm25s is swappable for Postgres
# full-text search (to_tsvector/ts_rank), Elasticsearch, or OpenSearch without
# touching the fusion logic — only the lexical retriever changes. Postgres FTS was
# the on-stack first choice; no local Postgres instance was reachable in this run,
# so the swappable bm25s backend is used and reported as such.


def _bm25_ranking(query: str, docs: list[tuple[str, str]], pool_k: int) -> list[str]:
    """Lexical-only ranking: BM25 over the full deduped doc set (NOT the dense pool),
    returning up to pool_k uids in lexical score order. Indexing the full corpus is
    what lets the lexical arm surface documents the dense ANN missed."""
    if not docs:
        return []
    import bm25s

    contents = [c for _uid, c in docs]
    uids = [u for u, _c in docs]
    corpus_tokens = bm25s.tokenize(contents, stopwords=None, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
    results, _scores = retriever.retrieve(
        query_tokens, k=min(pool_k, len(contents)), show_progress=False)
    return [uids[int(i)] for i in results[0].tolist()]


def _rrf_fuse(dense_ids: list[str], lexical_ids: list[str], top_k: int,
              rrf_k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion of two ranked id lists. rrf_k=60 is the standard
    constant (matches benchmarks/retrieval_ablation/run_bm25_hybrid.py)."""
    scores: dict[str, float] = {}
    for rank, uid in enumerate(dense_ids, start=1):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, uid in enumerate(lexical_ids, start=1):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]


def _cross_rerank(cross, question, id_to_content, ids):
    """Reorder ``ids`` by cross-encoder relevance to ``question``."""
    if not ids:
        return []
    scores = cross.predict([(question, id_to_content.get(uid, "")) for uid in ids])
    order = sorted(range(len(ids)), key=lambda j: float(scores[j]), reverse=True)
    return [ids[j] for j in order]


def _score_question(store, engine, query, question, relevant, pool_k, shuffle_seed,
                    cross, docs, with_hybrid, lexical_ranker=None, question_id=""):
    raw = store.retrieve_raw(query, k=pool_k, lane_id="bench")
    dense_ids = _ids(raw)
    t0 = perf_counter()
    osam = _ids(engine.rerank("bench", raw, query, alpha=_ALPHA, normalize=_NORMALIZE))
    rerank_ms = (perf_counter() - t0) * 1000.0
    pure = _ids(engine.rerank("bench", raw, query, alpha=0.0, normalize=_NORMALIZE))
    rng = np.random.default_rng(shuffle_seed)
    chance = [dense_ids[i] for i in rng.permutation(len(raw))] if raw else []
    ranked = {"chance": chance, "dense_only": dense_ids,
              "dense_plus_osam": osam, "pure_osam": pure}

    id_to_content = {uid: content for uid, content in docs}

    if with_hybrid:
        # Lexical arm is pluggable: bm25s / Postgres FTS / Elasticsearch / OpenSearch.
        # Same call shape; question_id lets external engines isolate IDF per question.
        rank_lex = lexical_ranker if lexical_ranker is not None else (
            lambda q, d, k, qid="": _bm25_ranking(q, d, k))
        lexical_ids = rank_lex(question, docs, pool_k, question_id)
        hybrid_ids = _rrf_fuse(dense_ids, lexical_ids, pool_k)
        ranked["lexical_only"] = lexical_ids
        ranked["hybrid_rrf"] = hybrid_ids

    if cross is not None:
        ranked["cross_encoder"] = _cross_rerank(cross, question, id_to_content, dense_ids)
        if with_hybrid:
            # The cross-encoder reranks the FUSED pool (dense ∪ lexical), not the
            # dense pool — chaining the validated lever onto the hybrid candidate set.
            ranked["hybrid_cross_encoder"] = _cross_rerank(
                cross, question, id_to_content, ranked["hybrid_rrf"])

    oracle = any(r in relevant for r in dense_ids)
    return ranked, rerank_ms, oracle


def _metrics_row(ranked, relevant) -> dict:
    row = {}
    for variant, ids in ranked.items():
        row[variant] = {
            "rr": reciprocal_rank(ids, relevant),
            "ndcg10": ndcg_at_k(ids, relevant, 10),
            **{f"hit@{k}": float(any(r in relevant for r in ids[:k])) for k in _KS},
            **{f"recall@{k}": recall_at_k(ids, relevant, k) for k in _KS},
        }
    return row


def _recency(record) -> int:
    n = len(record.history_chunks)
    evi = [c.order for c in record.history_chunks if c.is_evidence]
    return (n - 1 - max(evi)) if evi else -1


# ── Aggregation ───────────────────────────────────────────────────────────────


def _mean(xs) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _variant_summary(per_query, variant, dense_rr) -> dict:
    rr = [q[variant]["rr"] for q in per_query]
    deltas = [a - b for a, b in zip(rr, dense_rr)]
    ci = (0.0, 0.0) if variant == "dense_only" else bootstrap_ci(deltas)
    out = {
        "mrr": _mean(rr),
        "ndcg@10": _mean([q[variant]["ndcg10"] for q in per_query]),
        "delta_mrr_vs_dense": _mean(deltas),
        "ci95_delta_vs_dense": [round(ci[0], 4), round(ci[1], 4)],
    }
    for k in _KS:
        out[f"hit@{k}"] = _mean([q[variant][f"hit@{k}"] for q in per_query])
        out[f"recall@{k}"] = _mean([q[variant][f"recall@{k}"] for q in per_query])
    return out


def _help_hurt(per_query, variant, eps=1e-9) -> dict:
    h = u = z = 0
    for q in per_query:
        d = q[variant]["rr"] - q["dense_only"]["rr"]
        h, u, z = (h + 1, u, z) if d > eps else (h, u + 1, z) if d < -eps else (h, u, z + 1)
    return {"helped": h, "hurt": u, "unchanged": z}


def _recency_table(per_query, variant) -> dict:
    out = {}
    for lo, hi in _RECENCY_BUCKETS:
        sub = [q for q in per_query if lo <= q["recency"] <= hi]
        if not sub:
            continue
        label = f"{lo}-{'inf' if hi >= 10**9 else hi}"
        hh = _help_hurt(sub, variant)
        out[label] = {
            "n": len(sub),
            "dense_mrr": _mean([q["dense_only"]["rr"] for q in sub]),
            f"{variant}_mrr": _mean([q[variant]["rr"] for q in sub]),
            **hh,
        }
    return out


def _aggregate(per_query, excluded, collapsed, latencies, meta) -> dict:
    variants = list(per_query[0].keys()) if per_query else []
    variants = [v for v in variants if v not in ("record_id", "qtype", "recency", "oracle")]
    dense_rr = [q["dense_only"]["rr"] for q in per_query]
    summary = {v: _variant_summary(per_query, v, dense_rr) for v in variants}
    oracle = _mean([1.0 if q["oracle"] else 0.0 for q in per_query])
    tripwire = oracle > 0.5 and summary["dense_only"]["hit@10"] < 0.05
    return {
        **meta,
        "n_evaluated": len(per_query),
        "n_excluded_nondegeneracy": len(excluded),
        "excluded": excluded,
        "embeddings_collapsed_total": collapsed,
        "chance_mrr": summary.get("chance", {}).get("mrr"),
        "oracle_at_pool_k": oracle,
        "dense_ranking_inverted_tripwire": tripwire,
        "rerank_latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 3) if latencies else None,
            "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
        },
        "variants": summary,
        "osam_help_hurt_unchanged_vs_dense": _help_hurt(per_query, "dense_plus_osam"),
        "pure_osam_help_hurt_unchanged_vs_dense": _help_hurt(per_query, "pure_osam"),
        "osam_recency_buckets": _recency_table(per_query, "dense_plus_osam"),
        "pure_osam_recency_buckets": _recency_table(per_query, "pure_osam"),
        "per_query": [
            {"record_id": q["record_id"], "qtype": q["qtype"], "recency": q["recency"],
             "oracle": q["oracle"], "dense_rr": round(q["dense_only"]["rr"], 4),
             "osam_rr": round(q["dense_plus_osam"]["rr"], 4),
             "pure_osam_rr": round(q["pure_osam"]["rr"], 4)}
            for q in per_query
        ],
    }


# ── Orchestration ─────────────────────────────────────────────────────────────


def run(subset_n=50, pool_k=_POOL_K, seed=99, with_cross_encoder=True,
        granularity="turn", with_hybrid=False, backend="bm25s") -> dict:
    _embedder_obj, model, model_ref = _embedder()
    cache_dir = _ROOT / "data" / "emb_cache" / model_ref.model_id / granularity
    cache_dir.mkdir(parents=True, exist_ok=True)
    cross = None
    if with_cross_encoder:
        from sentence_transformers import CrossEncoder
        cross = CrossEncoder(_CROSS_MODEL)

    lexical_ranker, lexical_teardown = (None, lambda: None)
    if with_hybrid:
        from validation_mvp.lexical_backends import make_lexical_backend
        lexical_ranker, lexical_teardown = make_lexical_backend(backend)

    records = load_longmemeval_records(subset_n=subset_n, bench_seed=seed,
                                       granularity=granularity)
    per_query, excluded, latencies, collapsed_total = [], [], [], 0

    try:
        for i, rec in enumerate(records):
            qvec, tmat = _embed_record(rec, model, cache_dir)
            store, engine, relevant, collapsed, n_docs, docs = _seed_question(rec, tmat, model_ref)
            collapsed_total += collapsed
            if not (0 < len(relevant) < n_docs):
                excluded.append({"record_id": rec.record_id, "n_relevant": len(relevant),
                                 "n_docs": n_docs})
                continue
            ranked, ms, oracle = _score_question(
                store, engine, qvec, rec.question, relevant, pool_k, seed * 100003 + i,
                cross, docs, with_hybrid, lexical_ranker, rec.record_id)
            latencies.append(ms)
            row = _metrics_row(ranked, relevant)
            row.update(record_id=rec.record_id, qtype=rec.question_type,
                       recency=_recency(rec), oracle=oracle)
            per_query.append(row)
            print(f"  [{i + 1}/{len(records)}] {rec.record_id} {rec.question_type:24s} "
                  f"recency={_recency(rec):4d} dense_rr={row['dense_only']['rr']:.3f} "
                  f"osam_rr={row['dense_plus_osam']['rr']:.3f}")
    finally:
        lexical_teardown()

    meta = {
        "dataset": "xiaowu0162/longmemeval-cleaned",
        "split": "longmemeval_s_cleaned",
        "mechanism": "osam_simplified_hebbian",
        "alpha": _ALPHA, "normalize": _NORMALIZE, "pool_k": pool_k,
        "subset_n": subset_n, "bench_seed": seed,
        "embedding_model": model_ref.model_id,
        "granularity": granularity,
        "chunking": ("turn_level; precise round-level evidence (oracle has_answer join)"
                     if granularity == "turn"
                     else "session_level; gold-session evidence (granularity-isolation control)"),
    }
    if with_hybrid:
        from validation_mvp.lexical_backends import BACKEND_DESCRIPTORS
        meta["lexical_backend"] = backend
        meta["lexical_backend_descriptor"] = BACKEND_DESCRIPTORS[backend]
        meta["fusion"] = "reciprocal_rank_fusion (rrf_k=60); dense ANN ∪ lexical arm"
    return _aggregate(per_query, excluded, collapsed_total, latencies, meta)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--pool-k", type=int, default=_POOL_K)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--granularity", choices=("turn", "session"), default="turn")
    ap.add_argument("--no-cross-encoder", action="store_true")
    ap.add_argument("--hybrid", action="store_true",
                    help="Add lexical + hybrid(RRF) + hybrid+cross-encoder variants.")
    ap.add_argument("--backend", default="bm25s",
                    choices=("bm25s", "postgres", "elasticsearch", "opensearch"),
                    help="Lexical arm engine for --hybrid. Dense + cross-encoder arms unchanged.")
    args = ap.parse_args()

    result = run(args.n, args.pool_k, args.seed, not args.no_cross_encoder,
                 args.granularity, args.hybrid, args.backend)
    suffix = "" if args.granularity == "turn" else "_session"
    if args.hybrid:
        # Distinct filename — never overwrite the canonical OSAM/cross-encoder provenance.
        # bm25s keeps the canonical name; other backends get a backend-tagged file.
        tag = "" if args.backend == "bm25s" else f"_{args.backend}"
        out = _ROOT / f"state/intermediate/longmemeval_s_hybrid_results_n{args.n}{suffix}{tag}.json"
    else:
        out = _ROOT / f"state/intermediate/longmemeval_s_retrieval_results_n{args.n}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n== SUMMARY (n={result['n_evaluated']}, excluded={result['n_excluded_nondegeneracy']}) ==")
    print(f"chance_mrr           {result['chance_mrr']}")
    for v, s in result["variants"].items():
        if v == "chance":
            continue
        print(f"{v:20s} mrr={s['mrr']:.4f} hit@5={s['hit@5']:.3f} "
              f"recall@5={s['recall@5']:.3f} d_vs_dense={s['delta_mrr_vs_dense']:+.4f} "
              f"CI{s['ci95_delta_vs_dense']}")
    print(f"oracle@pool_k={result['oracle_at_pool_k']}  "
          f"tripwire={result['dense_ranking_inverted_tripwire']}  "
          f"collapsed={result['embeddings_collapsed_total']}")
    print(f"OSAM help/hurt/unchanged: {result['osam_help_hurt_unchanged_vs_dense']}")
    print(f"OSAM recency buckets: {json.dumps(result['osam_recency_buckets'])}")
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
