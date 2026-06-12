"""PPR arm of the §14 Phase 2a-1 Retrieval Ablation Gate (HippoRAG-style).

Builds a REAL entity graph per question via spaCy extraction (NOT topic-label edges
— that is the synthetic-graph trap the doc's Phase 2a-0 warns against), then scores
turns by Personalized PageRank seeded on query entities.

Stage 1 (this module's `reachability` mode) measures whether the extracted graph can
even connect query seeds to the evidence turn — the extraction-validity precondition.
A near-zero reachability means the arm is non-evaluable under spaCy extraction (Phase
2a-0 fails), not "Phase 3 unjustified". Stage 2 (`full` mode) runs the PPR scoring +
dense/ppr/dense+ppr variants only if reachability warrants it.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import networkx as nx
import spacy

from memlearn.adapters.in_memory.sentence_transformer_embedder import SentenceTransformerEmbedder
from validation_mvp.bench_datasets.longmemeval_adapter import load_longmemeval_records

_WS = re.compile(r"\s+")
_DROP_ENT = {"CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY", "TIME"}
_NLP = None
_MODEL = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _NLP


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformerEmbedder()._get_model()
    return _MODEL


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _extract(doc) -> list[str]:
    """Named entities + noun-chunk heads, normalized and filtered to useful connectors."""
    ents: set[str] = set()
    for ent in doc.ents:
        if ent.label_ not in _DROP_ENT:
            t = _norm(ent.text)
            if len(t) > 1:
                ents.add(t)
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ not in ("NOUN", "PROPN"):
            continue
        toks = [w.text for w in chunk if w.pos_ not in ("DET", "PRON")]
        t = _norm(" ".join(toks))
        if len(t) > 2:
            ents.add(t)
    return sorted(ents)


def record_turn_entities(record, cache_dir: Path) -> list[list[str]]:
    """Per-turn entity lists for a record, cached to disk (extraction is slow)."""
    cache = cache_dir / f"{record.record_id}.json"
    texts = [c.text for c in record.history_chunks]
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("n") == len(texts):
            return data["turns"]
    turns = [_extract(d) for d in _nlp().pipe(texts, batch_size=64)]
    cache.write_text(json.dumps({"n": len(texts), "turns": turns}), encoding="utf-8")
    return turns


def build_graph(turn_entities: list[list[str]]):
    """Entity co-occurrence graph + entity→turn-index map + node→component id."""
    graph = nx.Graph()
    ent_turns: dict[str, set[int]] = defaultdict(set)
    for ti, ents in enumerate(turn_entities):
        uniq = set(ents)
        for e in uniq:
            ent_turns[e].add(ti)
            graph.add_node(e)
        for a, b in itertools.combinations(sorted(uniq), 2):
            w = graph.get_edge_data(a, b, default={}).get("weight", 0) + 1
            graph.add_edge(a, b, weight=w)
    comp_id: dict[str, int] = {}
    for cid, comp in enumerate(nx.connected_components(graph)):
        for n in comp:
            comp_id[n] = cid
    return graph, ent_turns, comp_id


def seed_nodes(query_ents: list[str], nodes: list[str], node_emb, tau: float = 0.6) -> set[str]:
    """Map query entities to graph nodes: exact normalized match, else embedding cosine >= tau."""
    node_set = set(nodes)
    seeds: set[str] = set()
    fuzzy_q = [q for q in query_ents if q not in node_set]
    exact = [q for q in query_ents if q in node_set]
    seeds.update(exact)
    if fuzzy_q and len(nodes):
        qv = _model().encode(fuzzy_q, normalize_embeddings=True, show_progress_bar=False)
        sims = np.asarray(qv) @ node_emb.T               # both L2-normalized → cosine
        for i in range(len(fuzzy_q)):
            j = int(sims[i].argmax())
            if sims[i, j] >= tau:
                seeds.add(nodes[j])
    return seeds


def _evidence_entities(record, turn_entities) -> set[str]:
    orders = {c.order for c in record.history_chunks if c.is_evidence}
    return {e for o in orders for e in turn_entities[o]}


def _node_embeddings(nodes: list[str]):
    if not nodes:
        return np.zeros((0, 384), dtype=np.float32)
    return np.asarray(_model().encode(nodes, normalize_embeddings=True,
                                      show_progress_bar=False, batch_size=256), dtype=np.float32)


def _reachable(comp_id: dict, seeds: set[str], evidence_ents: set[str]) -> bool:
    seed_comps = {comp_id[s] for s in seeds if s in comp_id}
    return any(comp_id.get(e) in seed_comps for e in evidence_ents)


def reachability_smoke(n: int = 20, seed: int = 99, tau: float = 0.6) -> dict:
    """Stage 1: can the extracted graph connect query seeds to the evidence turn?"""
    ent_cache = _ROOT / "data" / "ent_cache"
    ent_cache.mkdir(parents=True, exist_ok=True)
    records = load_longmemeval_records(subset_n=n, bench_seed=seed)
    rows = []
    for i, rec in enumerate(records):
        turn_ents = record_turn_entities(rec, ent_cache)
        graph, _ent_turns, comp_id = build_graph(turn_ents)
        nodes = list(graph.nodes())
        node_emb = _node_embeddings(nodes)
        q_ents = _extract(_nlp()(rec.question))
        seeds = seed_nodes(q_ents, nodes, node_emb, tau)
        evi = _evidence_entities(rec, turn_ents)
        rows.append({
            "qtype": rec.question_type,
            "n_q_ents": len(q_ents), "n_nodes": len(nodes), "n_edges": graph.number_of_edges(),
            "n_components": len(set(comp_id.values())) if comp_id else 0,
            "seed_hit": bool(seeds), "evi_has_ents": bool(evi),
            "direct_overlap": bool(seeds & evi),
            "reachable": _reachable(comp_id, seeds, evi),
        })
        print(f"  [{i+1}/{len(records)}] {rec.record_id} {rec.question_type:22s} "
              f"q_ents={len(q_ents):2d} nodes={len(nodes):4d} seeds={len(seeds):2d} "
              f"evi_ents={len(evi):2d} reachable={rows[-1]['reachable']}")

    def rate(key):
        return round(sum(1 for r in rows if r[key]) / len(rows), 3)
    out = {
        "n": len(rows), "tau": tau,
        "seed_hit_rate": rate("seed_hit"),
        "evidence_has_entities_rate": rate("evi_has_ents"),
        "direct_overlap_rate": rate("direct_overlap"),
        "reachability_rate": rate("reachable"),
        "mean_nodes": round(sum(r["n_nodes"] for r in rows) / len(rows), 1),
        "mean_edges": round(sum(r["n_edges"] for r in rows) / len(rows), 1),
        "mean_components": round(sum(r["n_components"] for r in rows) / len(rows), 1),
        "reachability_by_type": {
            t: round(sum(1 for r in rows if r["qtype"] == t and r["reachable"])
                     / max(1, sum(1 for r in rows if r["qtype"] == t)), 3)
            for t in sorted({r["qtype"] for r in rows})
        },
    }
    return out


import math

from benchmarks.retrieval_ablation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from benchmarks.retrieval_ablation.tournament_base import bootstrap_ci

_KS = (1, 5, 10)


def _turn_embeddings(record, cache_dir: Path):
    """Load cached turn + query embeddings from the OSAM run; recompute if absent."""
    cache = cache_dir / f"{record.record_id}.npz"
    texts = [c.text for c in record.history_chunks]
    if cache.exists():
        data = np.load(cache)
        if data["turns"].shape[0] == len(texts):
            return data["query"].astype(np.float32), data["turns"].astype(np.float32)
    q = np.asarray(_model().encode(record.question, normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float32)
    t = np.asarray(_model().encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                   batch_size=128), dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, query=q, turns=t)
    return q, t


def _dedup_reps(turn_matrix, record):
    """Map turns→unique-embedding representatives. Returns rep order list + relevant rep set."""
    seen: dict[bytes, int] = {}
    reps: list[int] = []
    relevant: set[int] = set()
    evi_orders = {c.order for c in record.history_chunks if c.is_evidence}
    for order in range(turn_matrix.shape[0]):
        key = np.round(turn_matrix[order], 5).tobytes()
        rep = seen.setdefault(key, order)
        if rep == order:
            reps.append(order)
        if order in evi_orders:
            relevant.add(rep)
    return reps, relevant


def _idf(ent_turns: dict, n_turns: int) -> dict:
    return {e: math.log(n_turns / len(ts)) for e, ts in ent_turns.items() if ts}


def _ppr_chunk_scores(graph, ent_turns, seeds, idf, n_turns):
    """HippoRAG-style: PPR seeded on query entities, chunk score = Σ PPR(e)·IDF(e)."""
    scores = np.zeros(n_turns, dtype=np.float64)
    if not seeds or graph.number_of_nodes() == 0:
        return scores
    pers = {s: idf.get(s, 0.0) for s in seeds}
    if sum(pers.values()) <= 0:
        pers = {s: 1.0 for s in seeds}
    pr = nx.pagerank(graph, alpha=0.85, personalization=pers, weight="weight", max_iter=100)
    for ent, turns in ent_turns.items():
        contrib = pr.get(ent, 0.0) * idf.get(ent, 0.0)
        if contrib:
            for ti in turns:
                scores[ti] += contrib
    return scores


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def _rank_ids(reps, score_by_order) -> list[str]:
    return [str(o) for o in sorted(reps, key=lambda o: score_by_order[o], reverse=True)]


def _score_record(record, ent_cache, emb_cache, tau, shuffle_seed) -> dict | None:
    turn_ents = record_turn_entities(record, ent_cache)
    qvec, tmat = _turn_embeddings(record, emb_cache)
    reps, relevant = _dedup_reps(tmat, record)
    if not (0 < len(relevant) < len(reps)):
        return None
    rel_ids = {str(o) for o in relevant}
    graph, ent_turns, comp_id = build_graph(turn_ents)
    idf = _idf(ent_turns, len(turn_ents))
    nodes = list(graph.nodes())
    seeds = seed_nodes(_extract(_nlp()(record.question)), nodes, _node_embeddings(nodes), tau)
    dense = tmat @ qvec
    ppr = _ppr_chunk_scores(graph, ent_turns, seeds, idf, len(turn_ents))
    fused = {o: 0.5 * _minmax(dense)[o] + 0.5 * _minmax(ppr)[o] for o in reps}
    ranked = {
        "chance": [str(o) for o in np.random.default_rng(shuffle_seed).permutation(reps)],
        "dense_only": _rank_ids(reps, {o: dense[o] for o in reps}),
        "ppr_only": _rank_ids(reps, {o: ppr[o] for o in reps}),
        "dense_plus_ppr": _rank_ids(reps, fused),
    }
    row = {v: {"rr": reciprocal_rank(ids, rel_ids), "ndcg10": ndcg_at_k(ids, rel_ids, 10),
               **{f"hit@{k}": float(any(r in rel_ids for r in ids[:k])) for k in _KS},
               **{f"recall@{k}": recall_at_k(ids, rel_ids, k) for k in _KS}}
           for v, ids in ranked.items()}
    row.update(qtype=record.question_type, seed_hit=bool(seeds),
               reachable=_reachable(comp_id, seeds, _evidence_entities(record, turn_ents)))
    return row


def _summ(per_query, variant, dense_rr) -> dict:
    rr = [q[variant]["rr"] for q in per_query]
    deltas = [a - b for a, b in zip(rr, dense_rr)]
    ci = (0.0, 0.0) if variant == "dense_only" else bootstrap_ci(deltas)
    out = {"mrr": round(sum(rr) / len(rr), 4),
           "ndcg@10": round(sum(q[variant]["ndcg10"] for q in per_query) / len(per_query), 4),
           "delta_mrr_vs_dense": round(sum(deltas) / len(deltas), 4),
           "ci95_delta_vs_dense": [round(ci[0], 4), round(ci[1], 4)]}
    for k in _KS:
        out[f"hit@{k}"] = round(sum(q[variant][f"hit@{k}"] for q in per_query) / len(per_query), 4)
        out[f"recall@{k}"] = round(sum(q[variant][f"recall@{k}"] for q in per_query) / len(per_query), 4)
    return out


def _per_type_delta(per_query, variant) -> dict:
    out = {}
    for t in sorted({q["qtype"] for q in per_query}):
        sub = [q for q in per_query if q["qtype"] == t]
        d = [q[variant]["rr"] - q["dense_only"]["rr"] for q in sub]
        ci = bootstrap_ci(d) if len(d) > 1 else (0.0, 0.0)
        out[t] = {"n": len(sub), "delta_mrr": round(sum(d) / len(d), 4),
                  "ci95": [round(ci[0], 4), round(ci[1], 4)]}
    return out


def full_run(n=200, seed=99, tau=0.6) -> dict:
    ent_cache = _ROOT / "data" / "ent_cache"
    emb_cache = _ROOT / "data" / "emb_cache" / "all-MiniLM-L6-v2" / "turn"
    ent_cache.mkdir(parents=True, exist_ok=True)
    records = load_longmemeval_records(subset_n=n, bench_seed=seed)
    per_query, excluded = [], 0
    for i, rec in enumerate(records):
        row = _score_record(rec, ent_cache, emb_cache, tau, seed * 100003 + i)
        if row is None:
            excluded += 1
            continue
        per_query.append(row)
        print(f"  [{i+1}/{len(records)}] {rec.record_id} {rec.question_type:22s} "
              f"dense_rr={row['dense_only']['rr']:.3f} ppr_rr={row['ppr_only']['rr']:.3f} "
              f"d+ppr_rr={row['dense_plus_ppr']['rr']:.3f}")
    dense_rr = [q["dense_only"]["rr"] for q in per_query]
    variants = ["chance", "dense_only", "ppr_only", "dense_plus_ppr"]
    return {
        "dataset": "xiaowu0162/longmemeval-cleaned", "split": "longmemeval_s_cleaned",
        "arm": "PPR (HippoRAG-style, spaCy NER+noun-chunk graph, IDF-weighted)",
        "n_evaluated": len(per_query), "n_excluded": excluded, "subset_n": n, "bench_seed": seed,
        "tau_seed_match": tau,
        "seed_hit_rate": round(sum(1 for q in per_query if q["seed_hit"]) / len(per_query), 3),
        "reachability_rate": round(sum(1 for q in per_query if q["reachable"]) / len(per_query), 3),
        "variants": {v: _summ(per_query, v, dense_rr) for v in variants},
        "ppr_per_type_delta": _per_type_delta(per_query, "ppr_only"),
        "dense_plus_ppr_per_type_delta": _per_type_delta(per_query, "dense_plus_ppr"),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("reachability", "full"), default="reachability")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--tau", type=float, default=0.6)
    args = ap.parse_args()
    if args.mode == "reachability":
        result = reachability_smoke(args.n, args.seed, args.tau)
        out = _ROOT / f"state/intermediate/ppr_reachability_smoke_n{args.n}.json"
    else:
        result = full_run(args.n, args.seed, args.tau)
        out = _ROOT / f"state/intermediate/ppr_arm_results_n{args.n}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n== {args.mode.upper()} (n={result.get('n_evaluated', result.get('n'))}) ==")
    if args.mode == "full":
        print(f"seed_hit={result['seed_hit_rate']} reachability={result['reachability_rate']} "
              f"excluded={result['n_excluded']}")
        for v, s in result["variants"].items():
            if v == "chance":
                print(f"  chance   mrr={s['mrr']:.4f}")
                continue
            print(f"  {v:16s} mrr={s['mrr']:.4f} hit@5={s['hit@5']:.3f} recall@10={s['recall@10']:.3f} "
                  f"d_vs_dense={s['delta_mrr_vs_dense']:+.4f} CI{s['ci95_delta_vs_dense']}")
        print(f"  d+ppr per-type delta: {json.dumps(result['dense_plus_ppr_per_type_delta'])}")
    else:
        for k, v in result.items():
            print(f"  {k}: {v}")
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
