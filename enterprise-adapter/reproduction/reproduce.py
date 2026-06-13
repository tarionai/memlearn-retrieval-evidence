"""One-command reproduction of the adapter chain against your own Postgres.

    Neon source (synthetic stand-in) -> ingestion -> retrieval (native FTS)
      -> evaluation -> governed output -> audit record

No Neon access, no embeddings, no model downloads. Native PostgreSQL full-text
search stands in for the retrieval pipeline; the governance gate and audit
writer are the real published modules. This proves the chain mechanics and the
deterministic gate on a reviewer's own database; the full dense / cross-encoder
arm matrix is the frozen Neon evidence under ../evidence/.

Usage:
    docker run --rm -d --name repro-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
    python enterprise-adapter/reproduction/reproduce.py
    docker rm -f repro-pg

Override the connection with REPRO_PG_CONN if your Postgres is elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # make enterprise_adapter importable for direct script run

from enterprise_adapter import append_audit_record, evaluate_governance  # noqa: E402

from repro_metrics import aggregate, query_metrics  # noqa: E402

SCHEMA = "repro_candidate_search"
DEFAULT_CONN = "postgresql://postgres:postgres@localhost:5432/postgres"
OUT_DIR = HERE / "_output"
OVERREACH_CLAIM = "This synthetic run reflects production retrieval quality"


def _conn():
    return psycopg2.connect(os.environ.get("REPRO_PG_CONN", DEFAULT_CONN), connect_timeout=15)


def _tsquery(query_text: str) -> str:
    terms = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in query_text.split()]
    return " | ".join(t for t in terms if t)


def ingest(conn, corpus: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        cur.execute(f"CREATE SCHEMA {SCHEMA}")
        cur.execute(
            f"CREATE TABLE {SCHEMA}.candidate_profiles ("
            "candidate_id text PRIMARY KEY, profile_text text NOT NULL, "
            "search_vector tsvector, source_hash text NOT NULL, ingested_at timestamptz NOT NULL)"
        )
        now = datetime.now(timezone.utc)
        for row in corpus["candidates"]:
            source_hash = hashlib.sha256(row["profile_text"].encode("utf-8")).hexdigest()
            cur.execute(
                f"INSERT INTO {SCHEMA}.candidate_profiles "
                "(candidate_id, profile_text, search_vector, source_hash, ingested_at) "
                "VALUES (%s, %s, to_tsvector('english', %s), %s, %s)",
                (row["candidate_id"], row["profile_text"], row["profile_text"], source_hash, now),
            )
        cur.execute(
            f"CREATE INDEX candidate_profiles_fts_idx ON {SCHEMA}.candidate_profiles "
            "USING GIN (search_vector)"
        )
    conn.commit()


def retrieve_fts(conn, query_text: str, top_k: int = 20) -> list[str]:
    tsq = _tsquery(query_text)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT candidate_id, ts_rank_cd(search_vector, to_tsquery('english', %s)) AS rank "
            f"FROM {SCHEMA}.candidate_profiles "
            "WHERE search_vector @@ to_tsquery('english', %s) "
            "ORDER BY rank DESC, candidate_id LIMIT %s",
            (tsq, tsq, top_k),
        )
        return [r[0] for r in cur.fetchall()]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_report(conn, corpus: dict) -> dict:
    fts_records, baseline_records = [], []
    all_ids = [c["candidate_id"] for c in corpus["candidates"]]
    rng = random.Random(42)
    for query in corpus["queries"]:
        grades = corpus["qrels"][query["query_id"]]
        fts_records.append(query_metrics(retrieve_fts(conn, query["query_text"]), grades))
        baseline_records.append(query_metrics(rng.sample(all_ids, k=min(20, len(all_ids))), grades))
    digest = hashlib.sha256(json.dumps(corpus, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return {
        "evaluation_status": "complete",
        "qrels_status": "frozen",
        "run_id": digest,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": corpus["dataset_version"],
        "embedding_model_id": "none (lexical FTS only)",
        "corpus_size": len(all_ids),
        "git_commit": _git_commit(),
        "metrics_by_variant": {"fts": aggregate(fts_records), "random_baseline": aggregate(baseline_records)},
        "claim_boundaries": corpus["claim_boundaries"],
    }


def main() -> int:
    corpus = json.loads((HERE / "synthetic_corpus.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)
    try:
        conn = _conn()
    except Exception as exc:
        print(f"Could not connect to Postgres ({type(exc).__name__}). Start one with:")
        print("  docker run --rm -d --name repro-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16")
        return 2

    try:
        print("[1/5] ingest: loading synthetic corpus + building FTS index ...")
        ingest(conn, corpus)
        print("[2/5] retrieve + [3/5] evaluate: native Postgres FTS over frozen qrels ...")
        report = build_report(conn, corpus)
    finally:
        conn.close()

    (OUT_DIR / "synthetic_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fts = report["metrics_by_variant"]["fts"]
    print(f"      fts: MRR={fts['mrr']} NDCG@10={fts['ndcg@10']} | "
          f"random_baseline MRR={report['metrics_by_variant']['random_baseline']['mrr']}")

    print("[4/5] govern: applying the deterministic citability + claim-boundary gate ...")
    promoted = evaluate_governance(report)
    blocked = evaluate_governance(report, requested_claims=[OVERREACH_CLAIM])
    print(f"      PROMOTED: {promoted.status} ({len(promoted.allowed_claims)} allowed claims)")
    print(f"      BLOCKED:  {blocked.status} -> {blocked.failed_condition}")

    print("[5/5] audit: appending both governance runs to the audit log ...")
    log = OUT_DIR / "audit_log.jsonl"
    if log.exists():
        log.unlink()
    append_audit_record(log, promoted, source_load_report_ref="synthetic_v0", event_at=report["run_timestamp"])
    append_audit_record(log, blocked, source_load_report_ref="synthetic_v0", event_at=report["run_timestamp"])

    ok = promoted.is_promoted and not blocked.is_promoted
    print(f"\nChain reproduced. Gate emitted PROMOTED and BLOCKED decisions: {'OK' if ok else 'FAIL'}")
    print(f"Outputs: {OUT_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
