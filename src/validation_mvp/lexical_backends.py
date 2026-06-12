"""Pluggable lexical-retrieval backends for the LongMemEval hybrid harness (WP-110).

Each backend ranks a question's *own* deduped doc set (the ``docs`` list that
``run_longmemeval_retrieval._seed_question`` returns) and returns up to ``pool_k``
uids in lexical-score order. The dense and cross-encoder arms are unchanged and
shared across backends; only the lexical retriever swaps. This is the concrete
proof that the hybrid/RRF/rerank pipeline is backend-agnostic.

Isolation semantics differ by engine and are NOT interchangeable:

  * ``bm25s`` / Elasticsearch / OpenSearch use corpus-wide IDF (BM25). The IDF
    MUST be computed over a single question's docs only. Pooling all questions
    into one shared index silently corrupts every score. bm25s indexes per call;
    ES/OpenSearch get a FRESH index per question (created, refreshed, queried,
    deleted) so IDF is isolated by construction.
  * Postgres full-text search (``ts_rank_cd``) has NO corpus-wide IDF — it scores
    each document from its own term frequencies + proximity. A shared table with a
    per-question ``question_id`` filter is therefore safe (no IDF pooling) and is
    reported honestly as "Postgres FTS (ts_rank_cd, english config)", NOT as BM25.

Backend signature: ``rank(query: str, docs: list[tuple[uid, content]], pool_k) -> list[uid]``.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

LexicalRanker = Callable[[str, list[tuple[str, str]], int], list[str]]

# Human-readable lexical-arm descriptor per backend (surfaced in result meta and
# the dashboard so the honest cross-backend differences are explained, not hidden).
BACKEND_DESCRIPTORS: dict[str, str] = {
    "bm25s": "bm25s BM25 (rank-bm25 family, no stopwords, default k1/b)",
    "postgres": "Postgres FTS: to_tsvector('english') + OR-of-lexemes tsquery + ts_rank_cd (no corpus IDF)",
    "elasticsearch": "Elasticsearch 8 Lucene BM25 (standard analyzer, k1=1.2 b=0.75)",
    "opensearch": "OpenSearch 2 Lucene BM25 (standard analyzer, k1=1.2 b=0.75)",
}


# ── bm25s (existing, canonical path — kept bit-identical for reproduction) ──────


def bm25s_ranking(query: str, docs: list[tuple[str, str]], pool_k: int) -> list[str]:
    """Lexical-only ranking: BM25 over the full per-question deduped doc set,
    returning up to pool_k uids in lexical score order. Indexing the full corpus
    is what lets the lexical arm surface documents the dense ANN missed."""
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


# ── Postgres full-text search (shared table + question_id filter is IDF-safe) ──


class PostgresFtsBackend:
    """Postgres FTS lexical arm. ts_rank_cd has no corpus-wide IDF, so a single
    shared table partitioned by question_id is isolation-safe. One connection for
    the whole run; the table is truncated per question to bound size."""

    _DDL = """
        CREATE TABLE IF NOT EXISTS lexical_fts_docs (
            question_id TEXT NOT NULL,
            uid         TEXT NOT NULL,
            content     TEXT NOT NULL,
            tsv         tsvector
                GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            PRIMARY KEY (question_id, uid)
        );
    """
    _INDEX = ("CREATE INDEX IF NOT EXISTS idx_lexical_fts_tsv "
              "ON lexical_fts_docs USING GIN (tsv);")

    def __init__(self, conn_str: str) -> None:
        import psycopg2

        self._conn = psycopg2.connect(conn_str)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(self._DDL)
            cur.execute(self._INDEX)
        self._conn.commit()

    def _or_tsquery(self, cur, query: str):
        """Build an OR-of-lexemes tsquery (term1 | term2 | …) from the question.

        websearch_to_tsquery / plainto_tsquery produce CONJUNCTIVE (AND) queries:
        for a long natural-language question no single memory turn contains every
        stemmed term, so ``@@`` returns empty and lexical MRR collapses BELOW chance
        (measured 0.071 < 0.130 chance — a broken arm, not an honest difference).
        OR-of-lexemes ranked by ts_rank_cd is the standard FTS relevance pattern and
        matches the any-term-contributes scoring of bm25s / Lucene BM25. Returns the
        compiled tsquery, or None when the question normalizes to zero lexemes
        (e.g. all stopwords) — an honest empty result, not a crash."""
        cur.execute(
            "SELECT array_to_string("
            "  ARRAY(SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %s))),"
            "  ' | ')",
            (query,),
        )
        or_text = cur.fetchone()[0]
        if not or_text:
            return None
        cur.execute("SELECT to_tsquery('english', %s)", (or_text,))
        return cur.fetchone()[0]

    def rank(self, query: str, docs: list[tuple[str, str]], pool_k: int,
             question_id: str = "") -> list[str]:
        if not docs:
            return []
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM lexical_fts_docs WHERE question_id = %s",
                        (question_id,))
            cur.executemany(
                "INSERT INTO lexical_fts_docs (question_id, uid, content) "
                "VALUES (%s, %s, %s)",
                [(question_id, uid, content) for uid, content in docs],
            )
            tsquery = self._or_tsquery(cur, query)
            if tsquery is None:
                self._conn.commit()
                return []
            # ts_rank_cd = cover-density ranking (term frequency + proximity). No
            # corpus-wide IDF — unlike Lucene/bm25s BM25 — so this arm honestly may
            # trail on rare-verbatim-term questions where IDF does the work.
            cur.execute(
                """
                SELECT uid
                FROM lexical_fts_docs
                WHERE question_id = %s
                  AND tsv @@ %s::tsquery
                ORDER BY ts_rank_cd(tsv, %s::tsquery) DESC
                LIMIT %s
                """,
                (question_id, tsquery, tsquery, pool_k),
            )
            ranked = [str(row[0]) for row in cur.fetchall()]
        self._conn.commit()
        return ranked

    def drop(self) -> None:
        """Remove the benchmark table — never persist scratch state in the DB."""
        with self._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS lexical_fts_docs;")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ── Elasticsearch / OpenSearch (fresh index per question = IDF isolation) ──────


def _bulk_index_actions(index: str, docs: list[tuple[str, str]]):
    for uid, content in docs:
        yield {"index": {"_index": index, "_id": uid}}
        yield {"content": content}


class _LuceneBm25Backend:
    """Shared logic for ES and OpenSearch (both Lucene BM25). A fresh single-shard
    index per question keeps BM25 IDF scoped to that question's corpus; the index
    is refreshed at bulk time (newly-indexed docs are not searchable until refresh)
    and deleted afterward. Index settings pin 1 shard / 0 replicas so IDF is exact,
    not sharded-approximate. The ES8 and opensearch-py clients differ in call shape
    (``operations=`` vs ``body=``); ``_engine`` selects the right one."""

    _SETTINGS = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0,
                     "index": {"similarity": {"default": {"type": "BM25"}}}},
        "mappings": {"properties": {"content": {"type": "text",
                                                "analyzer": "standard"}}},
    }

    def __init__(self, client, prefix: str, engine: str) -> None:
        self._client = client
        self._prefix = prefix
        self._engine = engine  # "elasticsearch" | "opensearch"

    def _bulk(self, index: str, docs: list[tuple[str, str]]) -> None:
        actions = list(_bulk_index_actions(index, docs))
        if self._engine == "elasticsearch":
            self._client.bulk(operations=actions, refresh=True)
        else:
            self._client.bulk(body=actions, refresh=True)

    def _search(self, index: str, query: str, size: int) -> list[str]:
        body = {"query": {"match": {"content": query}}, "size": size, "_source": False}
        resp = self._client.search(index=index, body=body)
        return [str(h["_id"]) for h in resp["hits"]["hits"]]

    def rank(self, query: str, docs: list[tuple[str, str]], pool_k: int,
             question_id: str = "") -> list[str]:
        if not docs:
            return []
        index = f"{self._prefix}_{question_id}".lower()
        if self._client.indices.exists(index=index):
            self._client.indices.delete(index=index)
        self._client.indices.create(index=index, body=self._SETTINGS)
        try:
            self._bulk(index, docs)
            return self._search(index, query, min(pool_k, len(docs)))
        finally:
            if self._client.indices.exists(index=index):
                self._client.indices.delete(index=index)


def make_elasticsearch_backend(host: str = "http://localhost:9201") -> "_LuceneBm25Backend":
    from elasticsearch import Elasticsearch

    client = Elasticsearch(host, request_timeout=60)
    if not client.ping():
        raise RuntimeError(f"Elasticsearch not reachable at {host}")
    return _LuceneBm25Backend(client, prefix="lme", engine="elasticsearch")


def make_opensearch_backend(host: str = "http://localhost:9200") -> "_LuceneBm25Backend":
    from opensearchpy import OpenSearch

    client = OpenSearch(hosts=[host], use_ssl=False, verify_certs=False,
                        timeout=60)
    if not client.ping():
        raise RuntimeError(f"OpenSearch not reachable at {host}")
    return _LuceneBm25Backend(client, prefix="lme", engine="opensearch")


# ── Factory ────────────────────────────────────────────────────────────────────


def make_lexical_backend(name: str, pg_conn: Optional[str] = None):
    """Return (ranker, teardown). ``ranker(query, docs, pool_k, question_id)``;
    ``teardown()`` releases engine resources (drop table / close connection) or is
    a no-op. The harness passes question_id so external engines isolate per question.

    bm25s needs no question_id (it indexes per call); the returned ranker accepts
    and ignores it for a uniform call site."""
    if name == "bm25s":
        def ranker(query, docs, pool_k, question_id=""):
            return bm25s_ranking(query, docs, pool_k)
        return ranker, (lambda: None)

    if name == "postgres":
        conn = pg_conn or os.environ.get("MEMLEARN_PG_CONN")
        if not conn:
            raise RuntimeError("postgres backend requires MEMLEARN_PG_CONN in the env")
        backend = PostgresFtsBackend(conn)

        def teardown():
            backend.drop()
            backend.close()

        return backend.rank, teardown

    if name == "elasticsearch":
        backend = make_elasticsearch_backend()
        return backend.rank, (lambda: None)

    if name == "opensearch":
        backend = make_opensearch_backend()
        return backend.rank, (lambda: None)

    raise ValueError(f"unknown lexical backend: {name!r}")
