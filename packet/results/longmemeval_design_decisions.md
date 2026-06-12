# LongMemEval Retrieval-Stage Slice — Locked Design (Phase P1)

**Date:** 2026-06-04 · WP-110 retrieval-stage slice · mechanism `osam_simplified_hebbian` (α=0.7 frozen)

## Dataset (P1.1 resolved)
- WP-110 path `chat-longmemeval/long-mem-eval-v2` is **dead** (DatasetNotFoundError).
- Canonical replacement: **`xiaowu0162/longmemeval-cleaned`**, config `default`, split **`longmemeval_s_cleaned`** (the S/"small" variant — matches output name). Also has `longmemeval_oracle` (fallback) + `longmemeval_m_cleaned`.
- Cached locally: `data/longmemeval_s_cleaned.json` (277 MB, gitignored). **500 records.**
- Cannot `load_dataset()` non-streaming (pyarrow JSON block_size OverflowError on huge records). Use stdlib `json.load` on the downloaded file, OR streaming. Adapter reads the local JSON.

## Schema (verified, no assumptions)
- `question_id:str`, `question_type:str`, `question:str`, `question_date:str`, `answer:str`
- `answer_session_ids: list[str]` — **EVIDENCE LABELS, session-level**
- `haystack_session_ids: list[str]`, `haystack_dates: list[str]`, `haystack_sessions: list[session]` — three parallel lists
- session = `list[turn]`; **turn = JSON-encoded string** `'{"role":...,"content":...}'` → must `json.loads`. No per-turn `has_answer` flag in this cleaned version.
- **`answer_session_ids ⊆ haystack_session_ids`: 0 unresolved across all 500.** ✓

## Distribution
- ~50 sessions / ~504 turns per record. Evidence ≈ 1 session ≈ 2% of turns (multi-session/temporal have >1 evidence session).
- Evidence chronological position spreads 0.09→1.0 (oldest→newest) → enables λ-recency analysis.
- Types: multi-session 133, temporal-reasoning 133, knowledge-update 78, single-session-user 70, single-session-assistant 56, single-session-preference 30.
- **Streaming returns records type-grouped → subset MUST sample uniformly across the full set** (RNG seed=99).

## Chunking decision: TURN-LEVEL
- One chunk per turn; chunk_id = `f"{session_id}::t{idx}"`; embed turn `content`.
- **Evidence = ALL turns of the answer session(s)** (session-gold → turn expansion; advisor-blessed). Documented loosening.
- Rationale: faithful to OSAM online-sequence design + fine λ-recency resolution + avoids MiniLM truncation of whole sessions. Truncation/looseness affects all variants equally → comparison stays fair.

## Harness invariants
- **Per-question isolation**: fresh `InMemoryVectorStore`+`EpisodicStore`+`AssociativeStateEngine` per record (questions are independent — UNLIKE diagnostic_pool_ceiling's deliberately-shared store). Shared embedder only.
- **Stream chronologically** (sort sessions by `haystack_dates`, lexical OK for YYYY/MM/DD HH:MM) into OSAM — all ~504 turns (intentionally violates the ≤1/(1−λ)≈10 warmup guidance; the forgetting IS the measurement).
- **Non-degeneracy guard**: assert evidence ⊊ history per record; exclude+log any with evidence ≡ history (none expected).
- **DEDUP identical embeddings** (mandatory — todo.md/auto-memory): seed vector store with unique-embedding turns only (collapse dups; rep is evidence if any dup is); still stream ALL turns into OSAM. Log #collapsed.
- **CHANCE baseline** (mandatory): per-query deterministic shuffle of the deduped pool → random top-k MRR at pool relevant density. The floor any lift must beat.
- **Dense sanity tripwire**: if oracle@pool_k high but dense hit@10 ≈ 0 → ranking inverted (only then read InMemoryVectorStore.search). Convention confirmed: `score` = L2 distance ascending, `normalize=True` (matches run_wp10i `_ALPHA=0.7,_NORMALIZE=True`).

## Variants (frozen, ungated)
chance · dense_only (pool order) · dense_plus_osam (α=0.7, normalize=True) · pure_osam (α=0.0, normalize=True). pool_k=50.

## Metrics
MRR (full pool) lead; **hit@k (success@k)** for k∈{1,5,10} as the interpretable recall (multi-turn gold deflates `metrics.recall_at_k` fraction — report both); NDCG@10; help/hurt/unchanged vs dense; oracle@pool_k; paired `bootstrap_ci` on (variant−dense) RR; rerank latency P50/P95; λ-recency buckets (turns streamed after closest-to-end evidence turn).

## Reuse (no duplication)
`benchmarks/retrieval_ablation/metrics.py` (reciprocal_rank, recall_at_k, ndcg_at_k) + `tournament_base.bootstrap_ci`. Adapter location `src/validation_mvp/datasets/` (reusable by full WP-110). Do NOT build full WP-110 type tree.
