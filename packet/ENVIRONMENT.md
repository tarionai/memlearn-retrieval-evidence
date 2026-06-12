# Hardware and Execution Environment

**Status:** frozen · This is the single environment all n=200 numbers were measured on.
Latency figures in `METRICS.md` are scoped to this environment and do not transfer.

## Machine

| Component | Value |
|---|---|
| CPU | Intel Core i7-14700F (20 physical / 28 logical cores) |
| RAM | 96 GB |
| GPU | none used — all inference on CPU |
| OS | Windows 11 Home (build 10.0.26200) |

## Runtime

| Component | Value |
|---|---|
| Python | 3.10.11 (pyenv-win) |
| Key packages | pinned in `requirements.lock.txt` (numpy 2.2.6, torch 2.11.0 CPU, sentence-transformers 5.4.1, transformers 5.5.4, bm25s 0.3.9, networkx 3.4.2, spacy 3.8.14) |

## Models

| Role | Model | Notes |
|---|---|---|
| Dense bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` | 384-d embeddings; L2 distance ascending, normalized |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU inference, deterministic; ~512-token input truncation |
| PPR entity extraction | spaCy `en_core_web_sm` 3.8.0 | NER + noun chunks (not LLM-OpenIE — see `NEGATIVE_FINDINGS.md` scope notes) |
| Lexical | `bm25s` 0.3.9 | in-process BM25, default k1/b, no stopwords |

## Determinism

Fixed seeds throughout: benchmark subset seed=99; per-query deterministic shuffle for the
chance floor; deterministic CPU inference for both transformer models. Embeddings are cached
per record (`.npz`), so re-runs reuse identical vectors.

## What this environment statement licenses

Only this claim: the numbers in `METRICS.md` were measured here, under the pinned package
set. No cross-hardware latency claims, no throughput claims, no cost claims.
