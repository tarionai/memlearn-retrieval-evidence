# Dataset and Split Definition

**Status:** frozen · All numbers below are properties of the frozen benchmark sample (LongMemEval-S, n=200, seed=99).

## Source

| Field | Value |
|---|---|
| Dataset | [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (Hugging Face) |
| Config / split | `default` / `longmemeval_s_cleaned` — the S ("small") variant |
| Records | 500 |
| Local cache | `data/longmemeval_s_cleaned.json` (~277 MB) — **not redistributed in this packet**; download from the public HF dataset |
| Loading note | `load_dataset()` non-streaming fails (pyarrow JSON `block_size` OverflowError on very large records). Use stdlib `json.load` on the downloaded file, or streaming mode. The harness adapter reads the local JSON. |

## Record schema (as verified, not assumed)

Each record: `question_id`, `question_type`, `question`, `question_date`, `answer`,
`answer_session_ids` (evidence labels, session-level), and three parallel haystack lists
(`haystack_session_ids`, `haystack_dates`, `haystack_sessions`). A session is a list of turns;
each turn is a JSON-encoded string `'{"role":..., "content":...}'` that must be `json.loads`-ed.
This cleaned version has no per-turn `has_answer` flag — round-level evidence comes from the
oracle join described below. Verified across all 500 records: `answer_session_ids ⊆
haystack_session_ids` with 0 unresolved.

Typical record: ~50 sessions / ~504 turns; evidence ≈ 1 session ≈ 2% of turns (multi-session and
temporal types have more than one evidence session). Question types in the full 500:
multi-session 133, temporal-reasoning 133, knowledge-update 78, single-session-user 70,
single-session-assistant 56, single-session-preference 30.

## Sampling: 500 → 479 → 200

1. **500** records in the split.
2. **479 evaluable** — abstention questions (no evidence in the haystack; nothing to retrieve) are excluded before sampling.
3. **n=200** sampled **uniformly** over the evaluable set with RNG **seed=99**. Uniform sampling is mandatory because streaming returns records grouped by question type — a prefix sample would be type-biased.

All eight variants run on this same fixed subset, so per-query deltas are paired.

## Chunking: turn-level

One retrieval unit per conversation turn. `chunk_id = f"{session_id}::t{idx}"`; the turn's
`content` is embedded. Rationale: fine-grained recency analysis, faithful to the online-sequence
design of the mechanism under test, and avoids MiniLM truncating whole sessions. Any
truncation effects apply to all variants equally, so the comparison stays fair.

A session-level control run (one document per session) was also executed to confirm the
turn-level reading is not a chunking artifact — see `METRICS.md` § Session-level control.

## Evidence labeling (the rule that matters)

**Final rule: precise round-level evidence.** The evidence set for a question is the specific
turn(s) that contain the answer, recovered by an oracle `has_answer` join against the original
LongMemEval annotations (100% recovery; ≈1.9 evidence turns per record).

This replaced the originally locked rule (evidence = all turns of the answer session), which
saturated dense MRR to ≈1.0 — whole-session gold makes ranking trivially easy and fakes
headroom. The change is documented in `EVAL_CONTRACT.md` § Documented post-lock amendment
and in the frozen findings doc. It tightened the benchmark for every variant equally.

## Deduplication

Turns with identical embeddings are collapsed before pooling (**453 collapsed** across the
n=200 subset); the kept representative counts as evidence if any of its duplicates is evidence.
All turns (including duplicates) are still streamed into the OSAM state — dedup applies to the
candidate pool only. This prevents duplicate memories from inflating ranking scores, one of the
two controls (with the chance floor) whose absence produced a previously retracted local-corpus
"lift" — see `NEGATIVE_FINDINGS.md`.

## Non-degeneracy guard

Per record, assert evidence ⊊ history; any violating record is excluded and logged.
**0 exclusions occurred** in the n=200 run.
