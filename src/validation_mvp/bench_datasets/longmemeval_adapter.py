"""LongMemEval adapter — turn-level chunks + precise round-level evidence.

Source: HuggingFace ``xiaowu0162/longmemeval-cleaned`` (the WP-110 path
``chat-longmemeval/long-mem-eval-v2`` no longer resolves), cached to ``data/``:
  - ``longmemeval_s_cleaned.json``  — full haystacks (gold + distractor sessions)
  - ``longmemeval_oracle.json``     — gold sessions only, with per-turn ``has_answer``

Each record's haystack is flattened to turn-level chunks in chronological order
(sessions sorted by date). The S split dropped the per-turn ``has_answer`` flag, so
evidence is recovered by joining on ``question_id``: a turn is evidence iff it sits
in a gold (answer) session AND its normalized content matches a ``has_answer`` turn
in the oracle split. This yields ~1.9 precise evidence turns per record (vs ~35 for
whole-session gold, which saturated MRR at 1.0). Abstention questions (no oracle
evidence) are dropped — they test refusal, not retrieval.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = "data/longmemeval_s_cleaned.json"
ORACLE_PATH = "data/longmemeval_oracle.json"
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class Chunk:
    """One conversation turn = one retrievable/streamable memory unit."""

    chunk_id: str          # f"{session_id}::t{turn_idx}" — unique within a record
    text: str              # turn content (the substantive text that gets embedded)
    session_id: str
    role: str
    order: int             # global chronological index within the record (0-based)
    is_evidence: bool      # precise round-level gold (has_answer turn in a gold session)


@dataclass(frozen=True)
class LongMemEvalRecord:
    record_id: str
    question: str
    question_type: str
    answer: str
    history_chunks: tuple[Chunk, ...]      # chronological
    evidence_chunk_ids: frozenset[str]


def _parse_turn(turn: object) -> tuple[str, str, bool]:
    """Return (role, content, has_answer). Turns are dicts (oracle) or JSON strings (S)."""
    if isinstance(turn, dict):
        return (str(turn.get("role", "unknown")), str(turn.get("content", "")),
                bool(turn.get("has_answer")))
    try:
        obj = json.loads(turn)
        return (str(obj.get("role", "unknown")), str(obj.get("content", "")),
                bool(obj.get("has_answer")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return "unknown", str(turn), False


def load_oracle_evidence(path: str = ORACLE_PATH) -> dict[str, frozenset[str]]:
    """{question_id -> normalized contents of has_answer turns} from the oracle split."""
    oracle = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, frozenset[str]] = {}
    for rec in oracle:
        ev = {
            _norm(content)
            for _role, content, has in (_parse_turn(t)
                                        for sess in rec["haystack_sessions"] for t in sess)
            if has
        }
        out[rec["question_id"]] = frozenset(ev)
    return out


def _chronological_sessions(record: dict) -> list[tuple[str, list]]:
    """Zip session ids/dates/turns and sort chronologically by date string."""
    triples = list(zip(
        record["haystack_session_ids"],
        record["haystack_dates"],
        record["haystack_sessions"],
    ))
    triples.sort(key=lambda t: t[1])  # 'YYYY/MM/DD (DoW) HH:MM' → lexical == chronological
    return [(sid, turns) for sid, _date, turns in triples]


def _build_chunks(record: dict, evidence_contents: frozenset[str]):
    gold_sessions = set(record["answer_session_ids"])
    chunks: list[Chunk] = []
    evidence_ids: set[str] = set()
    order = 0
    for session_id, turns in _chronological_sessions(record):
        in_gold = session_id in gold_sessions
        for turn_idx, turn in enumerate(turns):
            _role, content, _has = _parse_turn(turn)
            is_evi = in_gold and _norm(content) in evidence_contents
            chunk_id = f"{session_id}::t{turn_idx}"
            chunks.append(Chunk(chunk_id, content, session_id, _role, order, is_evi))
            if is_evi:
                evidence_ids.add(chunk_id)
            order += 1
    return tuple(chunks), frozenset(evidence_ids)


def _build_session_chunks(record: dict):
    """One chunk per session (concatenated turn contents); evidence = gold session(s).

    For the granularity-isolation control: same harness/pool/n as turn mode, only the
    chunk unit and the (session-level) evidence differ.
    """
    gold_sessions = set(record["answer_session_ids"])
    chunks: list[Chunk] = []
    evidence_ids: set[str] = set()
    for order, (session_id, turns) in enumerate(_chronological_sessions(record)):
        text = " ".join(_parse_turn(t)[1] for t in turns)
        is_evi = session_id in gold_sessions
        chunks.append(Chunk(session_id, text, session_id, "session", order, is_evi))
        if is_evi:
            evidence_ids.add(session_id)
    return tuple(chunks), frozenset(evidence_ids)


def _to_record(raw: dict, evidence_contents: frozenset[str],
               granularity: str = "turn") -> LongMemEvalRecord:
    if granularity == "session":
        chunks, evidence_ids = _build_session_chunks(raw)
    else:
        chunks, evidence_ids = _build_chunks(raw, evidence_contents)
    return LongMemEvalRecord(
        record_id=raw["question_id"],
        question=raw["question"],
        question_type=raw["question_type"],
        answer=raw["answer"],
        history_chunks=chunks,
        evidence_chunk_ids=evidence_ids,
    )


def load_longmemeval_records(
    path: str = DEFAULT_PATH,
    oracle_path: str = ORACLE_PATH,
    subset_n: int | None = None,
    bench_seed: int = 99,
    drop_abstention: bool = True,
    granularity: str = "turn",
) -> list[LongMemEvalRecord]:
    """Load records with precise round-level evidence; optional reproducible subset.

    granularity="turn" (default): one chunk per turn, precise round-level evidence.
    granularity="session": one chunk per session, gold-session evidence (isolation control).

    Abstention questions (no oracle has_answer evidence) are dropped by default — they
    have no retrievable evidence. The drop is keyed on oracle evidence regardless of
    granularity, so both modes evaluate the IDENTICAL record set. Subsetting samples
    uniformly without replacement across the evaluable set (records are type-grouped on disk).
    """
    evidence = load_oracle_evidence(oracle_path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if drop_abstention:
        raw = [r for r in raw if evidence.get(r["question_id"])]
    records = [_to_record(r, evidence.get(r["question_id"], frozenset()), granularity) for r in raw]
    if subset_n is not None and subset_n < len(records):
        rng = np.random.default_rng(bench_seed)
        idx = sorted(int(j) for j in rng.choice(len(records), size=subset_n, replace=False))
        records = [records[i] for i in idx]
    return records
