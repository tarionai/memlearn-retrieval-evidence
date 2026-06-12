"""
P-layer type definitions for memlearn.

All types here are transcribed from or derived from:
  docs/architecture/EXTERNAL_MEMORY_LEARNING_KERNEL_CANONICAL.md §5 (core primitives),
  §6.1 / §7.1 (report types), §8 (lane taxonomy), §11 (forget policy taxonomy).

No Protocols — those are C-layer (WP-03+).
No runtime logic — pure type definitions and inline invariant validators.
Only stdlib + numpy imports permitted at this layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID

import numpy as np


# ── Enums ────────────────────────────────────────────────────────────────────


class LaneId(str, Enum):
    """Explicit memory lanes. Never auto-classified — caller-assigned only."""

    FACTS = "facts"
    PREFERENCES = "preferences"
    TASK_PROGRESS = "task_progress"
    EVENTS = "events"
    GENERAL = "general"


class MemorySource(str, Enum):
    """Origin of a MemoryRecord. Used for provenance tracking."""

    AGENT_OUTPUT = "agent_output"
    USER_INPUT = "user_input"
    TOOL_RESULT = "tool_result"


class ForgetPolicyKind(str, Enum):
    """Discriminator for forget policy types. Source: canonical §11."""

    AGE_DRIFT = "age_drift"
    IMPORTANCE_DECAY = "importance_decay"
    TASK_SCOPE_EXPIRY = "task_scope_expiry"
    EXPLICIT_KEY = "explicit_key"
    INTERFERENCE_PRUNE = "interference_prune"


# ── Core Primitives (verbatim from canonical §5) ──────────────────────────────


@dataclass(frozen=True)
class EmbeddingModelRef:
    """
    Version contract for stored vectors. Every vector in the system is
    tagged with the model that produced it. Re-indexing on model change
    is an explicit, logged operation — never a background assumption.
    Principle 7: embedding version is a first-class contract.
    """

    model_id: str    # e.g. "BAAI/bge-m3"
    version: str     # e.g. "1.5.0"
    dimension: int   # e.g. 1024

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"dimension must be > 0, got {self.dimension}")


@dataclass(frozen=True)
class MemoryRecord:
    """
    Atomic unit of episodic memory.
    Source: δ-mem SSW segment model + Episodic Memory Position Paper schema.

    Temporal validity contract (v2.3):
    valid_from defaults to timestamp (the moment of observation).
    valid_until = None means the record is currently valid.
    valid_until is set explicitly by ConflictFlagger.flag_supersession() when a
    newer record supersedes this one — never set by ingestion logic.
    """

    id: UUID
    content: str                          # raw text of the observation/event
    embedding: np.ndarray                 # dense vector
    embedding_model: EmbeddingModelRef    # which model produced this vector
    timestamp: datetime
    valid_from: datetime                  # defaults to timestamp at construction
    valid_until: datetime | None          # None = currently valid
    causal_parent_id: UUID | None         # linked episode chain
    importance: float                     # 0–1, scored at admission
    surprise: float                       # 0–1, KL-divergence from recent context mean
    source: str                           # "agent_output" | "user_input" | "tool_result"
    lane_id: str                          # explicit, caller-assigned lane; default "general"

    def __post_init__(self) -> None:
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(
                f"importance must be in [0.0, 1.0], got {self.importance}"
            )
        if not (0.0 <= self.surprise <= 1.0):
            raise ValueError(
                f"surprise must be in [0.0, 1.0], got {self.surprise}"
            )

@dataclass(frozen=True)
class ExtractionResult:
    """Output contract for entity/relation extraction from raw text."""

    entities: list[dict]    # [{id, label, type, properties}]
    relations: list[dict]   # [{source_id, target_id, relation, weight}]


@dataclass(frozen=True)
class ConflictFlag:
    """Typed representation of an active conflict between two memory nodes.

    conflict_type distinguishes unresolved factual contradictions ('CONTRADICTION')
    from resolved supersessions ('SUPERSESSION').
    severity applies to CONTRADICTION records only; empty string for SUPERSESSION.
    node_a and node_b carry the first 8 hex chars of each node's UUID for display.
    """

    conflict_type: str   # 'CONTRADICTION' | 'SUPERSESSION'
    severity: str        # 'HIGH' | 'LOW' | '' (empty for supersession)
    node_a: str          # UUID short form (first 8 hex chars)
    node_b: str          # UUID short form (first 8 hex chars)


@dataclass(frozen=True)
class AdmissionConfig:
    """
    External configuration for AdmissionGate behavior.
    Surfaces gate tuning to host applications for multi-tenant and
    task-scoped deployments where surprise thresholds differ by context.

    All fields have validated defaults. Host applications override only what they need.
    """

    surprise_threshold_percentile: float = 0.90    # rolling percentile for gate
    importance_floor: float = 0.05                 # below this, always discard
    rolling_window_size: int = 100                 # number of recent records for stats
    gate_enabled: bool = True                      # False = admit everything (testing)


@dataclass(frozen=True)
class DeltaClampPolicy:
    """
    Bounds enforcement for all learning-path weight mutations.
    Constructor-injected on LearningKernel. Enforced inside
    SemanticStore.apply_weight_deltas() — no weight write bypasses this guard.

    Principle 9: Weight mutation is bounded.
    """

    max_abs_delta_per_node: float = 0.1     # |delta| per node per learn() call
    max_total_delta_per_pass: float = 0.5   # sum(|deltas|) across all nodes per learn()
    min_weight: float = 0.0                 # absolute floor on node weight
    max_weight: float = 1.0                 # absolute ceiling on node weight

    def __post_init__(self) -> None:
        if self.max_abs_delta_per_node <= 0:
            raise ValueError(
                f"max_abs_delta_per_node must be > 0, got {self.max_abs_delta_per_node}"
            )
        if self.max_total_delta_per_pass <= 0:
            raise ValueError(
                f"max_total_delta_per_pass must be > 0, got {self.max_total_delta_per_pass}"
            )
        if self.min_weight > self.max_weight:
            raise ValueError(
                f"min_weight ({self.min_weight}) must be <= max_weight ({self.max_weight})"
            )


@dataclass(frozen=True)
class MemoryContext:
    """
    Assembled context block for LLM prompt injection.
    Sources: EpisodicStore (dense + OSAM-reranked) + SemanticStore (PPR).

    token_count is pre-computed via TokenizerAdapter before return.
    token_count <= token_budget is a hard invariant on return.

    context_id is assigned at assembly time by retrieve(). Used by
    RetrievalHitTracker in integrate() to correlate context with extraction.
    """

    context_id: UUID
    recent_episodes: list[MemoryRecord]     # top-N tail of EpisodicStore (valid only)
    retrieved_episodes: list[MemoryRecord]  # dense results, OSAM-reranked (valid only)
    semantic_facts: list[str]              # distilled nodes from SemanticStore (PPR)
    conflict_flags: list[ConflictFlag]    # active CONFLICTS_WITH contradiction edges
    token_count: int                       # pre-computed actual count
    token_budget: int                      # declared ceiling from caller

    def __post_init__(self) -> None:
        if self.token_count > self.token_budget:
            raise ValueError(
                f"token_count ({self.token_count}) exceeds token_budget ({self.token_budget})"
            )


# ── Report Types (derived from canonical §6.1 / §7.1 return signatures) ──────


@dataclass(frozen=True)
class ConsolidationReport:
    """Return type for MemoryKernel.consolidate(). Source: canonical §6.1."""

    promoted: int
    pruned: int
    conflicts_flagged: int
    conflicts_staled: int
    elapsed_ms: float


@dataclass(frozen=True)
class ForgetReport:
    """Return type for MemoryKernel.forget(). Source: canonical §6.1 + §11."""

    policy_kind: ForgetPolicyKind
    records_removed: int
    lanes_reset: list[str]
    elapsed_ms: float


@dataclass(frozen=True)
class LearningReport:
    """Return type for LearningKernel.learn(). Source: canonical §7.1."""

    nodes_updated: int
    nodes_rejected: int
    nodes_clamped: int
    interference_score: float
    total_delta_mass: float


@dataclass(frozen=True)
class RetrievalEffectivenessReport:
    """Return type for MemoryKernel.retrieval_effectiveness(). Source: canonical §6.1."""

    total_contexts: int
    total_records_served: int
    total_hits: int
    hit_rate: float
    per_lane_hit_rates: dict[str, float]
    conflict_accumulation: int
    stale_conflict_count: int


@dataclass(frozen=True)
class SnapshotHandle:
    """Return type for LearningKernel.snapshot(). Source: canonical §7.1."""

    tag: str
    timestamp: datetime
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class ForgettingAuditReport:
    """Return type for LearningKernel.audit_forgetting(). Source: canonical §7.1."""

    per_task_modification_counts: dict[str, int]
    interference_budget_usage: dict[str, float]
    retention_ratio: float
    acquisition_count: int
    rollback_count: int


# ── Forget Policy Types (independent frozen dataclasses; source: canonical §11) ──
# Each policy type carries its own discriminator via an init=False kind field.
# No shared base class — keeps each policy's contract small and independently implementable.


@dataclass(frozen=True)
class AgeDriftPolicy:
    """Forget records that have drifted too far from recent context mean."""

    kind: ForgetPolicyKind = field(default=ForgetPolicyKind.AGE_DRIFT, init=False)
    decay_lambda: float
    importance_floor: float


@dataclass(frozen=True)
class ImportanceDecayPolicy:
    """Forget records whose importance falls below threshold after N unretrieved cycles."""

    kind: ForgetPolicyKind = field(default=ForgetPolicyKind.IMPORTANCE_DECAY, init=False)
    unretrieved_cycles_threshold: int


@dataclass(frozen=True)
class TaskScopeExpiryPolicy:
    """Forget records scoped to a task when that task ends."""

    kind: ForgetPolicyKind = field(default=ForgetPolicyKind.TASK_SCOPE_EXPIRY, init=False)
    task_tag: str


@dataclass(frozen=True)
class ExplicitKeyPolicy:
    """Forget specific records by their IDs."""

    kind: ForgetPolicyKind = field(default=ForgetPolicyKind.EXPLICIT_KEY, init=False)
    target_ids: list[UUID]


@dataclass(frozen=True)
class InterferencePrunePolicy:
    """Prune records that interfere with a task below a confidence floor."""

    kind: ForgetPolicyKind = field(default=ForgetPolicyKind.INTERFERENCE_PRUNE, init=False)
    task_tag: str
    confidence_floor: float


# Union type alias for all forget policy kinds.
ForgetPolicy = (
    AgeDriftPolicy
    | ImportanceDecayPolicy
    | TaskScopeExpiryPolicy
    | ExplicitKeyPolicy
    | InterferencePrunePolicy
)
