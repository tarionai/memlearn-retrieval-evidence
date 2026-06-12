"""memlearn — external memory learning kernel primitives."""

from memlearn.ports import (
    EmbedderPort,
    EntityExtractorPort,
    GraphStoreAdapter,
    KVStoreAdapter,
    LLMPort,
    MergeStrategy,
    SearchResult,
    TokenizerAdapter,
    VectorStoreAdapter,
)
from memlearn.errors import (
    BudgetExceeded,
    EmbeddingDimensionMismatch,
    InvalidLaneId,
    MemLearnError,
    StaleEmbeddingModel,
)
from memlearn.services.associative_state_engine import AssociativeState
from memlearn.primitives import (
    AdmissionConfig,
    AgeDriftPolicy,
    ConsolidationReport,
    DeltaClampPolicy,
    EmbeddingModelRef,
    ExplicitKeyPolicy,
    ExtractionResult,
    ForgetPolicy,
    ForgetPolicyKind,
    ForgetReport,
    ForgettingAuditReport,
    ImportanceDecayPolicy,
    InterferencePrunePolicy,
    LaneId,
    LearningReport,
    MemoryContext,
    MemoryRecord,
    MemorySource,
    RetrievalEffectivenessReport,
    SnapshotHandle,
    TaskScopeExpiryPolicy,
)
