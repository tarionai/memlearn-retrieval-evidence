"""
Exception hierarchy for memlearn.

All memlearn exceptions inherit from MemLearnError so callers can catch
the entire domain with a single except clause when appropriate.

Service-layer exceptions (store connection errors, consolidation failures)
are defined in their respective service modules — not here.
"""


class MemLearnError(Exception):
    """Base class for all memlearn exceptions."""


class EmbeddingDimensionMismatch(MemLearnError):
    """Raised when an embedding vector's dimension != EmbeddingModelRef.dimension."""

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"Expected embedding dimension {expected}, got {got}")


class BudgetExceeded(MemLearnError):
    """Raised when token_count exceeds token_budget in a MemoryContext."""

    def __init__(self, token_count: int, token_budget: int) -> None:
        self.token_count = token_count
        self.token_budget = token_budget
        super().__init__(f"Token count {token_count} exceeds budget {token_budget}")


class StaleEmbeddingModel(MemLearnError):
    """Raised when an EmbeddingModelRef mismatch is detected between store and query."""

    def __init__(self, expected: str, got: str) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"Expected model '{expected}', got '{got}'")


class InvalidLaneId(MemLearnError):
    """Raised when a lane_id value is not a member of the LaneId enum."""

    def __init__(self, lane_id: str) -> None:
        self.lane_id = lane_id
        super().__init__(f"Invalid lane ID: '{lane_id}'")
