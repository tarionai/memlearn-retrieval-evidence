"""Fixture-driven fake implementation of EntityExtractorPort for deterministic testing."""

from __future__ import annotations

from memlearn.primitives import ExtractionResult


class FakeEntityExtractor:
    """Returns pre-registered ExtractionResult fixtures keyed by exact text match.

    Invariants:
    - No fixture dict → every extract() call returns ExtractionResult(entities=[], relations=[]).
    - Fixture dict present, key matches → returns the registered ExtractionResult exactly.
    - Fixture dict present, key absent → returns ExtractionResult(entities=[], relations=[]).
    - call_count increments on every extract() call regardless of fixture hit/miss.
    """

    _EMPTY = ExtractionResult(entities=[], relations=[])

    def __init__(self, fixtures: dict[str, ExtractionResult] | None = None) -> None:
        self._fixtures: dict[str, ExtractionResult] = fixtures if fixtures is not None else {}
        self.call_count: int = 0

    def extract(self, text: str) -> ExtractionResult:
        self.call_count += 1
        return self._fixtures.get(text, self._EMPTY)
