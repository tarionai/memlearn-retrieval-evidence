"""FakeLLMPort — deterministic LLMPort for testing.

Returns a fixed string on every complete() call. No network, subprocess,
or environment-variable access.
"""
from __future__ import annotations


class FakeLLMPort:
    """Configurable stub that satisfies the LLMPort protocol."""

    def __init__(self, response: str = "stub_response") -> None:
        self.response = response
        self.call_count: int = 0

    def complete(self, prompt: str, *, max_tokens: int, system: str = "") -> str:
        self.call_count += 1
        return self.response
