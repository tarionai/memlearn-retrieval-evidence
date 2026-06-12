"""FakeTokenizer — deterministic TokenizerAdapter for tests and offline use.

Counts tokens by whitespace splitting: len(text.split()).
No model, network, or filesystem dependencies.
"""
from memlearn.ports import TokenizerAdapter


class FakeTokenizer:
    """Stateless token counter using whitespace splitting.

    Satisfies TokenizerAdapter Protocol.  Suitable for unit tests and any
    context where reproducibility matters more than byte-pair accuracy.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split())


assert isinstance(FakeTokenizer(), TokenizerAdapter)
