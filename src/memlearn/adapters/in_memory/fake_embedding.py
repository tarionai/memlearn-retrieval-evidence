"""FakeEmbedding — deterministic EmbedderPort for testing.

Returns a content-seeded, L2-normalised float32 vector. No network,
no model weights, no filesystem reads, no environment-variable access.

Algorithm (Decision D2, WP-07 N-04 — do not deviate):
  seed = int.from_bytes(sha256(text.encode()).digest()[:4], "big")
  rng  = numpy.random.default_rng(seed)
  v    = rng.standard_normal(dimension).astype(float32)
  norm = numpy.linalg.norm(v)
  return zeros(dimension) if norm == 0 else v / norm
"""
from __future__ import annotations

import hashlib

import numpy as np

from memlearn.primitives import EmbeddingModelRef


class FakeEmbedding:
    """Deterministic EmbedderPort stub.

    Satisfies the EmbedderPort protocol (embed(str) -> np.ndarray).
    Identical inputs always produce identical outputs; distinct inputs
    almost certainly differ (SHA-256 collision resistance).
    """

    def __init__(self, embedding_model: EmbeddingModelRef) -> None:
        self.embedding_model = embedding_model
        self.dimension: int = embedding_model.dimension

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dimension, dtype=np.float32)
        seed = int.from_bytes(
            hashlib.sha256(text.encode()).digest()[:4], "big"
        )
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        return v / norm
