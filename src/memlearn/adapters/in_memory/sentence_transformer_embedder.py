"""SentenceTransformerEmbedder — real semantic EmbedderPort for benchmarking.

Uses sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0, 384-dim).
Model is lazy-loaded on first embed() call so importing this module
does not require torch/sentence-transformers to be installed.

Install requirements: pip install memlearn[real-embed]
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from memlearn.primitives import EmbeddingModelRef

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_ID = "all-MiniLM-L6-v2"
DIMENSION = 384


class SentenceTransformerEmbedder:
    """Real semantic EmbedderPort backed by sentence-transformers all-MiniLM-L6-v2.

    Satisfies the EmbedderPort protocol (embed(str) -> np.ndarray).
    Returns L2-normalized float32 vectors of dimension 384.
    Model is loaded from local cache on first embed() call.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModelRef] = None) -> None:
        self.embedding_model = embedding_model or EmbeddingModelRef(
            model_id=MODEL_ID, version="v2", dimension=DIMENSION
        )
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install memlearn[real-embed]"
                ) from exc
            self._model = SentenceTransformer(MODEL_ID)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.embedding_model.dimension, dtype=np.float32)
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vector, dtype=np.float32)
