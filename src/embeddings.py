"""Step 3 - Embeddings: turn text into vectors that capture meaning."""
import hashlib
import math
import re

from . import config


class Embedder:
    """backend='local' uses sentence-transformers (recommended).
    backend='hash' is a tiny offline fallback used for fast tests."""

    def __init__(self, backend=None, model_name=None):
        self.backend = backend or config.EMBEDDING_BACKEND
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts):
        if self.backend == "hash":
            return [self._hash_vector(t) for t in texts]
        vectors = self._load().encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @staticmethod
    def _hash_vector(text, dims=256):
        vec = [0.0] * dims
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % dims
            vec[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
