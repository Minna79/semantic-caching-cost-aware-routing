"""Embedders. All return L2-normalised float32 arrays, so cosine similarity is a dot product.

* HashEmbedder: offline, dependency-free stand-in (hashed character n-grams). It measures
  *lexical* overlap, not meaning. Use it for tests and plumbing only. Similarity thresholds
  tuned with it are NOT valid for the real model.
* SentenceTransformerEmbedder: the real multilingual model used for results.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter

import numpy as np

from .text import normalize


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash{dim}"

    def _features(self, text: str):
        norm = normalize(text)
        words = norm.split()
        for w in words:
            yield "w:" + w
        for w in words:
            padded = f"^{w}$"
            for n in (3, 4):
                for i in range(len(padded) - n + 1):
                    yield padded[i : i + n]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in Counter(self._features(text)).items():
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 16) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + math.log(count))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model = SentenceTransformer(model_name)
        self.name = model_name.replace("/", "_")
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(backend: str, model_name: str):
    if backend == "hash":
        return HashEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown embedding backend: {backend!r}")
