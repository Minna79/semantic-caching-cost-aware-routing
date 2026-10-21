"""Small in-process vector store (exact cosine search with numpy).

Enough for thousands of vectors and needs no server. A Qdrant-backed store can be added behind
the same interface (add / search / delete_where / save / load) later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Hit:
    score: float
    payload: dict
    index: int


class NumpyVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._payloads: list[dict] = []

    def __len__(self) -> int:
        return len(self._payloads)

    def add(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[1] != self.dim or len(vectors) != len(payloads):
            raise ValueError("vector/payload shape mismatch")
        self._vecs = np.vstack([self._vecs, vectors])
        self._payloads.extend(payloads)

    def search(
        self, vector: np.ndarray, k: int = 5, where: Callable[[dict], bool] | None = None
    ) -> list[Hit]:
        if not self._payloads:
            return []
        scores = self._vecs @ np.asarray(vector, dtype=np.float32).reshape(-1)
        if where is not None:
            mask = np.fromiter((where(p) for p in self._payloads), dtype=bool, count=len(scores))
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(float(scores[i]), self._payloads[i], int(i)) for i in top if np.isfinite(scores[i])]

    def delete_where(self, pred: Callable[[dict], bool]) -> int:
        keep = [i for i, p in enumerate(self._payloads) if not pred(p)]
        removed = len(self._payloads) - len(keep)
        if removed:
            self._vecs = self._vecs[keep]
            self._payloads = [self._payloads[i] for i in keep]
        return removed

    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(prefix.with_suffix(".npy"), self._vecs)
        with open(prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self._payloads, f, ensure_ascii=False)

    @classmethod
    def load(cls, prefix: Path) -> "NumpyVectorStore":
        prefix = Path(prefix)
        vecs = np.load(prefix.with_suffix(".npy"))
        with open(prefix.with_suffix(".json"), encoding="utf-8") as f:
            payloads = json.load(f)
        store = cls(vecs.shape[1])
        store._vecs, store._payloads = vecs.astype(np.float32), payloads
        return store
