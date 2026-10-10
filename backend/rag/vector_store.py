"""
FILE PURPOSE
------------
A small local vector database: stores one vector + metadata per schema chunk,
persists to disk, and answers "top-k most similar chunks" queries.

HOW IT WORKS
------------
* Vectors are L2-normalised, so INNER PRODUCT == COSINE SIMILARITY.
* FAISS `IndexFlatIP` does exact brute-force inner-product search (perfect for
  a schema of a few hundred tables). If faiss is not installed a numpy
  fallback gives identical results.
* On disk: <dir>/index.faiss (or vectors.npy) and <dir>/meta.json.

INPUT  : vectors (n, dim) + chunk dicts.   OUTPUT: [(score, chunk_dict), ...].
CALLED BY: rag/ingestion.py (add/save), rag/retriever.py (load/search).
EXAMPLE: store.search(query_vec, k=3) -> [(0.71, {"table": "rating", ...}), ...]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class VectorStoreError(RuntimeError):
    pass


class VectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.items: list[dict] = []
        self.info: dict = {}
        self._index = None

    # ---- build ----
    def build(self, vectors: np.ndarray, items: list[dict], info: dict | None = None) -> None:
        if len(vectors) != len(items):
            raise VectorStoreError("vectors and items differ in length")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.items, self.info = items, info or {}
        self._make_index()

    def _make_index(self) -> None:
        if faiss is not None and self.vectors is not None:
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    # ---- query ----
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[float, dict]]:
        if self.vectors is None or not self.items:
            raise VectorStoreError("Vector store is empty. Run schema ingestion first.")
        k = max(1, min(k, len(self.items)))
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.vectors.shape[1]:
            raise VectorStoreError("Query vector dimension differs from the index "
                                   "(did you change the embedding provider? re-run ingestion).")
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            return [(float(s), self.items[i]) for s, i in zip(scores[0], idx[0]) if i >= 0]
        sims = (self.vectors @ q.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.items[i]) for i in order]

    # ---- persistence ----
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps({"info": self.info, "items": self.items}, indent=2),
                                     encoding="utf-8")
        if faiss is not None and self._index is not None:
            faiss.write_index(self._index, str(d / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        if not (d / "meta.json").exists() or not (d / "vectors.npy").exists():
            raise VectorStoreError(f"No vector store found in {d}. Run: python -m backend.rag.ingestion")
        vs = cls()
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        vs.vectors = np.load(d / "vectors.npy")
        vs.items, vs.info = meta["items"], meta.get("info", {})
        vs._make_index()
        return vs
