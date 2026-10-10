"""
FILE PURPOSE
------------
Turns text into vectors (embeddings) behind a swappable interface.

WHAT IS AN EMBEDDING? (beginner version)
----------------------------------------
A list of numbers (a vector) that represents the MEANING of a text. Texts with
similar meaning get vectors pointing in similar directions. Similarity is
measured with COSINE SIMILARITY = (a . b) / (|a| |b|). If vectors are
length-normalised, cosine similarity is just the dot product, which is what
the FAISS inner-product index computes.

PROVIDERS
---------
1. LocalHashEmbedding (default, EMBEDDING_PROVIDER=local): fully offline and
   deterministic. Word + character-trigram features are hashed into a fixed
   number of buckets (the "hashing trick"), lightly canonicalised with a small
   domain synonym map, then L2-normalised. It is NOT a neural model: it matches
   on shared/similar words, not deep semantics. It exists so tests/evaluation
   run without any API key and give reproducible retrieval.
2. GeminiEmbedding (EMBEDDING_PROVIDER=gemini): neural embeddings through the
   google-genai SDK (`client.models.embed_content`), model from EMBEDDING_MODEL.

INPUT  : list of strings.   OUTPUT: float32 numpy array (n, dim), L2-normalised.
CALLED BY: rag/ingestion.py, rag/retriever.py.   CALLS: google-genai (optional).
EXAMPLE: get_embedder().embed(["average rating of vehicles"]).shape -> (1, 384)
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from backend.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    name = "base"
    dim = 0

    @abstractmethod
    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


_STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are", "was", "were", "with",
         "by", "from", "that", "which", "who", "whom", "have", "has", "had", "all", "each", "every",
         "show", "find", "list", "give", "me", "what", "how", "many", "much", "their", "there", "than",
         "table", "columns", "column", "primary", "key", "foreign", "keys", "referenced", "description"}
_CANON = {
    "rented": "rental", "rentals": "rental", "rent": "rental", "renting": "rental", "booking": "rental",
    "rated": "rating", "ratings": "rating", "reviews": "review", "reviewed": "review", "stars": "rating",
    "cars": "vehicle", "car": "vehicle", "vehicles": "vehicle", "suv": "vehicle", "suvs": "vehicle",
    "sedan": "vehicle", "sedans": "vehicle",
    "spent": "amount", "spending": "amount", "revenue": "amount", "paid": "amount", "earned": "amount",
    "payment": "amount", "income": "amount",
    "serviced": "maintenance", "service": "maintenance", "servicing": "maintenance", "repair": "maintenance",
    "repairs": "maintenance",
    "clients": "customer", "customers": "customer", "users": "customer", "people": "customer",
    "branches": "branch", "location": "branch", "outlets": "branch",
    "price": "price", "prices": "price", "priced": "price", "expensive": "price", "cheapest": "price",
}


def _tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z]+", text.lower().replace("_", " "))
    out = []
    for t in raw:
        if t in _STOP:
            continue
        t = _CANON.get(t, t)
        if len(t) > 3 and t.endswith("s") and t not in _CANON.values():
            t = t[:-1]
        out.append(_CANON.get(t, t))
    return out


def _h(s: str, dim: int) -> tuple[int, float]:
    d = hashlib.md5(s.encode()).digest()
    return int.from_bytes(d[:4], "little") % dim, 1.0 if d[4] & 1 else -1.0


class LocalHashEmbedding(EmbeddingProvider):
    name = "local-hash"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokens(text):
                idx, sign = _h("w:" + tok, self.dim)
                mat[i, idx] += sign * 1.0
                padded = f"#{tok}#"
                for j in range(len(padded) - 2):
                    idx, sign = _h("c:" + padded[j:j + 3], self.dim)
                    mat[i, idx] += sign * 0.25
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class GeminiEmbedding(EmbeddingProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is not set (needed for EMBEDDING_PROVIDER=gemini).")
        from google import genai
        from google.genai import types
        self._client, self._types, self._model = genai.Client(api_key=api_key), types, model
        self.dim = 0

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(texts), 50):
                resp = self._client.models.embed_content(
                    model=self._model, contents=texts[i:i + 50],
                    config=self._types.EmbedContentConfig(task_type=task))
                vectors.extend(e.values for e in resp.embeddings)
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding failed: {type(exc).__name__}") from None
        mat = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dim = mat.shape[1]
        return mat / norms


def get_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    if s.embedding_provider == "gemini":
        return GeminiEmbedding(s.gemini_api_key, s.embedding_model)
    if s.embedding_provider == "local":
        return LocalHashEmbedding(s.embedding_dim)
    raise EmbeddingError(f"Unknown EMBEDDING_PROVIDER '{s.embedding_provider}'.")
