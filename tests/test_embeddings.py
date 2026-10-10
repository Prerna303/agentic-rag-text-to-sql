import numpy as np

from backend.rag.embeddings import LocalHashEmbedding, get_embedder


def test_deterministic_and_normalised():
    e = LocalHashEmbedding(128)
    a, b = e.embed(["average rating"]), e.embed(["average rating"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a[0]) - 1) < 1e-5


def test_similar_text_is_closer():
    e = LocalHashEmbedding()
    v = e.embed(["customers from Delhi", "customer city Delhi", "vehicle maintenance cost"])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_default_provider_is_local():
    assert get_embedder().name == "local-hash"
