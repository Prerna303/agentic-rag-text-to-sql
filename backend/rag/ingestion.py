"""
FILE PURPOSE
------------
The schema INGESTION pipeline (run once, or whenever the schema changes):

    Database -> inspect schema -> table-wise chunks -> text -> embeddings
             -> vector store on disk (+ data/schema/schema.json snapshot)

WHY: the SQL agent must not send the whole schema to the LLM on every request.
Ingestion builds a searchable index so only relevant tables are retrieved.

INPUT  : Settings (DATABASE_URL, embedding provider, VECTOR_STORE_DIR).
OUTPUT : files in vector_store/ ; returns a small stats dict.
CALLED BY: `python -m backend.rag.ingestion`, POST /api/ingest, tests, evaluation.
CALLS  : database/schema_inspector.py, rag/chunking.py, rag/embeddings.py, rag/vector_store.py.
EXAMPLE: ingest_schema() -> {"tables": 6, "chunks": 6, "dim": 384, "embedder": "local-hash"}
"""
from __future__ import annotations

import json
import time

from backend.config import PROJECT_ROOT, Settings, get_settings
from backend.database.connection import get_engine
from backend.database.schema_inspector import inspect_database
from backend.logging_utils import log_event
from backend.rag.chunking import build_chunks
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import VectorStore


def ingest_schema(settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    t0 = time.perf_counter()
    schema = inspect_database(get_engine(s.database_url))
    chunks = build_chunks(schema)
    embedder = get_embedder(s)
    vectors = embedder.embed([c.text for c in chunks])
    items = [{"table": c.table, "text": c.text, "metadata": c.metadata} for c in chunks]
    store = VectorStore()
    store.build(vectors, items, info={"embedder": embedder.name, "dim": int(vectors.shape[1]),
                                      "model": s.embedding_model if s.embedding_provider == "gemini" else "local-hash"})
    store.save(s.vector_store_dir)
    snap = PROJECT_ROOT / "data" / "schema" / "schema.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(schema.model_dump_json(indent=2), encoding="utf-8")
    stats = {"tables": len(schema.tables), "chunks": len(chunks), "dim": int(vectors.shape[1]),
             "embedder": embedder.name, "seconds": round(time.perf_counter() - t0, 3)}
    log_event("ingestion", **stats)
    return stats


if __name__ == "__main__":
    print(json.dumps(ingest_schema(), indent=2))
