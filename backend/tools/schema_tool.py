"""
FILE PURPOSE
------------
Schema TOOL + shared `ToolContext`. The context bundles everything the tools need
(engine, discovered schema, retriever) so they are built once and reused.

WHY: "tool calling" in this project means the orchestrator invokes explicit,
separately testable tools (schema, retrieval, validation, execution) rather than
hiding everything inside one prompt.

INPUT  : Settings.  OUTPUT: ToolContext (engine, schema, retriever).
CALLED BY: tools/*.py, agents/orchestrator.py, main.py, evaluation/*.
CALLS  : database/*, rag/*.
EXAMPLE: ctx = get_context(); ctx.schema.table_names
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine

from backend.config import Settings, get_settings
from backend.database.connection import get_engine
from backend.database.schema_inspector import inspect_database
from backend.rag.retriever import SchemaRetriever
from backend.rag.vector_store import VectorStore
from backend.schemas import DatabaseSchema


@dataclass
class ToolContext:
    settings: Settings
    engine: Engine
    schema: DatabaseSchema
    retriever: SchemaRetriever


_CTX: ToolContext | None = None


def build_context(settings: Settings | None = None) -> ToolContext:
    s = settings or get_settings()
    engine = get_engine(s.database_url)
    schema = inspect_database(engine)
    retriever = SchemaRetriever(schema, VectorStore.load(s.vector_store_dir), settings=s)
    return ToolContext(s, engine, schema, retriever)


def get_context() -> ToolContext:
    global _CTX
    if _CTX is None:
        _CTX = build_context()
    return _CTX


def reset_context() -> None:
    global _CTX
    _CTX = None


def describe_schema(ctx: ToolContext) -> dict:
    return {"tables": [t.model_dump() for t in ctx.schema.tables],
            "relationships": [{"table": a, "column": b, "ref_table": c, "ref_column": d}
                              for a, b, c, d in ctx.schema.relationships()]}
