"""
FILE PURPOSE: retrieval TOOL - wraps SchemaRetriever (vector search + relationship expansion).
INPUT: ToolContext, question, expand flag.  OUTPUT: RetrievalResult.
CALLED BY: agents/orchestrator.py.  CALLS: rag/retriever.py.
EXAMPLE: retrieve_schema(ctx, "customers from Delhi").tables[0] -> 'customers'
"""
from __future__ import annotations

from backend.rag.retriever import RetrievalResult
from backend.tools.schema_tool import ToolContext


def retrieve_schema(ctx: ToolContext, question: str, expand: bool = True) -> RetrievalResult:
    return ctx.retriever.retrieve(question, top_k=ctx.settings.top_k, expand=expand)
