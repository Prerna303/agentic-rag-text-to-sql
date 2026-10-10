"""
FILE PURPOSE
------------
The FastAPI application: HTTP API + static frontend.

ENDPOINTS
---------
POST /api/query         question -> full agent run (SQL, result, plan, trace, timings)
GET  /api/schema        discovered schema (tables, columns, PK/FK, relationships)
POST /api/ingest        (re)build the vector store from the current database
GET  /api/health        liveness + configuration summary (never shows API keys)
POST /api/validate-sql  run the validator (safety/syntax/schema/EXPLAIN) on any SQL text
GET  /                  the web UI (frontend/index.html)

INPUT  : JSON bodies (pydantic validated).   OUTPUT: JSON.
CALLED BY: browser / curl / tests.   CALLS: agents/orchestrator.py, tools/*, rag/ingestion.py.
RUN    : uvicorn backend.main:app --reload
EXAMPLE: curl -X POST localhost:8000/api/query -H 'content-type: application/json' \
              -d '{"question": "How many customers are there?"}'
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.agents.orchestrator import run_agent
from backend.config import get_settings
from backend.rag.embeddings import EmbeddingError
from backend.rag.ingestion import ingest_schema
from backend.rag.vector_store import VectorStoreError
from backend.schemas import QueryRequest, ValidateRequest
from backend.tools.schema_tool import describe_schema, get_context, reset_context
from backend.tools.validation_tool import validate

app = FastAPI(title="Agentic RAG Text-to-SQL", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def _ctx():
    try:
        return get_context()
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=f"{exc}") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not initialise: {type(exc).__name__}: {exc}") from None


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "llm_provider": "mock" if s.mock_llm else s.llm_provider,
            "gemini_key_configured": bool(s.gemini_api_key), "embedding_provider": s.embedding_provider,
            "top_k": s.top_k, "max_repair_attempts": s.max_repair_attempts}


@app.get("/api/schema")
def schema() -> dict:
    return describe_schema(_ctx())


@app.post("/api/ingest")
def ingest() -> dict:
    try:
        stats = ingest_schema()
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    reset_context()
    return {"status": "ok", **stats}


@app.post("/api/validate-sql")
def validate_sql_endpoint(req: ValidateRequest) -> dict:
    return validate(_ctx(), req.sql).model_dump()


@app.post("/api/query")
def query(req: QueryRequest) -> dict:
    if req.mode not in ("agentic", "basic"):
        raise HTTPException(status_code=422, detail="mode must be 'agentic' or 'basic'")
    ctx = _ctx()
    st = run_agent(req.question, mode=req.mode, ctx=ctx)
    res = st.execution_result
    body = {
        "question": st.user_question, "mode": st.mode, "status": st.status,
        "sql": st.final_sql or st.generated_sql,
        "result": {"columns": res.columns, "rows": res.rows, "row_count": res.row_count,
                   "truncated": res.truncated} if res and res.success else None,
        "answer": st.final_answer, "error": st.error,
        "retrieved_tables": st.retrieved_tables,
        "query_plan": st.query_plan.model_dump() if st.query_plan else None,
        "execution_success": bool(res and res.success),
        "repair_attempts": st.repair_attempts,
        "latency_ms": round(st.timings["total"], 1),
        "agent_steps": len(st.trace), "tool_calls": st.tool_calls,
    }
    if req.debug:
        body["debug"] = {"retrieved_schema": st.retrieved_schema, "ranked_tables": st.ranked_tables,
                         "bridge_tables": st.expanded_tables,
                         "trace": [t.model_dump() for t in st.trace],
                         "timings_ms": {k: round(v, 1) for k, v in st.timings.items()},
                         "first_sql": st.generated_sql, "database_error": st.database_error}
    return body


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
