"""
FILE PURPOSE
------------
The ORCHESTRATOR: runs the whole pipeline and fills the `AgentState`.

AGENTIC MODE (default)
----------------------
 1 understand question -> 2 schema RAG retrieval (+ relationship expansion)
 3 query plan (LLM)    -> 4 SQL generation (LLM)
 5 validate (safety/syntax/schema/EXPLAIN)
      invalid -> error analysis -> repair (LLM) -> validate again ... (<= MAX_REPAIR_ATTEMPTS)
 6 execute (a runtime error also goes through the repair loop)
 7 explain result (LLM)

BASIC-RAG MODE (baseline for comparison)
----------------------------------------
 retrieve top-K (no expansion) -> ONE LLM call -> execute. No plan, no validation,
 no repair, no explanation. Only the mandatory read-only safety gate remains.

SPECIAL OUTCOMES
----------------
status = "success" | "unavailable" (schema cannot answer) | "failed" (errors / retries exhausted)

INPUT  : question (+ mode, optional ToolContext / LLMProvider for tests).
OUTPUT : `AgentState` with trace, timings, tool_calls, final_sql, execution_result.
CALLED BY: main.py (/api/query), evaluation/*, tests.
CALLS  : tools/*, agents/*, llm/factory.py, logging_utils.py.
EXAMPLE: run_agent("Show all vehicles").status -> "success"
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from backend.agents.explanation_agent import explain
from backend.agents.planner_agent import plan_query
from backend.agents.repair_agent import repair_sql
from backend.agents.sql_generator import generate_sql, generate_sql_basic
from backend.llm.base import LLMError, LLMProvider
from backend.llm.factory import get_llm
from backend.logging_utils import log_event
from backend.rag.embeddings import EmbeddingError
from backend.rag.vector_store import VectorStoreError
from backend.schemas import AgentState
from backend.tools.execution_tool import execute
from backend.tools.retrieval_tool import retrieve_schema
from backend.tools.schema_tool import ToolContext, get_context
from backend.tools.validation_tool import validate

UNAVAILABLE_MSG = "The requested information is not available in the database schema."


@contextmanager
def _timed(state: AgentState, key: str, count_tool: bool = True):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        state.timings[key] = state.timings.get(key, 0.0) + (time.perf_counter() - t0) * 1000
        if count_tool:
            state.tool_calls += 1


def _unavailable(state: AgentState, reason: str) -> None:
    state.status = "unavailable"
    state.final_answer = f"{UNAVAILABLE_MSG} ({reason})" if reason else UNAVAILABLE_MSG
    state.add_step("Schema cannot answer this question", ok=True, detail=reason)


def run_agent(question: str, mode: str = "agentic", ctx: ToolContext | None = None,
              llm: LLMProvider | None = None) -> AgentState:
    state = AgentState(mode=mode, user_question=question.strip())
    t_start = time.perf_counter()
    log_event("question", question=state.user_question, mode=mode)
    try:
        ctx = ctx or get_context()
        llm = llm or get_llm(ctx.settings)
        if mode == "basic":
            _run_basic(state, ctx, llm)
        else:
            _run_agentic(state, ctx, llm)
    except LLMError as exc:
        state.status, state.error = "failed", f"LLM error: {exc}"
        state.add_step("LLM call", ok=False, detail=str(exc))
    except (EmbeddingError, VectorStoreError) as exc:
        state.status, state.error = "failed", f"Retrieval error: {exc}"
        state.add_step("Retrieving schema", ok=False, detail=str(exc))
    except Exception as exc:  # last-resort guard: never crash the API
        state.status, state.error = "failed", f"Unexpected error: {type(exc).__name__}: {exc}"
        state.add_step("Unexpected error", ok=False, detail=str(exc))
    state.timings["total"] = (time.perf_counter() - t_start) * 1000
    log_event("finished", status=state.status, final_sql=state.final_sql, repairs=state.repair_attempts,
              latency_ms=round(state.timings["total"], 1), error=state.error)
    return state


def _retrieve(state: AgentState, ctx: ToolContext, expand: bool) -> None:
    state.add_step("Understanding question")
    with _timed(state, "retrieval"):
        res = retrieve_schema(ctx, state.user_question, expand=expand)
    state.retrieved_schema, state.retrieved_tables = res.schema_text, res.tables
    state.ranked_tables = [t for t, _ in res.ranked]
    state.expanded_tables = res.expanded
    state.add_step("Retrieving schema", detail="Retrieved tables: " + ", ".join(res.tables) +
                   (f" (bridge tables added: {', '.join(res.expanded)})" if res.expanded else ""))
    log_event("retrieval", tables=res.tables, bridges=res.expanded)


def _run_basic(state: AgentState, ctx: ToolContext, llm: LLMProvider) -> None:
    _retrieve(state, ctx, expand=False)
    with _timed(state, "llm"):
        sql, reason = generate_sql_basic(llm, state.user_question, state.retrieved_schema)
    if reason is not None:
        return _unavailable(state, reason)
    state.generated_sql = state.final_sql = sql
    state.add_step("Generating SQL (single LLM call)", detail=sql)
    with _timed(state, "execution"):
        res = execute(ctx, sql)
    state.execution_result = res
    if res.success:
        state.status = "success"
        state.final_answer = f"Query returned {res.row_count} row(s)."
        state.add_step("Executing query")
    else:
        state.status, state.error, state.database_error = "failed", res.error, res.error
        state.add_step("Executing query", ok=False, detail=res.error or "")


def _run_agentic(state: AgentState, ctx: ToolContext, llm: LLMProvider) -> None:
    _retrieve(state, ctx, expand=True)

    with _timed(state, "llm"):
        state.query_plan = plan_query(llm, state.user_question, state.retrieved_schema)
    state.add_step("Creating query plan", detail=state.query_plan.model_dump_json())
    if not state.query_plan.feasible:
        return _unavailable(state, "; ".join(state.query_plan.missing))

    with _timed(state, "llm"):
        sql, reason = generate_sql(llm, state.user_question, state.retrieved_schema, state.query_plan)
    if reason is not None:
        return _unavailable(state, reason)
    state.generated_sql = sql
    state.add_step("Generating SQL", detail=sql)

    history: list[str] = []
    current = sql
    while True:
        with _timed(state, "validation"):
            v = validate(ctx, current)
        state.validation_result = v
        if state.first_validation is None:
            state.first_validation = v
        state.add_step("Validating SQL" if state.repair_attempts == 0 else "Re-validating",
                       ok=v.valid, detail="" if v.valid else f"[{v.level}] " + "; ".join(v.errors))
        log_event("validation", valid=v.valid, validation_level=v.level, errors=v.errors)

        error_text: list[str] = []
        if v.valid:
            with _timed(state, "execution"):
                res = execute(ctx, current)
            state.execution_result = res
            if res.success:
                state.final_sql = current
                state.add_step("Executing query", detail=f"{res.row_count} row(s)")
                break
            state.add_step("Executing query", ok=False, detail=res.error or "")
            error_text = [res.error or "execution failed"]
            state.database_error = res.error
        else:
            error_text = v.errors
            state.database_error = "; ".join(v.errors)

        if state.repair_attempts >= ctx.settings.max_repair_attempts:
            state.status = "failed"
            state.error = f"Maximum repair attempts reached. Last error: {'; '.join(error_text)}"
            state.final_sql = current
            state.add_step("Maximum retries reached", ok=False, detail=state.error)
            return
        history.append(f"{current}  --> {'; '.join(error_text)[:200]}")
        with _timed(state, "llm"):
            new_sql, reason = repair_sql(llm, state.user_question, state.retrieved_schema, ctx.schema,
                                         state.query_plan, current, error_text, history)
        state.repair_attempts += 1
        if reason is not None:
            state.final_sql = ""
            return _unavailable(state, reason)
        state.add_step(f"Repairing SQL (attempt {state.repair_attempts})", detail=new_sql)
        log_event("repair", attempt=state.repair_attempts, sql=new_sql)
        current = new_sql

    with _timed(state, "llm"):
        state.final_answer = explain(llm, state.user_question, state.final_sql, state.execution_result)
    state.status = "success"
    state.add_step("Generating explanation")
