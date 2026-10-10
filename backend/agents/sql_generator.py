"""
FILE PURPOSE
------------
SQL GENERATION agent: (question + retrieved schema + relationships + query plan)
-> ONE SQLite SELECT statement, or an UNAVAILABLE marker when the schema cannot
answer the question. Also contains the simpler Basic-RAG generator (no plan).

INPUT  : LLMProvider, question, schema text, QueryPlan.
OUTPUT : (sql, unavailable_reason). Exactly one of them is non-empty/None.
CALLED BY: agents/orchestrator.py.   CALLS: llm/*, agents/common.py.
EXAMPLE: generate_sql(llm, q, schema, plan) -> ("SELECT v.model, AVG(r.rating) ...", None)
"""
from __future__ import annotations

from backend.agents.common import clean_sql, parse_unavailable, render_prompt
from backend.llm.base import LLMProvider
from backend.schemas import QueryPlan


def generate_sql(llm: LLMProvider, question: str, schema_text: str,
                 plan: QueryPlan | None) -> tuple[str, str | None]:
    plan_json = plan.model_dump_json(indent=2) if plan else "{}"
    raw = llm.generate(render_prompt("sql_generator", schema=schema_text, plan=plan_json, question=question))
    reason = parse_unavailable(raw)
    return ("", reason) if reason else (clean_sql(raw), None)


def generate_sql_basic(llm: LLMProvider, question: str, schema_text: str) -> tuple[str, str | None]:
    raw = llm.generate(render_prompt("basic_sql", schema=schema_text, question=question))
    reason = parse_unavailable(raw)
    return ("", reason) if reason else (clean_sql(raw), None)
