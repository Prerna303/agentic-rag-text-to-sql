"""
FILE PURPOSE
------------
QUERY PLANNING agent. Before any SQL is written, it asks the LLM for the LOGICAL
structure of the query as JSON: tables, joins, filters, group_by, aggregations,
order_by, limit (and `feasible` / `missing` when the schema cannot answer).

WHY: planning first separates "what data do I need and how is it related?" from
"how do I spell that in SQL?". It reduces hallucinated joins and lets the system
say "not available" BEFORE writing SQL.

INPUT  : LLMProvider, question, retrieved schema text.
OUTPUT : `QueryPlan` (pydantic). If the LLM output is not valid JSON, a permissive
         fallback plan (feasible=True, everything empty) is returned so the
         pipeline can continue.
CALLED BY: agents/orchestrator.py.   CALLS: llm/*, agents/common.py.
EXAMPLE: plan_query(llm, "average rating of each vehicle", schema_text).joins
         -> ["vehicles.vehicle_id = rental.vehicle_id", "rental.rental_id = rating.rental_id"]
"""
from __future__ import annotations

from backend.agents.common import extract_json, render_prompt
from backend.llm.base import LLMProvider
from backend.schemas import QueryPlan


def plan_query(llm: LLMProvider, question: str, schema_text: str) -> QueryPlan:
    raw = llm.generate(render_prompt("planner", schema=schema_text, question=question))
    try:
        data = extract_json(raw)
        allowed = QueryPlan.model_fields.keys()
        return QueryPlan(**{k: v for k, v in data.items() if k in allowed})
    except Exception:
        return QueryPlan()
