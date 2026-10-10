"""
FILE PURPOSE
------------
ERROR ANALYSIS + SQL REPAIR agent (the self-correcting heart of the system).

STEP 1 - ERROR ANALYSIS (deterministic, no LLM): turn the raw error into hints.
  e.g. error "Column 'vehicles.id' does not exist" ->
       "vehicles.id does not exist. Closest real columns of vehicles: vehicle_id."
  (uses difflib to find the nearest real column / table names).
STEP 2 - REPAIR (LLM): the prompt contains the schema, plan, failed SQL, the error,
  the analysis and earlier failed attempts (so the model does not repeat itself).
  The LLM may also answer UNAVAILABLE if the schema truly cannot answer.

The orchestrator limits the number of repair rounds (MAX_REPAIR_ATTEMPTS) so the
loop can never run forever.

INPUT  : LLM, question, schema text (str), DatabaseSchema, plan, failed SQL, errors, history.
OUTPUT : (repaired_sql, unavailable_reason).
CALLED BY: agents/orchestrator.py.   CALLS: llm/*, agents/common.py, difflib.
EXAMPLE: analyze_errors(["Column 'c.id' does not exist ..."], schema)
         -> "customers.id does not exist. Closest real columns: customer_id"
"""
from __future__ import annotations

import difflib
import re

from backend.agents.common import clean_sql, parse_unavailable, render_prompt
from backend.llm.base import LLMProvider
from backend.schemas import DatabaseSchema, QueryPlan


def analyze_errors(errors: list[str], schema: DatabaseSchema) -> str:
    hints: list[str] = []
    all_cols = sorted({c for t in schema.tables for c in t.column_names})
    for err in errors:
        m = re.search(r"Column '(?:(\w+)\.)?(\w+)' does not exist", err)
        if m:
            tname, col = m.group(1), m.group(2)
            table = schema.table(tname) if tname else None
            pool = table.column_names if table else all_cols
            close = difflib.get_close_matches(col, pool, n=3, cutoff=0.4)
            if not close:
                close = [c for c in pool if col.lower() in c.lower() or c.lower() in col.lower()][:3]
            hints.append(f"'{col}' does not exist" + (f" in {tname}" if tname else "") +
                         (f". Closest real columns: {', '.join(close)}." if close else
                          ". No similar column exists; the information may be unavailable."))
            continue
        m = re.search(r"Table '(\w+)' does not exist", err)
        if m:
            close = difflib.get_close_matches(m.group(1), schema.table_names, n=3, cutoff=0.4)
            hints.append(f"Table '{m.group(1)}' does not exist" +
                         (f". Closest real tables: {', '.join(close)}." if close else ". No similar table exists."))
            continue
        hints.append(err)
    return "\n".join(hints) if hints else "No specific hint."


def repair_sql(llm: LLMProvider, question: str, schema_text: str, schema: DatabaseSchema,
               plan: QueryPlan | None, failed_sql: str, errors: list[str],
               history: list[str]) -> tuple[str, str | None]:
    analysis = analyze_errors(errors, schema)
    prompt = render_prompt(
        "repair", schema=schema_text, plan=plan.model_dump_json() if plan else "{}",
        failed_sql=failed_sql, error="\n".join(errors), analysis=analysis,
        history="\n".join(f"- {h}" for h in history) or "(none)", question=question)
    raw = llm.generate(prompt)
    reason = parse_unavailable(raw)
    return ("", reason) if reason else (clean_sql(raw), None)
