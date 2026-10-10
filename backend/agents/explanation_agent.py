"""
FILE PURPOSE
------------
RESULT EXPLANATION agent: writes a short plain-language explanation of what the SQL
did and what the result says. If the LLM fails, a deterministic explanation is used
so the user still gets an answer.

INPUT  : LLM, question, final SQL, ExecutionResult.   OUTPUT: explanation string.
CALLED BY: agents/orchestrator.py.   CALLS: llm/*, agents/common.py.
EXAMPLE: explain(llm, "How many customers?", "SELECT COUNT(*) ...", result)
         -> "The query counts all rows in customers. The result is 100."
"""
from __future__ import annotations

from backend.agents.common import render_prompt
from backend.llm.base import LLMError, LLMProvider
from backend.schemas import ExecutionResult


def fallback_explanation(sql: str, res: ExecutionResult) -> str:
    if res.row_count == 0:
        return "The query ran successfully but returned no rows."
    return (f"The query ran successfully and returned {res.row_count} row(s) with columns "
            f"{', '.join(res.columns)}.")


def explain(llm: LLMProvider, question: str, sql: str, res: ExecutionResult) -> str:
    preview = "\n".join(str(r) for r in res.rows[:10]) or "(no rows)"
    try:
        return llm.generate(render_prompt("explanation", question=question, sql=sql,
                                          result=preview, row_count=str(res.row_count))).strip()
    except LLMError:
        return fallback_explanation(sql, res)
