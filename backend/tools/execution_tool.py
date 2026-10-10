"""
FILE PURPOSE: execution TOOL - safety re-check, then run the SQL on the read-only DB.
The safety check is repeated here on purpose (defence in depth): even Basic RAG,
which skips schema validation, can never execute a write statement.
INPUT: ToolContext, sql.  OUTPUT: ExecutionResult.
CALLED BY: agents/orchestrator.py.  CALLS: validator_agent.check_safety, database/executor.py.
EXAMPLE: execute(ctx, "DELETE FROM rental").error -> "Blocked by safety check: ..."
"""
from __future__ import annotations

from backend.agents.validator_agent import check_safety
from backend.database.executor import execute_sql
from backend.schemas import ExecutionResult
from backend.tools.schema_tool import ToolContext


def execute(ctx: ToolContext, sql: str) -> ExecutionResult:
    problems = check_safety(sql)
    if problems:
        return ExecutionResult(success=False, error="Blocked by safety check: " + "; ".join(problems))
    return execute_sql(sql, ctx.engine, ctx.settings.max_result_rows)
