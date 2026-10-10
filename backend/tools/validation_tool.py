"""
FILE PURPOSE: validation TOOL - wraps agents/validator_agent.validate_sql with the
context's schema and read-only engine (so level 3 `EXPLAIN` is included).
INPUT: ToolContext, sql.  OUTPUT: ValidationResult.
CALLED BY: agents/orchestrator.py, main.py (/api/validate-sql).
EXAMPLE: validate(ctx, "DROP TABLE customers").level -> "safety"
"""
from __future__ import annotations

from backend.agents.validator_agent import validate_sql
from backend.schemas import ValidationResult
from backend.tools.schema_tool import ToolContext


def validate(ctx: ToolContext, sql: str) -> ValidationResult:
    return validate_sql(sql, ctx.schema, ctx.engine)
