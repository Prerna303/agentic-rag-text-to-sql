"""
FILE PURPOSE
------------
Runs a validated, read-only SQL statement and returns rows as a pandas
DataFrame-derived result. Enforces a row cap so a huge result cannot flood the API.

WHY: keeps ALL database execution in one audited place.
INPUT  : SQL string.  OUTPUT: `ExecutionResult` (columns, rows, row_count, truncated, error).
CALLED BY: tools/execution_tool.py, evaluation/*.   CALLS: database/connection.py, pandas.
SAFETY : (1) validator already restricted SQL to SELECT/WITH, (2) SQLite connection
         has PRAGMA query_only=ON, (3) rows are capped at MAX_RESULT_ROWS.
EXAMPLE: execute_sql("SELECT COUNT(*) AS n FROM customers").rows -> [[100]]
"""
from __future__ import annotations

import math
import time

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.config import get_settings
from backend.database.connection import get_engine
from backend.schemas import ExecutionResult


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _root_message(exc: BaseException) -> str:
    """Drill down to the underlying sqlite3 error (drop SQLAlchemy/pandas wrapper noise)."""
    seen = 0
    while seen < 6:
        nxt = getattr(exc, "orig", None) or exc.__cause__
        if nxt is None:
            break
        exc, seen = nxt, seen + 1
    return str(exc)


def execute_sql(sql: str, engine: Engine | None = None, max_rows: int | None = None) -> ExecutionResult:
    engine = engine or get_engine()
    cap = max_rows or get_settings().max_result_rows
    start = time.perf_counter()
    try:
        with engine.connect() as con:
            df = pd.read_sql_query(text(sql), con)
    except Exception as exc:
        return ExecutionResult(success=False, error=_root_message(exc),
                               latency_ms=(time.perf_counter() - start) * 1000)
    truncated = len(df) > cap
    df_cap = df.head(cap)
    rows = [[_clean(v) for v in r] for r in df_cap.itertuples(index=False, name=None)]
    return ExecutionResult(success=True, columns=list(df.columns), rows=rows,
                           row_count=len(df), truncated=truncated,
                           latency_ms=(time.perf_counter() - start) * 1000)
