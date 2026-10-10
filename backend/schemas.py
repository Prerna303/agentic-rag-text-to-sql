"""
FILE PURPOSE
------------
All shared data structures (Pydantic models) in one file: schema description,
query plan, validation result, execution result, the AGENT STATE and the API
request/response bodies.

WHY THIS FILE EXISTS
--------------------
The agents must not pass "random strings" to each other. Every stage reads and
writes typed fields on one `AgentState` object, which is also what the frontend
shows as the agent trace.

AGENT STATE FIELDS
------------------
user_question, retrieved_schema, retrieved_tables, query_plan, generated_sql,
validation_result, database_error, repair_attempts, final_sql, execution_result,
final_answer  (+ trace, timings, tool_calls, status).

INPUT/OUTPUT: pure data classes.  CALLED BY: every module.  CALLS: pydantic only.
EXAMPLE: AgentState(user_question="Show all vehicles").status -> "running"
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ------------------------- schema description -------------------------
class ColumnInfo(BaseModel):
    name: str
    data_type: str
    is_primary_key: bool = False
    description: str = ""
    sample_values: list[str] = Field(default_factory=list)


class ForeignKey(BaseModel):
    column: str
    ref_table: str
    ref_column: str


class TableInfo(BaseModel):
    name: str
    description: str = ""
    columns: list[ColumnInfo]
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class DatabaseSchema(BaseModel):
    tables: list[TableInfo]

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def table(self, name: str) -> Optional[TableInfo]:
        for t in self.tables:
            if t.name.lower() == name.lower():
                return t
        return None

    def relationships(self) -> list[tuple[str, str, str, str]]:
        """(table, column, ref_table, ref_column) for every foreign key."""
        return [(t.name, fk.column, fk.ref_table, fk.ref_column)
                for t in self.tables for fk in t.foreign_keys]

    def subset(self, names: list[str]) -> "DatabaseSchema":
        keep = {n.lower() for n in names}
        return DatabaseSchema(tables=[t for t in self.tables if t.name.lower() in keep])


# ------------------------- pipeline objects -------------------------
class QueryPlan(BaseModel):
    feasible: bool = True
    missing: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    joins: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    aggregations: list[str] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    limit: Optional[int] = None


class ValidationResult(BaseModel):
    valid: bool
    level: str = "ok"          # safety | syntax | schema | database | ok
    errors: list[str] = Field(default_factory=list)
    hallucinated_tables: list[str] = Field(default_factory=list)
    hallucinated_columns: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    success: bool
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0


class TraceStep(BaseModel):
    step: int
    name: str
    ok: bool = True
    detail: str = ""


class AgentState(BaseModel):
    mode: str = "agentic"                     # agentic | basic
    user_question: str
    retrieved_schema: str = ""
    retrieved_tables: list[str] = Field(default_factory=list)
    ranked_tables: list[str] = Field(default_factory=list)   # raw vector ranking (before expansion)
    expanded_tables: list[str] = Field(default_factory=list)
    query_plan: Optional[QueryPlan] = None
    generated_sql: str = ""                   # first SQL produced
    validation_result: Optional[ValidationResult] = None
    first_validation: Optional[ValidationResult] = None
    database_error: Optional[str] = None
    repair_attempts: int = 0
    final_sql: str = ""
    execution_result: Optional[ExecutionResult] = None
    final_answer: str = ""
    status: str = "running"                   # success | unavailable | failed | running
    error: Optional[str] = None
    trace: list[TraceStep] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=lambda: {
        "retrieval": 0.0, "llm": 0.0, "validation": 0.0, "execution": 0.0, "total": 0.0})
    tool_calls: int = 0

    def add_step(self, name: str, ok: bool = True, detail: str = "") -> None:
        self.trace.append(TraceStep(step=len(self.trace) + 1, name=name, ok=ok, detail=detail))


# ------------------------- API bodies -------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: str = "agentic"
    debug: bool = True


class ValidateRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=5000)
