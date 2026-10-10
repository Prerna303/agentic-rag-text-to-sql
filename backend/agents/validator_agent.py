"""
FILE PURPOSE
------------
The SQL VALIDATION agent/tool. Three escalating levels plus a safety gate:

  Level 0  SAFETY   : exactly ONE statement, and it must be a read-only
                      SELECT / WITH (UNION allowed). Any DROP, DELETE, UPDATE,
                      INSERT, ALTER, TRUNCATE, CREATE, REPLACE, ATTACH, PRAGMA...
                      anywhere in the parse tree is rejected. Multiple
                      statements (`SELECT 1; DROP TABLE x`) are rejected.
  Level 1  SYNTAX   : the SQL must parse with sqlglot (SQLite dialect).
  Level 2  SCHEMA   : every table exists; every column exists in the table it
                      is qualified with (aliases resolved) or, if unqualified,
                      in at least one table of the query. CTE names, subquery
                      aliases and SELECT aliases are understood.
  Level 3  DATABASE : `EXPLAIN QUERY PLAN <sql>` on the read-only SQLite
                      connection. This asks the real engine to compile the
                      query WITHOUT running it.

SQL-injection note: the system never pastes user text into SQL itself; the LLM
writes the SQL and this validator (plus a read-only DB connection) is the gate.
Where the app inserts values (e.g. tests) it uses bound parameters.

INPUT  : sql string, DatabaseSchema, optional Engine.
OUTPUT : `ValidationResult` (valid, level, errors, hallucinated tables/columns).
CALLED BY: tools/validation_tool.py, agents/orchestrator.py, evaluation/metrics.py, API.
CALLS  : sqlglot, sqlalchemy.
EXAMPLE: validate_sql("SELECT vehicles.id FROM vehicles", schema)
         -> valid=False, level="schema", errors=["Column 'vehicles.id' does not exist. ..."]
"""
from __future__ import annotations

import re

import sqlglot
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlglot import exp

from backend.schemas import DatabaseSchema, ValidationResult

_FORBIDDEN_NODES = tuple(getattr(exp, n) for n in (
    "Drop", "Delete", "Update", "Insert", "Alter", "Create", "TruncateTable", "Command",
    "Merge", "Attach", "Pragma", "Set", "Use", "Copy", "Grant", "Transaction", "Commit", "Rollback")
    if hasattr(exp, n))
_FORBIDDEN_WORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|"
    r"LOAD_EXTENSION|GRANT|REVOKE)\b", re.I)
_DANGEROUS_FUNCS = {"load_extension", "readfile", "writefile", "edit"}


def _strip_literals(sql: str) -> str:
    """Remove quoted strings so words inside literals ('DROP') are not flagged."""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def check_safety(sql: str) -> list[str]:
    errors = []
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return ["Empty SQL."]
    try:
        stmts = [s for s in sqlglot.parse(stripped, read="sqlite") if s is not None]
    except Exception:
        stmts = []
    if len(stmts) > 1 or ";" in _strip_literals(stripped):
        errors.append("Multiple statements are not allowed.")
    if _FORBIDDEN_WORDS.search(_strip_literals(stripped)):
        errors.append("Only read-only SELECT/WITH queries are allowed "
                      "(DROP/DELETE/UPDATE/INSERT/ALTER/TRUNCATE/CREATE are blocked).")
    for s in stmts:
        if not isinstance(s, (exp.Select, exp.Union, exp.Subquery, exp.With)) and not errors:
            errors.append("Only SELECT or WITH statements are allowed.")
        if any(True for _ in s.find_all(*_FORBIDDEN_NODES)) and not errors:
            errors.append("Statement contains a forbidden (write/DDL) operation.")
        for f in s.find_all(exp.Anonymous):
            if str(f.this).lower() in _DANGEROUS_FUNCS:
                errors.append(f"Dangerous function '{f.this}' is not allowed.")
    return errors


def check_schema(sql: str, schema: DatabaseSchema) -> tuple[list[str], list[str], list[str]]:
    """Returns (errors, hallucinated_tables, hallucinated_columns)."""
    tree = sqlglot.parse_one(sql.strip().rstrip(";"), read="sqlite")
    errors: list[str] = []
    bad_tables: list[str] = []
    bad_cols: list[str] = []
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    alias_to_table: dict[str, str] = {}
    query_tables: list[str] = []
    derived_aliases: set[str] = set()

    for sub in tree.find_all(exp.Subquery):
        if sub.alias:
            derived_aliases.add(sub.alias.lower())
    for t in tree.find_all(exp.Table):
        name = t.name
        if not name or name.lower() in ctes:
            if name and t.alias:
                derived_aliases.add(t.alias.lower())
            continue
        real = schema.table(name)
        if real is None:
            if name not in bad_tables:
                bad_tables.append(name)
                errors.append(f"Table '{name}' does not exist. Available tables: "
                              f"{', '.join(schema.table_names)}.")
            continue
        query_tables.append(real.name)
        alias_to_table[(t.alias or real.name).lower()] = real.name
        alias_to_table[real.name.lower()] = real.name

    select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    for col in tree.find_all(exp.Column):
        cname, qual = col.name, (col.table or "").lower()
        if cname == "*" or not cname:
            continue
        if qual:
            if qual in derived_aliases or qual in ctes:
                continue
            tname = alias_to_table.get(qual)
            if tname is None:
                if not bad_tables:
                    errors.append(f"Unknown table or alias '{col.table}' used for column '{cname}'.")
                continue
            table = schema.table(tname)
            if cname.lower() not in [c.lower() for c in table.column_names]:
                ref = f"{tname}.{cname}"
                if ref not in bad_cols:
                    bad_cols.append(ref)
                    errors.append(f"Column '{ref}' does not exist. Columns of {tname}: "
                                  f"{', '.join(table.column_names)}.")
        else:
            if cname.lower() in select_aliases:
                continue
            if not query_tables and (ctes or derived_aliases):
                continue
            found = any(cname.lower() in [c.lower() for c in schema.table(t).column_names]
                        for t in set(query_tables))
            if not found and not (ctes or derived_aliases):
                if cname not in bad_cols:
                    bad_cols.append(cname)
                    errors.append(f"Column '{cname}' does not exist in tables "
                                  f"{', '.join(sorted(set(query_tables))) or '(none)'}.")
    return errors, bad_tables, bad_cols


def validate_sql(sql: str, schema: DatabaseSchema, engine: Engine | None = None) -> ValidationResult:
    sql = (sql or "").strip()
    safety = check_safety(sql)
    if safety:
        return ValidationResult(valid=False, level="safety", errors=safety)
    try:
        sqlglot.parse_one(sql.rstrip(";"), read="sqlite", error_level=sqlglot.ErrorLevel.RAISE)
    except Exception as exc:
        return ValidationResult(valid=False, level="syntax", errors=[f"Syntax error: {re.sub(chr(27) + r'\[[0-9;]*m', '', str(exc))[:300]}"])
    errs, bt, bc = check_schema(sql, schema)
    if errs:
        return ValidationResult(valid=False, level="schema", errors=errs,
                                hallucinated_tables=bt, hallucinated_columns=bc)
    if engine is not None:
        try:
            with engine.connect() as con:
                con.execute(text("EXPLAIN QUERY PLAN " + sql.rstrip(";"))).fetchall()
        except Exception as exc:
            return ValidationResult(valid=False, level="database",
                                    errors=[str(getattr(exc, "orig", exc))[:300]])
    return ValidationResult(valid=True, level="ok")
