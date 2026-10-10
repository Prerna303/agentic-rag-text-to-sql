"""
FILE PURPOSE
------------
Automatically DISCOVERS the schema of any SQL database via SQLAlchemy's
inspector: tables, columns, data types, primary keys, foreign keys, plus
optional human descriptions and a few sample values per text column.
Nothing about the demo schema is hard-coded here or in the RAG pipeline.

WHY: the RAG pipeline needs structured schema facts to build text chunks, and the
validator needs the same facts to detect hallucinated tables/columns.

INPUT  : SQLAlchemy Engine (+ optional data/schema/descriptions.json).
OUTPUT : `DatabaseSchema` (pydantic) -> list of `TableInfo`.
CALLED BY: rag/ingestion.py, tools/schema_tool.py, agents/validator_agent.py, main.py.
CALLS  : database/connection.py.
EXAMPLE: inspect_database(engine).table("rental").foreign_keys[0]
         -> ForeignKey(column="customer_id", ref_table="customers", ref_column="customer_id")
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from backend.config import PROJECT_ROOT
from backend.schemas import ColumnInfo, DatabaseSchema, ForeignKey, TableInfo

DESCRIPTIONS_PATH = PROJECT_ROOT / "data" / "schema" / "descriptions.json"


def _load_descriptions(path: Path | None = None) -> dict:
    p = path or DESCRIPTIONS_PATH
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def inspect_database(engine: Engine, descriptions_path: Path | None = None,
                     sample_values: int = 8) -> DatabaseSchema:
    insp = inspect(engine)
    desc = _load_descriptions(descriptions_path)
    tables: list[TableInfo] = []
    for tname in sorted(insp.get_table_names()):
        if tname.startswith("sqlite_"):
            continue
        pk = set(insp.get_pk_constraint(tname).get("constrained_columns") or [])
        tdesc = desc.get(tname, {})
        cols = []
        for c in insp.get_columns(tname):
            cols.append(ColumnInfo(
                name=c["name"], data_type=str(c["type"]), is_primary_key=c["name"] in pk,
                description=tdesc.get("columns", {}).get(c["name"], "")))
        fks = []
        for fk in insp.get_foreign_keys(tname):
            for lc, rc in zip(fk["constrained_columns"], fk["referred_columns"]):
                fks.append(ForeignKey(column=lc, ref_table=fk["referred_table"], ref_column=rc))
        info = TableInfo(name=tname, description=tdesc.get("description", ""),
                         columns=cols, primary_keys=sorted(pk), foreign_keys=fks)
        _attach_samples(engine, info, sample_values)
        tables.append(info)
    return DatabaseSchema(tables=tables)


def _attach_samples(engine: Engine, info: TableInfo, limit: int) -> None:
    """Low-cardinality TEXT columns get sample values (e.g. city = Delhi, Mumbai).
    These make chunks match questions such as 'customers from Delhi'."""
    with engine.connect() as con:
        for col in info.columns:
            if "CHAR" in col.data_type.upper() or "TEXT" in col.data_type.upper():
                if col.name.endswith("date") or col.name in ("email", "review_text", "name"):
                    continue
                try:
                    rows = con.execute(text(
                        f'SELECT DISTINCT "{col.name}" FROM "{info.name}" '
                        f'WHERE "{col.name}" IS NOT NULL LIMIT 200')).fetchall()
                except Exception:
                    continue
                vals = sorted(str(r[0]) for r in rows)
                if 0 < len(vals) <= 25:
                    col.sample_values = vals[:limit]
