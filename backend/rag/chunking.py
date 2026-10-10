"""
FILE PURPOSE
------------
Converts a discovered `DatabaseSchema` into RAG "documents" (chunks).

CHUNKING STRATEGY (exact)
-------------------------
TABLE-WISE chunking: exactly ONE chunk per table. We do NOT cut the schema into
fixed-size token windows, because a window could split a table in half (columns
in one chunk, its primary key or foreign keys in another) and the LLM would
then see an incomplete table. A table is the natural unit a SQL query refers to.

Because joins are essential for SQL, every table chunk also carries:
  * its OUTGOING foreign keys        (rental.customer_id -> customers.customer_id)
  * the tables that REFERENCE it     ("Referenced by: rating.rental_id")
so a chunk is self-contained about relationships.

WHAT TEXT IS EMBEDDED (example)
-------------------------------
    Table: vehicles
    Description: Stores rental vehicles.
    Columns:
    - vehicle_id INTEGER PRIMARY KEY
    - category VARCHAR (values: Hatchback, Luxury, SUV, Sedan, Van)
    Primary Key: vehicle_id
    Foreign Keys: branch_id -> branches.branch_id
    Referenced by: rental.vehicle_id, maintenance.vehicle_id

METADATA stored next to the vector (not embedded):
    {"table": "vehicles", "type": "table_schema", "primary_keys": [...],
     "foreign_keys": [...], "columns": [...]}

INPUT  : DatabaseSchema.   OUTPUT: list[SchemaChunk].
CALLED BY: rag/ingestion.py, tests.   CALLS: backend/schemas.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.schemas import DatabaseSchema, TableInfo


@dataclass
class SchemaChunk:
    table: str
    text: str
    metadata: dict = field(default_factory=dict)


def table_to_text(table: TableInfo, schema: DatabaseSchema) -> str:
    lines = [f"Table: {table.name}"]
    if table.description:
        lines.append(f"Description: {table.description}")
    lines.append("Columns:")
    for c in table.columns:
        extra = " PRIMARY KEY" if c.is_primary_key else ""
        desc = f" - {c.description}" if c.description else ""
        vals = f" (values: {', '.join(c.sample_values)})" if c.sample_values else ""
        lines.append(f"- {c.name} {c.data_type}{extra}{desc}{vals}")
    if table.primary_keys:
        lines.append(f"Primary Key: {', '.join(table.primary_keys)}")
    if table.foreign_keys:
        lines.append("Foreign Keys: " + ", ".join(
            f"{fk.column} -> {fk.ref_table}.{fk.ref_column}" for fk in table.foreign_keys))
    referenced_by = [f"{t}.{c}" for (t, c, rt, _rc) in schema.relationships() if rt == table.name]
    if referenced_by:
        lines.append("Referenced by: " + ", ".join(referenced_by))
    return "\n".join(lines)


def build_chunks(schema: DatabaseSchema) -> list[SchemaChunk]:
    chunks = []
    for t in schema.tables:
        chunks.append(SchemaChunk(
            table=t.name,
            text=table_to_text(t, schema),
            metadata={
                "table": t.name,
                "type": "table_schema",
                "primary_keys": t.primary_keys,
                "foreign_keys": [fk.model_dump() for fk in t.foreign_keys],
                "columns": t.column_names,
            }))
    return chunks
