from backend.rag.chunking import build_chunks


def test_one_chunk_per_table(schema):
    chunks = build_chunks(schema)
    assert [c.table for c in chunks] == schema.table_names


def test_chunk_contains_relationships(schema):
    by = {c.table: c for c in build_chunks(schema)}
    assert "customer_id -> customers.customer_id" in by["rental"].text
    assert "Referenced by:" in by["rental"].text and "rating.rental_id" in by["rental"].text
    assert "Primary Key: vehicle_id" in by["vehicles"].text


def test_metadata(schema):
    m = {c.table: c.metadata for c in build_chunks(schema)}["vehicles"]
    assert m["type"] == "table_schema" and m["primary_keys"] == ["vehicle_id"]
    assert m["foreign_keys"][0]["ref_table"] == "branches"
