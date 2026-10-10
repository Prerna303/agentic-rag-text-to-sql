def test_all_tables_discovered(schema):
    assert set(schema.table_names) == {"customers", "branches", "vehicles", "rental", "rating", "maintenance"}


def test_primary_and_foreign_keys(schema):
    assert schema.table("customers").primary_keys == ["customer_id"]
    fks = {(f.column, f.ref_table, f.ref_column) for f in schema.table("rental").foreign_keys}
    assert fks == {("customer_id", "customers", "customer_id"), ("vehicle_id", "vehicles", "vehicle_id")}
    assert ("rating", "rental_id", "rental", "rental_id") in schema.relationships()


def test_column_types_and_descriptions(schema):
    col = {c.name: c for c in schema.table("vehicles").columns}
    assert "REAL" in col["price_per_day"].data_type.upper()
    assert schema.table("vehicles").description


def test_sample_values_for_low_cardinality_text(schema):
    city = {c.name: c for c in schema.table("customers").columns}["city"]
    assert "Delhi" in city.sample_values
