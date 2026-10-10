import pytest

from backend.agents.validator_agent import validate_sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers", "DELETE FROM rental", "UPDATE customers SET city='x'",
    "INSERT INTO branches VALUES (9,'a','b',1)", "ALTER TABLE customers ADD x INT",
    "CREATE TABLE t(x INT)", "SELECT 1; DROP TABLE customers", "PRAGMA table_info(customers)",
])
def test_dangerous_sql_blocked(sql, schema, ctx):
    r = validate_sql(sql, schema, ctx.engine)
    assert not r.valid and r.level == "safety"


def test_keyword_inside_string_literal_is_fine(schema, ctx):
    assert validate_sql("SELECT name FROM customers WHERE city = 'DROP'", schema, ctx.engine).valid


def test_syntax_error(schema, ctx):
    r = validate_sql("SELEC name FROM customers", schema, ctx.engine)
    assert not r.valid and r.level == "syntax"


def test_unknown_table_and_column(schema, ctx):
    r = validate_sql("SELECT * FROM engines", schema, ctx.engine)
    assert r.hallucinated_tables == ["engines"] and r.level == "schema"
    r = validate_sql("SELECT vehicles.id FROM vehicles", schema, ctx.engine)
    assert r.hallucinated_columns == ["vehicles.id"]
    r = validate_sql("SELECT horsepower FROM vehicles", schema, ctx.engine)
    assert not r.valid


def test_valid_complex_queries(schema, ctx):
    ok = ["SELECT v.model, AVG(g.rating) AS a FROM vehicles v JOIN rental r ON v.vehicle_id=r.vehicle_id "
          "JOIN rating g ON g.rental_id=r.rental_id GROUP BY v.vehicle_id ORDER BY a DESC LIMIT 5",
          "WITH s AS (SELECT customer_id, SUM(amount) t FROM rental GROUP BY customer_id) "
          "SELECT c.name FROM customers c JOIN s ON s.customer_id=c.customer_id WHERE s.t > (SELECT AVG(t) FROM s)"]
    for sql in ok:
        assert validate_sql(sql, schema, ctx.engine).valid


def test_database_level_explain(schema, ctx):
    # passes schema check (all columns exist) but is invalid for the engine: aggregate misuse
    r = validate_sql("SELECT customer_id FROM rental WHERE SUM(amount) > 5", schema, ctx.engine)
    assert not r.valid and r.level == "database"
