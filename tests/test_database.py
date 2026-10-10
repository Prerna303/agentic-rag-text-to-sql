from backend.database.executor import execute_sql


def test_row_counts(ctx):
    assert execute_sql("SELECT COUNT(*) FROM customers", ctx.engine).rows[0][0] == 100
    assert execute_sql("SELECT COUNT(*) FROM vehicles", ctx.engine).rows[0][0] == 50
    assert execute_sql("SELECT COUNT(*) FROM rental", ctx.engine).rows[0][0] == 500
    assert execute_sql("SELECT COUNT(*) FROM rating", ctx.engine).rows[0][0] == 300


def test_database_is_read_only(ctx):
    r = execute_sql("DELETE FROM customers", ctx.engine)
    assert not r.success
    assert execute_sql("SELECT COUNT(*) FROM customers", ctx.engine).rows[0][0] == 100


def test_row_cap(ctx):
    r = execute_sql("SELECT * FROM rental", ctx.engine, max_rows=10)
    assert len(r.rows) == 10 and r.row_count == 500 and r.truncated


def test_sql_error_is_reported(ctx):
    r = execute_sql("SELECT nope FROM customers", ctx.engine)
    assert not r.success and "no such column" in r.error
