from backend.agents.sql_generator import generate_sql, generate_sql_basic
from backend.llm.mock import MockLLM
from backend.agents.common import clean_sql, parse_unavailable


def test_generates_select():
    sql, reason = generate_sql(MockLLM(), "How many customers are from Delhi?", "", None)
    assert reason is None and sql.upper().startswith("SELECT COUNT")


def test_unavailable_marker():
    sql, reason = generate_sql_basic(MockLLM(), "Find vehicles based on engine horsepower.", "")
    assert sql == "" and reason


def test_clean_sql_and_marker():
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert parse_unavailable("UNAVAILABLE: no such data") == "no such data"
    assert parse_unavailable("SELECT 1") is None
