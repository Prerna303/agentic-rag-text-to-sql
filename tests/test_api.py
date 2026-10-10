from fastapi.testclient import TestClient

from backend.main import app

c = TestClient(app)


def test_health():
    j = c.get("/api/health").json()
    assert j["status"] == "ok" and "api_key" not in str(j).lower().replace("gemini_key_configured", "")


def test_schema_endpoint():
    j = c.get("/api/schema").json()
    assert len(j["tables"]) == 6 and j["relationships"]


def test_query_endpoint():
    j = c.post("/api/query", json={"question": "How many customers are from Delhi?"}).json()
    assert j["execution_success"] and j["result"]["rows"][0][0] >= 1
    assert j["debug"]["trace"] and "retrieved_tables" in j and j["repair_attempts"] == 0


def test_query_validation_error():
    assert c.post("/api/query", json={"question": "x"}).status_code == 422
    assert c.post("/api/query", json={"question": "hello there", "mode": "zzz"}).status_code == 422


def test_validate_sql_endpoint():
    assert c.post("/api/validate-sql", json={"sql": "DROP TABLE customers"}).json()["valid"] is False
    assert c.post("/api/validate-sql", json={"sql": "SELECT name FROM customers"}).json()["valid"] is True


def test_ingest_endpoint():
    j = c.post("/api/ingest").json()
    assert j["status"] == "ok" and j["tables"] == 6


def test_frontend_served():
    assert "Agentic RAG" in c.get("/").text
