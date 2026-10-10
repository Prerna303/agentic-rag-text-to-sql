"""
FILE PURPOSE: pytest setup. Runs everything OFFLINE: MOCK_LLM=true, a throw-away
SQLite database and vector store in a temp folder (your real data/ is untouched).
No API key or internet is needed for `pytest`.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="t2sql_"))
os.environ.update({
    "MOCK_LLM": "true", "MOCK_FAULT_RATE": "0", "LLM_PROVIDER": "mock", "EMBEDDING_PROVIDER": "local",
    "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}", "VECTOR_STORE_DIR": str(_TMP / "vs"),
    "GEMINI_API_KEY": "", "TOP_K": "5", "MAX_REPAIR_ATTEMPTS": "3",
})

import pytest  # noqa: E402

from backend.config import reset_settings_cache  # noqa: E402
from backend.database.models import create_demo_database  # noqa: E402

create_demo_database(_TMP / "test.db")
reset_settings_cache()

from backend.rag.ingestion import ingest_schema  # noqa: E402
from backend.tools.schema_tool import build_context, reset_context  # noqa: E402

ingest_schema()
reset_context()


@pytest.fixture(scope="session")
def ctx():
    return build_context()


@pytest.fixture(scope="session")
def schema(ctx):
    return ctx.schema
