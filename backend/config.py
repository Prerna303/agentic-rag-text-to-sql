"""
FILE PURPOSE
------------
Central configuration. Reads every setting from environment variables (or a
local .env file) exactly once and exposes them as a frozen `Settings` object.

WHY THIS FILE EXISTS
--------------------
No other module should call os.getenv(). Keeping all settings in one place
means (a) API keys are never hard-coded, (b) tests can override settings, and
(c) you can explain the full configuration surface in a viva by reading one file.

WHAT HAPPENS INSIDE
-------------------
1. python-dotenv loads `.env` if it exists (silently skipped otherwise).
2. `Settings.from_env()` reads each variable and converts it to the right type.
3. `get_settings()` returns a cached Settings instance.

INPUT   : environment variables (see .env.example).
OUTPUT  : `Settings` dataclass.
CALLED BY: almost every module (llm/factory.py, rag/*, agents/*, main.py, evaluation/*).
CALLS   : nothing in this project.
EXAMPLE : `get_settings().top_k` -> 5
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

try:  # optional dependency at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    grok_api_key: str = ""
    grok_model: str = "grok-3-mini"
    llm_timeout_seconds: int = 60
    mock_llm: bool = False
    mock_fault_rate: float = 0.0
    embedding_provider: str = "local"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 384
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'database.db'}"
    vector_store_dir: str = str(PROJECT_ROOT / "vector_store")
    top_k: int = 5
    max_repair_attempts: int = 3
    max_result_rows: int = 500

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        db_url = e("DATABASE_URL") or cls.database_url
        # Make relative sqlite paths independent of the current working directory.
        if db_url.startswith("sqlite:///./"):
            db_url = "sqlite:///" + str(PROJECT_ROOT / db_url[len("sqlite:///./"):])
        vs = e("VECTOR_STORE_DIR") or cls.vector_store_dir
        if vs.startswith("./"):
            vs = str(PROJECT_ROOT / vs[2:])
        return cls(
            llm_provider=(e("LLM_PROVIDER") or "gemini").lower(),
            gemini_api_key=e("GEMINI_API_KEY", ""),
            gemini_model=e("GEMINI_MODEL") or "gemini-2.5-flash",
            grok_api_key=e("GROK_API_KEY", ""),
            grok_model=e("GROK_MODEL") or "grok-3-mini",
            llm_timeout_seconds=int(e("LLM_TIMEOUT_SECONDS") or 60),
            mock_llm=_bool(e("MOCK_LLM")),
            mock_fault_rate=float(e("MOCK_FAULT_RATE") or 0.0),
            embedding_provider=(e("EMBEDDING_PROVIDER") or "local").lower(),
            embedding_model=e("EMBEDDING_MODEL") or "gemini-embedding-001",
            embedding_dim=int(e("EMBEDDING_DIM") or 384),
            database_url=db_url,
            vector_store_dir=vs,
            top_k=int(e("TOP_K") or 5),
            max_repair_attempts=int(e("MAX_REPAIR_ATTEMPTS") or 3),
            max_result_rows=int(e("MAX_RESULT_ROWS") or 500),
        )

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
