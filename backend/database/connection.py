"""
FILE PURPOSE
------------
Creates the SQLAlchemy engine used everywhere. For SQLite it turns on
`PRAGMA query_only=ON` on every connection so that even if a bad statement
slipped past the validator, the database itself refuses to write.

WHY: defence in depth for the read-only guarantee.
INPUT: database URL.  OUTPUT: sqlalchemy Engine.
CALLED BY: schema_inspector.py, executor.py, models.py (seeding uses a writable engine).
EXAMPLE: get_engine().connect()
"""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from backend.config import get_settings


def make_engine(url: str | None = None, read_only: bool = True) -> Engine:
    url = url or get_settings().database_url
    engine = create_engine(url, future=True)
    if url.startswith("sqlite") and read_only:
        @event.listens_for(engine, "connect")
        def _ro(dbapi_conn, _rec):  # noqa: ANN001
            dbapi_conn.execute("PRAGMA query_only = ON")
    return engine


@lru_cache(maxsize=4)
def _cached(url: str) -> Engine:
    return make_engine(url, read_only=True)


def get_engine(url: str | None = None) -> Engine:
    """Cached READ-ONLY engine."""
    return _cached(url or get_settings().database_url)
