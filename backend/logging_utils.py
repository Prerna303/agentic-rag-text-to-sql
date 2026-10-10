"""
FILE PURPOSE
------------
Structured (JSON-lines) logging with automatic secret redaction.

WHY: Every pipeline stage (question, retrieval, plan, SQL, validation, repair,
execution) must be observable, but API keys must never reach a log file.

INPUT  : event name + keyword fields.   OUTPUT: one JSON line on the logger.
CALLED BY: agents/orchestrator.py, main.py.  CALLS: stdlib logging only.
EXAMPLE: log_event("retrieval", tables=["vehicles"]) ->
         {"event": "retrieval", "tables": ["vehicles"]}
"""
from __future__ import annotations

import json
import logging
import re

_SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|xai-[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{20,})")

logger = logging.getLogger("text2sql")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def redact(text: str) -> str:
    """Replace anything that looks like an API key."""
    return _SECRET.sub("[REDACTED]", text)


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    payload = {"event": event, **fields}
    logger.log(level, redact(json.dumps(payload, default=str)))
