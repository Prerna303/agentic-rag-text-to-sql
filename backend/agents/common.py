"""
FILE PURPOSE: small helpers shared by the agents: prompt template loading and
cleaning of LLM output (markdown fences, JSON extraction, UNAVAILABLE marker).
INPUT: template name + values / raw LLM text.  OUTPUT: prompt string / cleaned SQL / dict.
CALLED BY: planner_agent, sql_generator, repair_agent, explanation_agent.
EXAMPLE: clean_sql("```sql\nSELECT 1;\n```") -> "SELECT 1"
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def render_prompt(name: str, **values: str) -> str:
    text = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def clean_sql(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.S | re.I)
    if m:
        raw = m.group(1).strip()
    return raw.strip().rstrip(";").strip()


def parse_unavailable(raw: str) -> str | None:
    m = re.match(r"\s*UNAVAILABLE\s*:?\s*(.*)", raw.strip(), re.S | re.I)
    return (m.group(1).strip() or "The requested information is not available in the database schema.") if m else None


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    if m:
        raw = m.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(raw[start:end + 1])
