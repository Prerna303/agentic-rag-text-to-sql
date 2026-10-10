"""
FILE PURPOSE
------------
An OFFLINE, RULE-BASED stand-in for a real LLM (`MOCK_LLM=true`). It is NOT an AI:
it recognises ~20 question patterns with regular expressions and emits SQL for them.

WHY IT EXISTS
-------------
* `pytest` must run without an API key or internet.
* It lets you demo/debug the whole pipeline (retrieval, planning, validation,
  repair loop, execution, evaluation, charts) for free.

HONESTY NOTE
------------
Numbers produced with this mock measure the PIPELINE PLUMBING and the coverage of
these hand-written rules. They say NOTHING about Gemini/Grok accuracy. The
evaluation report labels every run as "mock" or "real LLM".

FAULT INJECTION
---------------
`MOCK_FAULT_RATE` (0..1) deterministically breaks that fraction of FIRST-attempt SQL
(e.g. `customer_id` -> `id`), to exercise the validator/repair loop. The repair task
always answers with the un-broken SQL.

INPUT  : prompt (first line `TASK: planner|sql_generator|basic_sql|repair|explanation`,
         a line `QUESTION: ...`).
OUTPUT : text in the format that task expects (JSON plan / SQL / UNAVAILABLE / prose).
CALLED BY: llm/factory.py.   CALLS: sqlglot (to derive the plan from the SQL).
EXAMPLE: MockLLM().generate("TASK: basic_sql\nQUESTION: Show all customers") -> "SELECT customer_id, ..."
"""
from __future__ import annotations

import hashlib
import json
import re

import sqlglot
from sqlglot import exp

from backend.llm.base import LLMProvider

TABLE_COLS = {
    "customers": "customer_id, name, email, city, registration_date",
    "branches": "branch_id, branch_name, city, opened_year",
    "vehicles": "vehicle_id, model, manufacturer, category, price_per_day, branch_id",
    "rental": "rental_id, customer_id, vehicle_id, rental_date, return_date, amount",
    "rating": "rating_id, rental_id, rating, review_text",
    "maintenance": "maintenance_id, vehicle_id, service_date, cost, description",
}
PLURAL = {"customers": "customers", "vehicles": "vehicles", "branches": "branches", "rentals": "rental",
          "ratings": "rating", "maintenance records": "maintenance", "maintenance": "maintenance"}
CATS = {"suv": "SUV", "sedan": "Sedan", "hatchback": "Hatchback", "luxury": "Luxury", "van": "Van"}
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Jaipur"]
UNAVAILABLE_WORDS = re.compile(
    r"\b(horsepower|engine power|engine size|salary|colou?r|fuel|mileage|age|gender|phone|address|"
    r"licen[sc]e|credit score|loyalty|tax|discount|insurance|efficiency|weight)\b")
AVG_RATING_JOIN = ("FROM vehicles v JOIN rental r ON v.vehicle_id = r.vehicle_id "
                   "JOIN rating g ON r.rental_id = g.rental_id")


def _cat(q: str) -> str | None:
    for k, v in CATS.items():
        if re.search(rf"\b{k}s?\b", q):
            return v
    return None


def _city(q: str) -> str | None:
    for c in CITIES:
        if c.lower() in q:
            return c
    return None


def rule_sql(question: str) -> str | None:
    """Return SQL for a recognised pattern, or None (= 'not available / not understood')."""
    q = re.sub(r"[?.!]+$", "", question.strip().lower())
    if UNAVAILABLE_WORDS.search(q):
        return None
    m = re.match(r"^(?:show|list|display) all (customers|vehicles|branches|rentals|ratings|maintenance records)$", q)
    if m:
        t = PLURAL[m.group(1)]
        return f"SELECT {TABLE_COLS[t]} FROM {t}"
    m = re.match(r"^show all vehicles in the (\w+) category$", q)
    if m and _cat(m.group(1)):
        return f"SELECT {TABLE_COLS['vehicles']} FROM vehicles WHERE category = '{_cat(m.group(1))}'"
    m = re.match(r"^list customers from (\w+)$", q)
    if m and _city(m.group(1)):
        return f"SELECT name FROM customers WHERE city = '{_city(m.group(1))}'"
    m = re.match(r"^how many (customers|vehicles|branches|rentals|ratings|maintenance records) are there$", q)
    if m:
        return f"SELECT COUNT(*) AS total FROM {PLURAL[m.group(1)]}"
    m = re.match(r"^how many customers are from (\w+)$", q)
    if m and _city(m.group(1)):
        return f"SELECT COUNT(*) AS total FROM customers WHERE city = '{_city(m.group(1))}'"
    if q == "what is the average price per day of vehicles":
        return "SELECT AVG(price_per_day) AS avg_price FROM vehicles"
    if q == "what is the total rental revenue":
        return "SELECT SUM(amount) AS total_revenue FROM rental"
    if q == "what is the total maintenance cost":
        return "SELECT SUM(cost) AS total_cost FROM maintenance"
    if q == "what is the overall average rating":
        return "SELECT AVG(rating) AS avg_rating FROM rating"
    m = re.match(r"^what is the average price per day for (\w+) vehicles$", q)
    if m and _cat(m.group(1)):
        return f"SELECT AVG(price_per_day) AS avg_price FROM vehicles WHERE category = '{_cat(m.group(1))}'"
    if q == "find the average rating of each vehicle":
        return f"SELECT v.vehicle_id, v.model, AVG(g.rating) AS avg_rating {AVG_RATING_JOIN} GROUP BY v.vehicle_id, v.model"
    m = re.match(r"^find the names of customers who rented (\w+) vehicles$", q)
    if m and _cat(m.group(1)):
        return ("SELECT DISTINCT c.name FROM customers c JOIN rental r ON c.customer_id = r.customer_id "
                f"JOIN vehicles v ON r.vehicle_id = v.vehicle_id WHERE v.category = '{_cat(m.group(1))}'")
    m = re.match(r"^how many rentals were made for (\w+) vehicles$", q)
    if m and _cat(m.group(1)):
        return ("SELECT COUNT(*) AS total FROM rental r JOIN vehicles v ON r.vehicle_id = v.vehicle_id "
                f"WHERE v.category = '{_cat(m.group(1))}'")
    if q == "find the total rental amount for each vehicle category":
        return ("SELECT v.category, SUM(r.amount) AS total_amount FROM vehicles v "
                "JOIN rental r ON v.vehicle_id = r.vehicle_id GROUP BY v.category")
    if q == "how many vehicles does each branch have":
        return ("SELECT b.branch_name, COUNT(v.vehicle_id) AS vehicle_count FROM branches b "
                "LEFT JOIN vehicles v ON b.branch_id = v.branch_id GROUP BY b.branch_id, b.branch_name")
    m = re.match(r"^find the top (\d+) vehicles by average rating(?: that (?:have )?received at least (\d+) ratings?)?$", q)
    if m:
        having = f" HAVING COUNT(g.rating_id) >= {m.group(2)}" if m.group(2) else ""
        return (f"SELECT v.vehicle_id, v.model, AVG(g.rating) AS avg_rating {AVG_RATING_JOIN} "
                f"GROUP BY v.vehicle_id, v.model{having} ORDER BY avg_rating DESC LIMIT {m.group(1)}")
    m = re.match(r"^find the top (\d+) customers by total spending$", q)
    if m:
        return ("SELECT c.customer_id, c.name, SUM(r.amount) AS total_spent FROM customers c "
                "JOIN rental r ON c.customer_id = r.customer_id GROUP BY c.customer_id, c.name "
                f"ORDER BY total_spent DESC LIMIT {m.group(1)}")
    if q == "find customers whose total rental spending is greater than the average customer spending":
        return ("WITH spend AS (SELECT customer_id, SUM(amount) AS total FROM rental GROUP BY customer_id) "
                "SELECT c.customer_id, c.name, s.total FROM customers c JOIN spend s ON c.customer_id = s.customer_id "
                "WHERE s.total > (SELECT AVG(total) FROM spend)")
    if q == "find vehicles that were never rented":
        return ("SELECT vehicle_id, model FROM vehicles WHERE vehicle_id NOT IN "
                "(SELECT vehicle_id FROM rental)")
    return None


def _plan_from_sql(sql: str) -> dict:
    tree = sqlglot.parse_one(sql, read="sqlite")
    plan = {"feasible": True, "missing": [], "tables": [], "joins": [], "filters": [],
            "group_by": [], "aggregations": [], "order_by": [], "limit": None}
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    plan["tables"] = list(dict.fromkeys(t.name for t in tree.find_all(exp.Table) if t.name.lower() not in ctes))
    plan["joins"] = [j.args["on"].sql() for j in tree.find_all(exp.Join) if j.args.get("on")]
    where = tree.find(exp.Where)
    if where:
        plan["filters"] = [where.this.sql()]
    grp = tree.find(exp.Group)
    if grp:
        plan["group_by"] = [e.sql() for e in grp.expressions]
    plan["aggregations"] = list(dict.fromkeys(a.sql() for a in tree.find_all(exp.AggFunc)))
    order = tree.find(exp.Order)
    if order:
        plan["order_by"] = [e.sql() for e in order.expressions]
    lim = tree.find(exp.Limit)
    if lim:
        try:
            plan["limit"] = int(lim.expression.name)
        except Exception:
            pass
    return plan


def _break(sql: str) -> str:
    """Introduce a typical LLM schema mistake (used only for fault injection)."""
    for col in ("customer_id", "vehicle_id", "rental_id", "branch_id"):
        if re.search(rf"\b{col}\b", sql):
            return re.sub(rf"\b{col}\b", "id", sql, count=1)
    return sql.replace("price_per_day", "price", 1) if "price_per_day" in sql else sql + " WHERE nonexistent_col = 1"


class MockLLM(LLMProvider):
    name = "mock"

    def __init__(self, fault_rate: float = 0.0) -> None:
        self.fault_rate = fault_rate

    def _should_fault(self, question: str) -> bool:
        if self.fault_rate <= 0:
            return False
        h = int(hashlib.md5(question.encode()).hexdigest()[:8], 16) % 1000 / 1000
        return h < self.fault_rate

    def generate(self, prompt: str) -> str:
        task = re.match(r"\s*TASK:\s*(\w+)", prompt)
        task = task.group(1) if task else "unknown"
        m = re.search(r"^QUESTION:\s*(.+)$", prompt, re.M)
        question = m.group(1).strip() if m else ""
        sql = rule_sql(question)
        if task == "planner":
            if sql is None:
                return json.dumps({"feasible": False, "missing": ["information not present in the schema"],
                                   "tables": [], "joins": [], "filters": [], "group_by": [],
                                   "aggregations": [], "order_by": [], "limit": None})
            return json.dumps(_plan_from_sql(sql))
        if task in ("sql_generator", "basic_sql"):
            if sql is None:
                return "UNAVAILABLE: the question needs information that is not in the schema (or is not understood by the mock rules)"
            return _break(sql) if self._should_fault(question) else sql
        if task == "repair":
            return sql if sql else "UNAVAILABLE: no valid table/column exists for the requested information"
        if task == "explanation":
            n = re.search(r"\((?:first rows, )?(\d+) rows total\)", prompt)
            return f"The query answers the question using the tables shown in the SQL and returned {n.group(1) if n else 'some'} row(s). (mock explanation)"
        return "UNAVAILABLE: unknown task"
