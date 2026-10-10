from backend.agents.planner_agent import plan_query
from backend.agents.common import extract_json
from backend.llm.mock import MockLLM


def test_plan_structure():
    p = plan_query(MockLLM(), "Find the top 5 vehicles by average rating that have received at least 2 ratings.", "")
    assert p.feasible and set(["vehicles", "rental", "rating"]) <= set(p.tables)
    assert p.limit == 5 and p.group_by and p.aggregations


def test_plan_marks_unavailable():
    p = plan_query(MockLLM(), "Find vehicles based on engine horsepower.", "")
    assert not p.feasible and p.missing


def test_bad_json_falls_back():
    class Bad:
        def generate(self, prompt):
            return "not json"
    assert plan_query(Bad(), "x", "").feasible


def test_extract_json_handles_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
