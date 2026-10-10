import json
from pathlib import Path

from backend.llm.mock import MockLLM
from evaluation.baseline import run_mode
from evaluation.metrics import normalize_sql, recall_precision_at_k, results_match
from evaluation.run_evaluation import summarize

DATA = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "dataset.json").read_text())


def test_dataset_shape():
    assert len(DATA) == 150
    for d in ("simple", "aggregation", "join", "complex", "multi-step"):
        assert sum(1 for x in DATA if x["difficulty"] == d) == 30
    assert all({"question", "expected_sql", "expected_tables", "difficulty"} <= set(x) for x in DATA)
    assert sum(1 for x in DATA if x["expected_sql"] is None) == 10


def test_every_gold_sql_executes(ctx):
    from backend.database.executor import execute_sql
    for x in DATA:
        if x["expected_sql"]:
            assert execute_sql(x["expected_sql"], ctx.engine, 100000).success, x["question"]


def test_metric_functions():
    assert normalize_sql("SELECT  a FROM   t") == normalize_sql("select a\nfrom t;")
    assert results_match([[1, 2.00001]], [[1, 2.00002]]) and not results_match([[1]], [[2]])
    assert recall_precision_at_k(["a", "b", "c"], ["a", "c"], 3) == (1.0, 2 / 3)
    assert recall_precision_at_k(["b", "a"], ["a"], 1) == (0.0, 0.0)


def test_pipeline_end_to_end_small(ctx):
    items = [x for x in DATA if x["id"] in (1, 61, 91)] + [x for x in DATA if x["expected_sql"] is None][:1]
    rows_a = run_mode(items, "agentic", ctx, MockLLM(), progress=False)
    rows_b = run_mode(items, "basic", ctx, MockLLM(), progress=False)
    sa, sb = summarize(rows_a), summarize(rows_b)
    assert sa["n_questions"] == 4 and sa["execution_accuracy_pct"] is not None
    assert sb["repair_success_pct"] is None      # nothing needed repair -> N/A, not a fake number
    neg = [r for r in rows_a if r["expected_behavior"] == "unavailable"][0]
    assert neg["execution_accuracy"] is True
