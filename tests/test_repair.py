from backend.agents.orchestrator import run_agent
from backend.agents.repair_agent import analyze_errors
from backend.llm.mock import MockLLM

Q = "Find the top 5 vehicles by average rating that have received at least 2 ratings."


def test_error_analysis_suggests_real_column(schema):
    hint = analyze_errors(["Column 'customers.id' does not exist. Columns of customers: customer_id, name"], schema)
    assert "customer_id" in hint


def test_repair_loop_fixes_broken_sql(ctx):
    st = run_agent(Q, ctx=ctx, llm=MockLLM(fault_rate=1.0))
    assert st.status == "success" and st.repair_attempts == 1
    assert st.first_validation and not st.first_validation.valid
    assert st.execution_result.row_count == 5
    assert any("Repairing" in t.name for t in st.trace)


def test_basic_rag_does_not_repair(ctx):
    st = run_agent(Q, mode="basic", ctx=ctx, llm=MockLLM(fault_rate=1.0))
    assert st.status == "failed" and st.repair_attempts == 0


def test_max_retries_prevents_infinite_loop(ctx):
    class AlwaysBroken(MockLLM):
        def generate(self, prompt):
            if "TASK: sql_generator" in prompt or "TASK: repair" in prompt:
                return "SELECT bogus FROM customers"
            return super().generate(prompt)
    st = run_agent("How many customers are there?", ctx=ctx, llm=AlwaysBroken())
    assert st.status == "failed" and st.repair_attempts == ctx.settings.max_repair_attempts
    assert "Maximum repair attempts" in st.error


def test_hallucination_case_returns_unavailable(ctx):
    st = run_agent("Find vehicles based on engine horsepower.", ctx=ctx)
    assert st.status == "unavailable" and "not available" in st.final_answer
