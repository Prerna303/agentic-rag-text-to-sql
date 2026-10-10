"""
FILE PURPOSE: choose the provider from settings (mock > grok > gemini).
INPUT: Settings.  OUTPUT: LLMProvider.  CALLED BY: agents/orchestrator.py, main.py.
CALLS: llm/gemini.py, llm/grok.py, llm/mock.py.
EXAMPLE: get_llm() -> GeminiProvider (default) ; MOCK_LLM=true -> MockLLM
"""
from __future__ import annotations

from backend.config import Settings, get_settings
from backend.llm.base import LLMError, LLMProvider


def get_llm(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.mock_llm or s.llm_provider == "mock":
        from backend.llm.mock import MockLLM
        return MockLLM(fault_rate=s.mock_fault_rate)
    if s.llm_provider == "gemini":
        from backend.llm.gemini import GeminiProvider
        return GeminiProvider(s.gemini_api_key, s.gemini_model, s.llm_timeout_seconds)
    if s.llm_provider == "grok":
        from backend.llm.grok import GrokProvider
        return GrokProvider(s.grok_api_key, s.grok_model, s.llm_timeout_seconds)
    raise LLMError(f"Unknown LLM_PROVIDER '{s.llm_provider}'. Use gemini, grok or mock.")
