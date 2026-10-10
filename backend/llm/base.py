"""
FILE PURPOSE
------------
Defines the single interface every LLM provider must implement, plus the
project-wide `LLMError`.

WHY THIS FILE EXISTS
--------------------
The agents (planner, SQL generator, repair, explanation) must not know whether
Gemini, Grok or a mock is behind them. They call `provider.generate(prompt)`.
Switching provider = changing LLM_PROVIDER in .env. This is the classic
"strategy pattern".

INPUT  : a prompt string.        OUTPUT: the model's text response.
CALLED BY: agents/*.py (through llm/factory.py).
CALLS  : nothing.
EXAMPLE: provider.generate("TASK: planner\nQUESTION: ...") -> '{"tables": [...]}'
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Raised for invalid key, timeout, network failure or empty response.
    Messages never contain API keys."""


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's text answer for `prompt`."""
