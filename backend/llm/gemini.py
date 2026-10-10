"""
FILE PURPOSE
------------
Gemini (Google) implementation of LLMProvider using the `google-genai` SDK
(`from google import genai`; `client.models.generate_content(...)`).

WHY: Gemini Flash is the project's primary, cheap and fast LLM.

INPUT  : prompt string.   OUTPUT: response text.
CALLED BY: llm/factory.py.  CALLS: google-genai SDK (network).
NOTE   : model name comes from GEMINI_MODEL so you can update it without
         touching code. Temperature is 0 for reproducible SQL.
"""
from __future__ import annotations

from backend.llm.base import LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env or use MOCK_LLM=true.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise LLMError("google-genai is not installed: pip install google-genai") from exc
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout_seconds

    def generate(self, prompt: str) -> str:
        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(temperature=0.0),
            )
            text = (resp.text or "").strip()
        except Exception as exc:
            msg = str(exc)
            if "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
                raise LLMError("Gemini rejected the API key (invalid or lacks access).") from None
            raise LLMError(f"Gemini request failed: {type(exc).__name__}") from None
        if not text:
            raise LLMError("Gemini returned an empty response.")
        return text
