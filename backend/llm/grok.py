"""
FILE PURPOSE
------------
Grok (xAI) implementation of LLMProvider. xAI exposes an OpenAI-compatible
REST endpoint, so a plain HTTPS call via httpx is enough (no extra SDK).

WHY: proves the provider layer is swappable (LLM_PROVIDER=grok).
INPUT: prompt.  OUTPUT: text.  CALLED BY: llm/factory.py.  CALLS: httpx (network).
NOTE : verify the current base URL / model name in xAI docs; both are overridable.
"""
from __future__ import annotations

import httpx

from backend.llm.base import LLMError, LLMProvider


class GrokProvider(LLMProvider):
    name = "grok"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60,
                 base_url: str = "https://api.x.ai/v1") -> None:
        if not api_key:
            raise LLMError("GROK_API_KEY is not set.")
        self._key, self._model, self._timeout, self._url = api_key, model, timeout_seconds, base_url

    def generate(self, prompt: str) -> str:
        try:
            r = httpx.post(
                f"{self._url}/chat/completions",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self._model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=self._timeout,
            )
        except httpx.TimeoutException:
            raise LLMError("Grok request timed out.") from None
        except httpx.HTTPError as exc:
            raise LLMError(f"Grok network error: {type(exc).__name__}") from None
        if r.status_code in (401, 403):
            raise LLMError("Grok rejected the API key.")
        if r.status_code >= 400:
            raise LLMError(f"Grok returned HTTP {r.status_code}.")
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            raise LLMError("Grok returned an unexpected response.") from None
