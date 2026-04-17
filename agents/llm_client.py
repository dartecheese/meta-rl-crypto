"""
LLM client using the Google Gemini SDK (google-genai).
"""

import logging
import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import types

from config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    finish_reason: str = "stop"
    usage: dict | None = None


class LLMClient:
    """Google Gemini chat client."""

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = os.getenv("GEMINI_API_KEY", config.api_key)
        self._client = genai.Client(api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[LLMResponse]:
        """Send a chat completion request, return n responses."""
        system = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature if temperature is not None else self.config.temperature,
            top_p=top_p if top_p is not None else self.config.top_p,
            max_output_tokens=max_tokens or self.config.max_tokens,
        )

        results = []
        for i in range(n):
            if i > 0:
                time.sleep(0.5)
            results.append(self._call_with_retry(contents, config))
        return results

    def _call_with_retry(self, contents, config) -> LLMResponse:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )
                usage = None
                if response.usage_metadata:
                    usage = {
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
                finish_reason = "stop"
                if response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason).lower()
                return LLMResponse(text=response.text, finish_reason=finish_reason, usage=usage)
            except Exception as e:
                err_str = str(e).lower()
                # Don't retry auth errors
                if any(x in err_str for x in ("api key", "permission", "unauthorized", "403", "401")):
                    logger.error("Authentication error — check GEMINI_API_KEY: %s", e)
                    raise
                if any(x in err_str for x in ("quota", "rate", "429", "resource exhausted")):
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                    time.sleep(wait)
                elif attempt == max_retries - 1:
                    logger.error("Gemini API error: %s", e)
                    raise
                else:
                    time.sleep(2 ** attempt)
        raise RuntimeError("Max retries exceeded due to rate limiting")

    def generate_candidates(self, messages: list[dict], k: int | None = None) -> list[LLMResponse]:
        """Generate K candidate responses sequentially."""
        k = k or self.config.num_candidates
        results = []
        for i in range(k):
            if i > 0:
                time.sleep(1)
            try:
                results.extend(self.chat(messages, n=1))
            except Exception as e:
                logger.warning("Candidate %d/%d failed: %s", i + 1, k, e)
        return results
