"""
Unified LLM client supporting OpenAI-compatible APIs.
Works with vLLM, Ollama, OpenAI, Anthropic-compatible endpoints, etc.
"""

import os
import logging
import time
from dataclasses import dataclass

import requests

from config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    finish_reason: str = "stop"
    usage: dict | None = None


class LLMClient:
    """OpenAI-compatible chat completions client."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = os.getenv("LLM_API_KEY", config.api_key)
        self.base_url = os.getenv("LLM_API_BASE", config.api_base).rstrip("/")

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[LLMResponse]:
        """Send a chat completion request, return n responses."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "max_tokens": max_tokens or self.config.max_tokens,
            "n": n,
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.HTTPError:
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error("LLM request failed: %s", e)
                    raise
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("Max retries exceeded due to rate limiting")

        responses = []
        for choice in data.get("choices", []):
            responses.append(LLMResponse(
                text=choice.get("message", {}).get("content", ""),
                finish_reason=choice.get("finish_reason", "stop"),
                usage=data.get("usage"),
            ))
        return responses

    def generate_candidates(self, messages: list[dict], k: int | None = None) -> list[LLMResponse]:
        """Generate K candidate responses using nucleus sampling."""
        k = k or self.config.num_candidates
        # Many APIs (Gemini, etc.) don't support n>1, so always do sequential
        results = []
        for i in range(k):
            if i > 0:
                time.sleep(1)  # rate limit spacing
            try:
                results.extend(self.chat(messages, n=1))
            except Exception as e:
                logger.warning("Candidate %d/%d failed: %s", i + 1, k, e)
        return results
