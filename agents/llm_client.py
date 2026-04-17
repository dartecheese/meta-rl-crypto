"""
Unified LLM client supporting OpenAI-compatible APIs.
Works with vLLM, Ollama, OpenAI, Anthropic-compatible endpoints, etc.
"""

import os
import logging
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

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("LLM request failed: %s", e)
            raise

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
        # Some APIs don't support n>1, so fall back to sequential
        try:
            return self.chat(messages, n=k)
        except Exception:
            logger.info("Falling back to sequential candidate generation")
            results = []
            for _ in range(k):
                results.extend(self.chat(messages, n=1))
            return results
