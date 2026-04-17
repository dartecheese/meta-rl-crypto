"""
Actor Agent — generates trading forecasts and position signals.

The Actor processes on-chain metrics, news, and sentiment data to produce
next-day forecasts with normalized position signals α_t ∈ [-1, 1].
"""

import json
import logging
import re
from dataclasses import dataclass

import numpy as np

from config import LLMConfig
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

ACTOR_SYSTEM_PROMPT = """You are a quantitative crypto trading agent. Your role is to analyze market data, on-chain metrics, and news sentiment to generate trading signals.

You MUST respond with valid JSON in the following format:
{
  "forecasts": {
    "BTC/USDT": {
      "direction": "long" | "short" | "neutral",
      "alpha": <float between -1.0 and 1.0>,
      "confidence": <float between 0.0 and 1.0>,
      "price_target_pct": <expected % move>,
      "rationale": "<brief reasoning>"
    },
    "ETH/USDT": { ... },
    "SOL/USDT": { ... }
  },
  "market_regime": "bullish" | "bearish" | "sideways",
  "risk_assessment": "<brief risk summary>",
  "reasoning": "<detailed analysis>"
}

Rules:
- alpha > 0 means go long (buy), alpha < 0 means go short (sell/reduce), alpha = 0 means hold
- The magnitude of alpha represents conviction: 1.0 = max position, 0.0 = no position
- Consider transaction costs (10 bps) — small alpha signals may not be worth trading
- Factor in liquidity — SOL has higher slippage than BTC/ETH
- Account for current portfolio positions when sizing
- Be explicit about your reasoning chain
"""


def _build_actor_prompt(market_data: dict, onchain_data: dict, news_digest: dict, portfolio_state: dict) -> str:
    """Build the structured input prompt for the Actor."""
    sections = []

    # Market data
    sections.append("## Current Market Data")
    for sym, data in market_data.items():
        sections.append(f"### {sym}")
        sections.append(f"- Close: ${data.get('close', 'N/A'):,.2f}")
        sections.append(f"- 24h Volume: ${data.get('volume', 0):,.0f}")
        sections.append(f"- 1d Return: {data.get('returns_1d', 0)*100:.2f}%")
        sections.append(f"- 7d Return: {data.get('returns_7d', 0)*100:.2f}%")
        sections.append(f"- 30d Volatility: {data.get('volatility_30d', 0)*100:.2f}%")
        if "market_cap" in data:
            sections.append(f"- Market Cap: ${data['market_cap']:,.0f}")

    # On-chain metrics
    sections.append("\n## On-Chain Metrics")
    for sym, metrics in onchain_data.items():
        sections.append(f"### {sym} ({metrics.get('chain', '')})")
        sections.append(f"- 24h Transactions: {metrics.get('tx_count_24h', 'N/A'):,}")
        sections.append(f"- Active Wallets: {metrics.get('active_wallets_24h', 'N/A'):,}")
        sections.append(f"- Value Transferred: ${metrics.get('total_value_transferred_usd', 0):,.0f}")
        if metrics.get("mean_gas_gwei"):
            sections.append(f"- Mean Gas: {metrics['mean_gas_gwei']:.1f} Gwei")

    # News and sentiment
    sections.append("\n## News Digest")
    sections.append(news_digest.get("news_summary", "No news available."))
    sent = news_digest.get("sentiment_vector", [0]*5)
    labels = ["Overall", "BTC", "ETH", "SOL", "News Volume"]
    sections.append("\n## Sentiment Scores (-1 bearish to +1 bullish)")
    for label, val in zip(labels, sent):
        sections.append(f"- {label}: {val:.2f}")

    # Portfolio state
    sections.append("\n## Current Portfolio")
    sections.append(f"- Total Value: ${portfolio_state.get('total_value', 0):,.2f}")
    sections.append(f"- Cash: ${portfolio_state.get('cash', 0):,.2f}")
    positions = portfolio_state.get("positions", {})
    for sym, pos in positions.items():
        sections.append(f"- {sym}: {pos.get('quantity', 0):.6f} units (${pos.get('value', 0):,.2f})")

    return "\n".join(sections)


@dataclass
class ActorForecast:
    """Parsed forecast from the Actor."""
    raw_text: str
    alphas: dict[str, float]  # symbol -> alpha signal
    confidences: dict[str, float]
    rationale: str
    market_regime: str

    @classmethod
    def parse(cls, text: str) -> "ActorForecast":
        """Parse Actor LLM output into structured forecast."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group(0) if json_match else "{}"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Actor JSON, using neutral signals")
            data = {}

        forecasts = data.get("forecasts", {})
        alphas = {}
        confidences = {}
        for sym, f in forecasts.items():
            alphas[sym] = max(-1.0, min(1.0, float(f.get("alpha", 0.0))))
            confidences[sym] = max(0.0, min(1.0, float(f.get("confidence", 0.5))))

        return cls(
            raw_text=text,
            alphas=alphas,
            confidences=confidences,
            rationale=data.get("reasoning", text),
            market_regime=data.get("market_regime", "unknown"),
        )


class Actor:
    """Actor agent that generates K candidate trading forecasts."""

    def __init__(self, config: LLMConfig):
        self.client = LLMClient(config)
        self.config = config

    def generate_forecasts(
        self,
        market_data: dict,
        onchain_data: dict,
        news_digest: dict,
        portfolio_state: dict,
        k: int | None = None,
    ) -> list[ActorForecast]:
        """Generate K candidate forecasts via nucleus sampling."""
        user_prompt = _build_actor_prompt(market_data, onchain_data, news_digest, portfolio_state)

        messages = [
            {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        responses = self.client.generate_candidates(messages, k=k)
        forecasts = []
        for resp in responses:
            try:
                forecast = ActorForecast.parse(resp.text)
                forecasts.append(forecast)
            except Exception as e:
                logger.warning("Failed to parse forecast candidate: %s", e)

        if not forecasts:
            logger.warning("No valid forecasts generated — returning neutral")
            forecasts.append(ActorForecast(
                raw_text="",
                alphas={"BTC/USDT": 0.0, "ETH/USDT": 0.0, "SOL/USDT": 0.0},
                confidences={"BTC/USDT": 0.0, "ETH/USDT": 0.0, "SOL/USDT": 0.0},
                rationale="No valid forecast generated",
                market_regime="unknown",
            ))

        return forecasts
