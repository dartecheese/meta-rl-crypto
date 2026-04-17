"""
Judge Agent — evaluates Actor forecasts using multi-objective rewards and Elo ratings.

The Judge ranks candidate forecasts based on realized performance across
five reward channels, using a dynamic Elo rating system.
"""

import json
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from config import LLMConfig, EloConfig, RewardConfig
from agents.llm_client import LLMClient
from agents.actor import ActorForecast
from agents.rewards import RewardVector

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a quantitative trading judge. Your role is to evaluate and compare trading forecasts.

Given two candidate forecasts and their realized performance metrics, determine which forecast was superior.

Evaluate across these dimensions:
1. **Absolute Returns** — Did the position generate profit after fees?
2. **Risk-Adjusted Returns (Sharpe)** — Was the return appropriate for the risk taken?
3. **Drawdown Control** — Did the forecast avoid large drawdowns?
4. **Sentiment Alignment** — Was the reasoning consistent with market sentiment?

Respond with JSON:
{
  "winner": 1 | 2,
  "confidence": <float 0-1>,
  "dimension_scores": {
    "returns": [<score_1>, <score_2>],
    "sharpe": [<score_1>, <score_2>],
    "drawdown": [<score_1>, <score_2>],
    "sentiment": [<score_1>, <score_2>]
  },
  "reasoning": "<why this forecast was better>"
}
"""


@dataclass
class CandidateScore:
    """Score for a single forecast candidate."""
    index: int
    reward: RewardVector
    total_score: float
    elo: float = 1500.0


@dataclass
class JudgePair:
    """A compared pair of forecasts with judgment."""
    winner_idx: int
    loser_idx: int
    confidence: float
    dimension_scores: dict


class Judge:
    """Judge agent that evaluates and ranks Actor forecasts using Elo."""

    def __init__(self, llm_config: LLMConfig, elo_config: EloConfig, reward_config: RewardConfig):
        self.client = LLMClient(llm_config)
        self.elo_config = elo_config
        self.rc = reward_config
        self._elo_ratings: dict[int, float] = {}  # candidate index -> elo
        self._comparison_matrix: dict[tuple[int, int], int] = {}  # (i,j) -> wins for i

    def score_candidates(
        self,
        forecasts: list[ActorForecast],
        rewards: list[RewardVector],
    ) -> list[CandidateScore]:
        """Score all candidates and return ranked list."""
        scores = []
        for i, (forecast, reward) in enumerate(zip(forecasts, rewards)):
            total = reward.total()
            scores.append(CandidateScore(
                index=i,
                reward=reward,
                total_score=total,
                elo=self._elo_ratings.get(i, self.elo_config.initial_elo),
            ))
        return sorted(scores, key=lambda s: s.total_score, reverse=True)

    def compare_pair(
        self,
        forecast_a: ActorForecast,
        reward_a: RewardVector,
        forecast_b: ActorForecast,
        reward_b: RewardVector,
        daily_volatility: float = 0.02,
    ) -> JudgePair:
        """Compare two forecasts using the LLM Judge and Elo update."""
        # First, try LLM-based comparison
        pair = self._llm_compare(forecast_a, reward_a, forecast_b, reward_b)

        # Update Elo ratings with dynamic K factor
        k_t = self.elo_config.k_base * (1 + daily_volatility / self.elo_config.sigma_max)
        self._update_elo(pair.winner_idx, pair.loser_idx, k_t)

        return pair

    def _llm_compare(
        self,
        forecast_a: ActorForecast,
        reward_a: RewardVector,
        forecast_b: ActorForecast,
        reward_b: RewardVector,
    ) -> JudgePair:
        """Use LLM to compare two forecasts."""
        user_prompt = f"""## Candidate 1
**Signals:** {json.dumps(forecast_a.alphas)}
**Rationale:** {forecast_a.rationale[:500]}
**Realized Rewards:**
- Return: {reward_a.r_return:.4f}
- Sharpe: {reward_a.r_sharpe:.4f}
- Drawdown: {reward_a.r_dd:.4f}
- Sentiment: {reward_a.r_sent:.4f}
- Liquidity: {reward_a.r_liq:.4f}
- Total: {reward_a.total():.4f}

## Candidate 2
**Signals:** {json.dumps(forecast_b.alphas)}
**Rationale:** {forecast_b.rationale[:500]}
**Realized Rewards:**
- Return: {reward_b.r_return:.4f}
- Sharpe: {reward_b.r_sharpe:.4f}
- Drawdown: {reward_b.r_dd:.4f}
- Sentiment: {reward_b.r_sent:.4f}
- Liquidity: {reward_b.r_liq:.4f}
- Total: {reward_b.total():.4f}

Which candidate performed better overall? Consider all dimensions."""

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            responses = self.client.chat(messages, temperature=0.3, n=1)
            text = responses[0].text
            # Parse response
            import re
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                winner = data.get("winner", 1)
                confidence = data.get("confidence", 0.5)
                dim_scores = data.get("dimension_scores", {})
            else:
                # Fall back to reward-based comparison
                winner = 1 if reward_a.total() >= reward_b.total() else 2
                confidence = abs(reward_a.total() - reward_b.total())
                dim_scores = {}
        except Exception as e:
            logger.warning("LLM judge comparison failed: %s — using reward totals", e)
            winner = 1 if reward_a.total() >= reward_b.total() else 2
            confidence = 0.5
            dim_scores = {}

        winner_idx = 0 if winner == 1 else 1
        loser_idx = 1 if winner == 1 else 0

        return JudgePair(
            winner_idx=winner_idx,
            loser_idx=loser_idx,
            confidence=confidence,
            dimension_scores=dim_scores,
        )

    def _update_elo(self, winner: int, loser: int, k_t: float):
        """Update Elo ratings with non-zero-sum dynamic K adjustment."""
        r_w = self._elo_ratings.get(winner, self.elo_config.initial_elo)
        r_l = self._elo_ratings.get(loser, self.elo_config.initial_elo)

        # Expected scores
        e_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400))
        e_l = 1.0 / (1.0 + 10 ** ((r_w - r_l) / 400))

        # Update
        self._elo_ratings[winner] = r_w + k_t * (1.0 - e_w)
        self._elo_ratings[loser] = r_l + k_t * (0.0 - e_l)

        # Record in comparison matrix
        self._comparison_matrix[(winner, loser)] = (
            self._comparison_matrix.get((winner, loser), 0) + 1
        )

    def select_preferences(
        self,
        scores: list[CandidateScore],
        forecasts: list[ActorForecast],
    ) -> tuple[ActorForecast, ActorForecast]:
        """
        Select positive (y_c) and negative (y_r) preference pairs.
        Per the paper: shortest top-tier as positive, longest low-tier as negative.
        """
        rho = self.rc.score_partition_rho
        all_scores = [s.total_score for s in scores]
        s_min, s_max = min(all_scores), max(all_scores)
        score_range = s_max - s_min

        if score_range < 1e-8:
            # All scores equal — return first and last
            return forecasts[scores[0].index], forecasts[scores[-1].index]

        # Top tier: [(1-rho)*s_max + rho*s_min, s_max]
        top_threshold = (1 - rho) * s_max + rho * s_min
        # Low tier: [s_min, (1-rho)*s_min + rho*s_max]
        low_threshold = (1 - rho) * s_min + rho * s_max

        top_tier = [s for s in scores if s.total_score >= top_threshold]
        low_tier = [s for s in scores if s.total_score <= low_threshold]

        if not top_tier:
            top_tier = [scores[0]]
        if not low_tier:
            low_tier = [scores[-1]]

        # Select shortest top-tier as positive (concise = better)
        positive = min(top_tier, key=lambda s: len(forecasts[s.index].raw_text))
        # Select longest low-tier as negative (verbose + bad = worst)
        negative = max(low_tier, key=lambda s: len(forecasts[s.index].raw_text))

        return forecasts[positive.index], forecasts[negative.index]

    def get_elo_rankings(self) -> dict[int, float]:
        return dict(sorted(self._elo_ratings.items(), key=lambda x: x[1], reverse=True))

    def reset_elos(self):
        self._elo_ratings.clear()
        self._comparison_matrix.clear()
