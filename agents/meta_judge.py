"""
Meta-Judge Agent — refines the Judge's reward policy via preference comparisons.

Prevents reward drift and length bias by validating Judge preferences
through DPO-style loss and training a lightweight judge alignment model.
"""

import json
import logging
import re

import numpy as np

from config import LLMConfig, TrainingConfig
from agents.llm_client import LLMClient
from agents.rewards import RewardVector

logger = logging.getLogger(__name__)

META_JUDGE_SYSTEM_PROMPT = """You are a meta-evaluator for a trading system. Your role is to evaluate whether a Judge's preference between two trading forecasts is correct and well-calibrated.

Given:
- Two reward vectors from two candidate forecasts
- The Judge's preference (which it chose as winner)
- The Judge's reasoning

Determine if the Judge's preference is:
1. **Correct** — Did the Judge pick the genuinely better candidate?
2. **Well-Calibrated** — Is the confidence level appropriate?
3. **Free of Bias** — Is there evidence of length bias, recency bias, or reward hacking?

Respond with JSON:
{
  "judge_correct": true | false,
  "calibration_score": <float 0-1>,
  "bias_detected": {
    "length_bias": <float 0-1>,
    "reward_hacking": <float 0-1>,
    "recency_bias": <float 0-1>
  },
  "corrected_preference": 1 | 2 | null,
  "meta_confidence": <float 0-1>,
  "reasoning": "<explanation>"
}
"""


class PreferenceRecord:
    """A single preference observation for training."""

    def __init__(
        self,
        reward_chosen: RewardVector,
        reward_rejected: RewardVector,
        judge_correct: bool,
        meta_confidence: float,
    ):
        self.reward_chosen = reward_chosen
        self.reward_rejected = reward_rejected
        self.judge_correct = judge_correct
        self.meta_confidence = meta_confidence


class MetaJudge:
    """Meta-Judge that validates and corrects Judge preferences."""

    def __init__(self, llm_config: LLMConfig, training_config: TrainingConfig):
        self.client = LLMClient(llm_config)
        self.tc = training_config
        self.preference_history: list[PreferenceRecord] = []

        # Lightweight judge weights (simple linear model for alignment)
        # 5 reward dimensions -> preference probability
        self._judge_weights = np.ones(5) / 5
        self._meta_weights = np.ones(5) / 5

    def evaluate_judge_preference(
        self,
        reward_chosen: RewardVector,
        reward_rejected: RewardVector,
        judge_reasoning: str = "",
    ) -> dict:
        """Evaluate whether the Judge's preference is correct."""
        user_prompt = f"""## Judge's Chosen Candidate Rewards
- Return: {reward_chosen.r_return:.4f}
- Sharpe: {reward_chosen.r_sharpe:.4f}
- Drawdown: {reward_chosen.r_dd:.4f}
- Liquidity: {reward_chosen.r_liq:.4f}
- Sentiment: {reward_chosen.r_sent:.4f}
- Total: {reward_chosen.total():.4f}

## Judge's Rejected Candidate Rewards
- Return: {reward_rejected.r_return:.4f}
- Sharpe: {reward_rejected.r_sharpe:.4f}
- Drawdown: {reward_rejected.r_dd:.4f}
- Liquidity: {reward_rejected.r_liq:.4f}
- Sentiment: {reward_rejected.r_sent:.4f}
- Total: {reward_rejected.total():.4f}

## Judge's Reasoning
{judge_reasoning or 'No reasoning provided.'}

Is the Judge's preference correct and well-calibrated?"""

        messages = [
            {"role": "system", "content": META_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            responses = self.client.chat(messages, temperature=0.2, n=1)
            text = responses[0].text
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = self._heuristic_evaluation(reward_chosen, reward_rejected)
        except Exception as e:
            logger.warning("Meta-judge LLM failed: %s — using heuristic", e)
            result = self._heuristic_evaluation(reward_chosen, reward_rejected)

        # Record preference for training
        judge_correct = result.get("judge_correct", True)
        meta_confidence = result.get("meta_confidence", 0.5)

        self.preference_history.append(PreferenceRecord(
            reward_chosen=reward_chosen,
            reward_rejected=reward_rejected,
            judge_correct=judge_correct,
            meta_confidence=meta_confidence,
        ))

        return result

    def compute_dpo_loss(
        self,
        reward_chosen: RewardVector,
        reward_rejected: RewardVector,
    ) -> float:
        """
        Compute DPO-style loss: L_meta = -log σ(M_φ(r1, r2))
        Using the lightweight judge model M_φ.
        """
        beta = self.tc.dpo_beta

        # M_φ preference score
        chosen_score = np.dot(self._meta_weights, reward_chosen.as_array())
        rejected_score = np.dot(self._meta_weights, reward_rejected.as_array())

        diff = (chosen_score - rejected_score) / beta
        # Numerically stable log sigmoid
        if diff > 20:
            loss = 0.0
        elif diff < -20:
            loss = -diff
        else:
            loss = -np.log(1.0 / (1.0 + np.exp(-diff)))

        return float(loss)

    def compute_alignment_loss(
        self,
        reward_chosen: RewardVector,
        reward_rejected: RewardVector,
    ) -> float:
        """
        Compute Judge alignment loss: L_align = E[(M_φ - M_θ)²]
        Measures divergence between meta-judge and judge preference models.
        """
        meta_pref = np.dot(self._meta_weights, reward_chosen.as_array()) - \
                     np.dot(self._meta_weights, reward_rejected.as_array())
        judge_pref = np.dot(self._judge_weights, reward_chosen.as_array()) - \
                     np.dot(self._judge_weights, reward_rejected.as_array())

        return float((meta_pref - judge_pref) ** 2)

    def update_weights(self, learning_rate: float | None = None):
        """
        Update judge alignment weights based on accumulated preferences.
        Gradient step on alignment loss to bring judge closer to meta-judge.
        """
        lr = learning_rate or self.tc.learning_rate
        if not self.preference_history:
            return

        # Compute gradient of alignment loss w.r.t. judge weights
        grad = np.zeros(5)
        for record in self.preference_history[-50:]:  # last 50 records
            chosen = record.reward_chosen.as_array()
            rejected = record.reward_rejected.as_array()
            diff = chosen - rejected

            meta_pref = np.dot(self._meta_weights, diff)
            judge_pref = np.dot(self._judge_weights, diff)
            residual = judge_pref - meta_pref

            # Weight by meta-judge confidence and correctness
            weight = record.meta_confidence
            if not record.judge_correct:
                weight *= -1  # Penalize incorrect preferences

            grad += 2 * residual * diff * weight

        grad /= len(self.preference_history[-50:])

        # SGD step
        self._judge_weights -= lr * grad
        # Normalize to sum to 1
        self._judge_weights = np.maximum(self._judge_weights, 0.01)
        self._judge_weights /= self._judge_weights.sum()

        logger.info("Updated judge weights: %s", self._judge_weights.tolist())

    def get_reward_weights(self) -> np.ndarray:
        """Get current judge reward channel weights."""
        return self._judge_weights.copy()

    def _heuristic_evaluation(
        self, reward_chosen: RewardVector, reward_rejected: RewardVector
    ) -> dict:
        """Fallback heuristic when LLM is unavailable."""
        chosen_total = reward_chosen.total()
        rejected_total = reward_rejected.total()
        correct = chosen_total >= rejected_total

        return {
            "judge_correct": correct,
            "calibration_score": 0.5,
            "bias_detected": {"length_bias": 0.0, "reward_hacking": 0.0, "recency_bias": 0.0},
            "corrected_preference": None,
            "meta_confidence": min(abs(chosen_total - rejected_total) * 5, 1.0),
            "reasoning": "Heuristic evaluation based on reward totals",
        }
