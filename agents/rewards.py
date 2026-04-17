"""
Multi-reward channel construction.

Five normalized reward signals as defined in the paper:
  R_return  — realized net % gain after fees
  R_sharpe  — incremental Sharpe ratio contribution
  R_dd      — max intra-day drawdown penalty
  R_liq     — liquidity/slippage bonus
  R_sent    — cosine similarity with sentiment vector
"""

import numpy as np
from dataclasses import dataclass

from config import RewardConfig, TradingConfig


@dataclass
class RewardVector:
    """Multi-dimensional reward for a single trading step."""
    r_return: float
    r_sharpe: float
    r_dd: float
    r_liq: float
    r_sent: float

    def total(self, weights: np.ndarray | None = None) -> float:
        """Weighted sum of reward channels."""
        vec = self.as_array()
        if weights is None:
            weights = np.ones(5) / 5
        return float(np.dot(vec, weights))

    def as_array(self) -> np.ndarray:
        return np.array([self.r_return, self.r_sharpe, self.r_dd, self.r_liq, self.r_sent])


class RewardCalculator:
    """Computes multi-channel reward signals."""

    def __init__(self, reward_config: RewardConfig, trading_config: TradingConfig):
        self.rc = reward_config
        self.tc = trading_config
        self._returns_history: list[float] = []
        self._ew_var: float = 0.0
        self._ew_mean: float = 0.0
        self._peak_value: float = 0.0

    def compute(
        self,
        alpha: dict[str, float],
        price_returns: dict[str, float],
        portfolio_value: float,
        prev_portfolio_value: float,
        sentiment_vector: np.ndarray,
        actor_rationale: str,
        volumes: dict[str, float] | None = None,
    ) -> RewardVector:
        """Compute all five reward channels for a single step."""
        r_return = self._compute_return_reward(alpha, price_returns)
        r_sharpe = self._compute_sharpe_reward(r_return)
        r_dd = self._compute_drawdown_reward(portfolio_value, alpha)
        r_liq = self._compute_liquidity_reward(alpha, volumes)
        r_sent = self._compute_sentiment_reward(actor_rationale, sentiment_vector)

        return RewardVector(
            r_return=np.clip(r_return, -1, 1),
            r_sharpe=np.clip(r_sharpe, -1, 1),
            r_dd=np.clip(r_dd, -1, 1),
            r_liq=np.clip(r_liq, -1, 1),
            r_sent=np.clip(r_sent, -1, 1),
        )

    def _compute_return_reward(self, alpha: dict[str, float], price_returns: dict[str, float]) -> float:
        """R_return: realized net % gain after fees, weighted by position alpha."""
        total_return = 0.0
        fee_rate = self.tc.fee_bps / 10_000
        for sym in alpha:
            a = alpha.get(sym, 0.0)
            r = price_returns.get(sym, 0.0)
            slippage = np.random.normal(0, self.tc.slippage.get(sym, 0.0005))
            net = a * r - abs(a) * fee_rate - abs(a) * abs(slippage)
            total_return += net
        # Normalize by number of assets
        n = max(len(alpha), 1)
        return total_return / n * 10  # scale up for signal

    def _compute_sharpe_reward(self, step_return: float) -> float:
        """R_sharpe: incremental Sharpe ratio using exponentially-weighted variance."""
        self._returns_history.append(step_return)

        # EW update
        decay = self.rc.sharpe_decay
        self._ew_mean = decay * self._ew_mean + (1 - decay) * step_return
        self._ew_var = decay * self._ew_var + (1 - decay) * (step_return - self._ew_mean) ** 2

        ew_std = max(np.sqrt(self._ew_var), 1e-8)
        sharpe = self._ew_mean / ew_std
        return np.clip(sharpe / 3.0, -1, 1)  # normalize: Sharpe of 3 → reward of 1

    def _compute_drawdown_reward(self, portfolio_value: float, alpha: dict[str, float]) -> float:
        """R_dd: penalty proportional to drawdown * total position size."""
        self._peak_value = max(self._peak_value, portfolio_value)
        if self._peak_value == 0:
            return 0.0

        drawdown = (self._peak_value - portfolio_value) / self._peak_value
        total_exposure = sum(abs(a) for a in alpha.values()) / max(len(alpha), 1)

        # Penalty: larger drawdowns and larger positions get penalized more
        return -drawdown * total_exposure * 5  # scale for signal

    def _compute_liquidity_reward(
        self, alpha: dict[str, float], volumes: dict[str, float] | None
    ) -> float:
        """R_liq: bonus for keeping expected slippage below threshold."""
        if not volumes:
            return 0.0

        total_reward = 0.0
        for sym, a in alpha.items():
            vol = volumes.get(sym, 1e8)
            expected_slippage = abs(a) * self.tc.slippage.get(sym, 0.0005)
            if expected_slippage < self.rc.slippage_threshold:
                total_reward += 0.5  # bonus for staying under threshold
            else:
                total_reward -= (expected_slippage - self.rc.slippage_threshold) * 10

        return total_reward / max(len(alpha), 1)

    def _compute_sentiment_reward(
        self, rationale: str, sentiment_vector: np.ndarray
    ) -> float:
        """R_sent: cosine similarity between Actor's rationale direction and sentiment."""
        if len(sentiment_vector) == 0 or np.linalg.norm(sentiment_vector) < 1e-8:
            return 0.0

        # Extract directional signal from rationale text
        bullish_words = {"bullish", "long", "buy", "surge", "rally", "growth", "positive", "upside"}
        bearish_words = {"bearish", "short", "sell", "crash", "decline", "negative", "downside", "risk"}

        words = set(rationale.lower().split())
        bull_score = len(words & bullish_words)
        bear_score = len(words & bearish_words)
        total = bull_score + bear_score
        if total == 0:
            return 0.0

        rationale_signal = (bull_score - bear_score) / total  # [-1, 1]

        # Cosine similarity with overall sentiment
        overall_sentiment = sentiment_vector[0] if len(sentiment_vector) > 0 else 0.0
        # Simple alignment: same direction = positive reward
        return rationale_signal * overall_sentiment

    def reset(self):
        """Reset state for new episode."""
        self._returns_history.clear()
        self._ew_var = 0.0
        self._ew_mean = 0.0
        self._peak_value = 0.0
