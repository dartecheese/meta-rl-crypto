"""
Core Trading Engine — orchestrates the triple-loop Actor/Judge/Meta-Judge cycle.

This is the main loop that:
1. Collects market data
2. Generates K candidate forecasts (Actor)
3. Scores candidates with multi-channel rewards (Judge)
4. Validates Judge preferences (Meta-Judge)
5. Executes the winning forecast
6. Updates weights and preferences
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from config import Config
from agents.actor import Actor, ActorForecast
from agents.judge import Judge
from agents.meta_judge import MetaJudge
from agents.rewards import RewardCalculator, RewardVector
from data.market_data import MarketDataCollector
from data.onchain_data import OnChainDataCollector
from data.news_sentiment import NewsSentimentCollector
from portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Meta-RL-Crypto Trading Engine.

    Implements the closed-loop triple architecture:
      Actor → Judge → Meta-Judge → (feedback) → Actor
    """

    def __init__(self, config: Config):
        self.config = config

        # Data collectors
        self.market_data = MarketDataCollector(config.data)
        self.onchain_data = OnChainDataCollector(config.data)
        self.news_sentiment = NewsSentimentCollector(config.data)

        # Agents
        self.actor = Actor(config.llm)
        self.judge = Judge(config.llm, config.elo, config.reward)
        self.meta_judge = MetaJudge(config.llm, config.training)

        # Reward calculator
        self.reward_calc = RewardCalculator(config.reward, config.trading)

        # Portfolio
        self.portfolio = PortfolioManager(config.trading)

        # State
        self.step_count = 0
        self.state_dir = Path("state")
        self.state_dir.mkdir(exist_ok=True)

    def run_step(self) -> dict:
        """
        Execute one complete trading step (the full triple-loop).
        Returns a summary dict of what happened.
        """
        self.step_count += 1
        timestamp = datetime.utcnow()
        logger.info("=== Trading Step %d at %s ===", self.step_count, timestamp)

        # ── Step 1: Collect data ──
        market_snapshot = self.market_data.get_latest_snapshot(self.config.trading.assets)
        onchain_metrics = self.onchain_data.get_all_chain_metrics()
        news_digest = self.news_sentiment.get_news_digest()

        # Current prices
        prices = {sym: data["close"] for sym, data in market_snapshot.items()}
        self.portfolio.update_prices(prices)

        portfolio_state = self.portfolio.get_state().to_dict()

        # ── Step 2: Actor generates K candidates ──
        logger.info("Generating %d candidate forecasts...", self.config.llm.num_candidates)
        forecasts = self.actor.generate_forecasts(
            market_data=market_snapshot,
            onchain_data=onchain_metrics,
            news_digest=news_digest,
            portfolio_state=portfolio_state,
            k=self.config.llm.num_candidates,
        )
        logger.info("Generated %d valid forecasts", len(forecasts))

        # ── Step 3: Score each candidate with multi-channel rewards ──
        # For live trading, we use projected rewards (not realized yet)
        # For backtesting, realized rewards are computed after price moves
        rewards = []
        sentiment_vec = np.array(news_digest.get("sentiment_vector", [0]*5))
        volumes = {sym: data.get("volume", 0) for sym, data in market_snapshot.items()}

        prev_value = self.portfolio.total_value
        for forecast in forecasts:
            # Project reward based on recent price momentum (not future prices)
            price_returns = {sym: data.get("returns_1d", 0) for sym, data in market_snapshot.items()}
            reward = self.reward_calc.compute(
                alpha=forecast.alphas,
                price_returns=price_returns,
                portfolio_value=prev_value,
                prev_portfolio_value=prev_value,
                sentiment_vector=sentiment_vec,
                actor_rationale=forecast.rationale,
                volumes=volumes,
            )
            rewards.append(reward)

        # ── Step 4: Judge ranks candidates ──
        scored = self.judge.score_candidates(forecasts, rewards)

        # Compare top pairs for Elo updates
        daily_vol = np.mean([data.get("volatility_30d", 0.02) for data in market_snapshot.values()])
        if len(scored) >= 2:
            for i in range(min(len(scored) - 1, 3)):
                idx_a, idx_b = scored[i].index, scored[i + 1].index
                pair = self.judge.compare_pair(
                    forecasts[idx_a], rewards[idx_a],
                    forecasts[idx_b], rewards[idx_b],
                    daily_volatility=daily_vol,
                )

        # Select preference pair
        positive_forecast, negative_forecast = self.judge.select_preferences(scored, forecasts)

        # ── Step 5: Meta-Judge validates ──
        pos_reward = rewards[scored[0].index]
        neg_reward = rewards[scored[-1].index]

        meta_result = self.meta_judge.evaluate_judge_preference(
            reward_chosen=pos_reward,
            reward_rejected=neg_reward,
        )
        logger.info("Meta-Judge: correct=%s, confidence=%.2f",
                     meta_result.get("judge_correct"), meta_result.get("meta_confidence", 0))

        # Compute training losses
        dpo_loss = self.meta_judge.compute_dpo_loss(pos_reward, neg_reward)
        align_loss = self.meta_judge.compute_alignment_loss(pos_reward, neg_reward)

        # Update reward weights periodically
        if self.step_count % 5 == 0:
            self.meta_judge.update_weights()
            logger.info("Updated reward weights: %s", self.meta_judge.get_reward_weights().tolist())

        # ── Step 6: Execute the best forecast ──
        best_forecast = forecasts[scored[0].index]
        logger.info("Executing best forecast: %s", best_forecast.alphas)

        trades = self.portfolio.execute_signals(
            alphas=best_forecast.alphas,
            prices=prices,
            timestamp=timestamp,
        )

        for trade in trades:
            logger.info("  %s %s: %.6f @ $%.2f (fee=$%.2f)",
                        trade.side.upper(), trade.symbol,
                        trade.quantity, trade.price, trade.fee)

        # ── Summary ──
        summary = {
            "step": self.step_count,
            "timestamp": timestamp.isoformat(),
            "portfolio_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
            "num_candidates": len(forecasts),
            "best_alphas": best_forecast.alphas,
            "market_regime": best_forecast.market_regime,
            "trades_executed": len(trades),
            "dpo_loss": dpo_loss,
            "alignment_loss": align_loss,
            "meta_judge_correct": meta_result.get("judge_correct"),
            "reward_weights": self.meta_judge.get_reward_weights().tolist(),
            "elo_rankings": self.judge.get_elo_rankings(),
        }

        # Persist
        self._save_step(summary)
        return summary

    def _save_step(self, summary: dict):
        """Save step results and portfolio state."""
        step_dir = self.state_dir / f"step_{self.step_count:06d}"
        step_dir.mkdir(exist_ok=True)

        with open(step_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.portfolio.save_state(str(step_dir / "portfolio.json"))

    def get_performance_report(self) -> dict:
        """Generate a performance summary."""
        returns = self.portfolio.returns_history
        if not returns:
            return {"status": "no trades yet"}

        returns_arr = np.array(returns)
        total_return = (self.portfolio.total_value / self.config.trading.initial_capital) - 1

        # Sharpe ratio (daily, rf=0)
        sharpe = float(returns_arr.mean() / max(returns_arr.std(), 1e-8)) if len(returns_arr) > 1 else 0.0

        # Max drawdown
        cumulative = np.cumprod(1 + returns_arr)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / peak
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        return {
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "total_trades": len(self.portfolio.trade_history),
            "portfolio_value": self.portfolio.total_value,
            "steps": self.step_count,
            "mean_daily_return_pct": float(returns_arr.mean() * 100),
        }
