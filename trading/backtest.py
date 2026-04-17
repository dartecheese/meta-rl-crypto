"""
Backtesting Engine — replay historical data through the trading engine.

Uses stored OHLCV data to simulate the Meta-RL-Crypto agent's performance
without look-ahead bias.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from agents.actor import Actor
from agents.judge import Judge
from agents.meta_judge import MetaJudge
from agents.rewards import RewardCalculator
from portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)


class Backtester:
    """
    Replays historical price data through the Meta-RL-Crypto system.
    No look-ahead bias — each step only sees data up to that point.
    """

    def __init__(self, config: Config):
        self.config = config
        self.actor = Actor(config.llm)
        self.judge = Judge(config.llm, config.elo, config.reward)
        self.meta_judge = MetaJudge(config.llm, config.training)
        self.reward_calc = RewardCalculator(config.reward, config.trading)

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """
        Run a backtest over historical price data.

        Args:
            price_data: dict of symbol -> DataFrame with OHLCV columns and datetime index
            start_date: optional start date string (YYYY-MM-DD)
            end_date: optional end date string (YYYY-MM-DD)

        Returns:
            Backtest results dictionary
        """
        portfolio = PortfolioManager(self.config.trading)
        self.reward_calc.reset()
        self.judge.reset_elos()

        # Align dates across all assets
        common_dates = None
        for sym, df in price_data.items():
            dates = set(df.index.date) if hasattr(df.index, 'date') else set(df.index)
            common_dates = dates if common_dates is None else common_dates & dates

        common_dates = sorted(common_dates)
        if start_date:
            start = pd.Timestamp(start_date).date()
            common_dates = [d for d in common_dates if d >= start]
        if end_date:
            end = pd.Timestamp(end_date).date()
            common_dates = [d for d in common_dates if d <= end]

        if len(common_dates) < 2:
            raise ValueError("Not enough overlapping dates for backtest")

        logger.info("Backtesting %d days: %s to %s", len(common_dates), common_dates[0], common_dates[-1])

        results = []
        prev_prices = {}

        for i, date in enumerate(common_dates):
            # Build market snapshot (no look-ahead)
            market_snapshot = {}
            prices = {}
            price_returns = {}

            for sym, df in price_data.items():
                # Get data up to current date
                mask = df.index.date <= date if hasattr(df.index, 'date') else df.index <= str(date)
                hist = df[mask]
                if hist.empty:
                    continue

                latest = hist.iloc[-1]
                prices[sym] = float(latest["close"])

                # Returns
                if len(hist) > 1:
                    ret_1d = float(latest["close"] / hist.iloc[-2]["close"] - 1)
                else:
                    ret_1d = 0.0
                ret_7d = float(latest["close"] / hist.iloc[-min(7, len(hist))]["close"] - 1) if len(hist) > 1 else 0.0
                vol_30d = float(hist["close"].pct_change().tail(30).std()) if len(hist) > 5 else 0.02

                price_returns[sym] = ret_1d

                market_snapshot[sym] = {
                    "open": float(latest["open"]),
                    "high": float(latest["high"]),
                    "low": float(latest["low"]),
                    "close": float(latest["close"]),
                    "volume": float(latest["volume"]),
                    "returns_1d": ret_1d,
                    "returns_7d": ret_7d,
                    "volatility_30d": vol_30d,
                }

            if not prices:
                continue

            portfolio.update_prices(prices)

            # Generate forecasts (with reduced candidates for speed)
            onchain_stub = {sym: {"chain": sym.split("/")[0].lower(), "tx_count_24h": 0, "active_wallets_24h": 0, "total_value_transferred_usd": 0, "mean_gas_gwei": 0, "median_gas_gwei": 0, "total_gas_used": 0} for sym in prices}
            news_stub = {"news_summary": "Backtesting — no live news.", "sentiment_vector": [0, 0, 0, 0, 0], "article_count": 0}

            forecasts = self.actor.generate_forecasts(
                market_data=market_snapshot,
                onchain_data=onchain_stub,
                news_digest=news_stub,
                portfolio_state=portfolio.get_state().to_dict(),
                k=max(self.config.llm.num_candidates // 2, 2),
            )

            # Score candidates
            sentiment_vec = np.zeros(5)
            volumes = {sym: data.get("volume", 0) for sym, data in market_snapshot.items()}
            prev_value = portfolio.total_value

            rewards = []
            for fc in forecasts:
                reward = self.reward_calc.compute(
                    alpha=fc.alphas,
                    price_returns=price_returns,
                    portfolio_value=prev_value,
                    prev_portfolio_value=prev_value,
                    sentiment_vector=sentiment_vec,
                    actor_rationale=fc.rationale,
                    volumes=volumes,
                )
                rewards.append(reward)

            # Judge
            scored = self.judge.score_candidates(forecasts, rewards)
            best = forecasts[scored[0].index]

            # Execute
            trades = portfolio.execute_signals(best.alphas, prices, datetime.combine(date, datetime.min.time()))

            step_result = {
                "date": str(date),
                "portfolio_value": portfolio.total_value,
                "cash": portfolio.cash,
                "alphas": best.alphas,
                "trades": len(trades),
                "regime": best.market_regime,
            }
            results.append(step_result)

            if (i + 1) % 10 == 0:
                logger.info("  Day %d/%d: $%.2f (%.2f%%)", i + 1, len(common_dates),
                            portfolio.total_value,
                            (portfolio.total_value / self.config.trading.initial_capital - 1) * 100)

            prev_prices = dict(prices)

        # Compute final metrics
        values = [r["portfolio_value"] for r in results]
        returns_arr = np.diff(values) / np.array(values[:-1]) if len(values) > 1 else np.array([])

        total_return = (values[-1] / self.config.trading.initial_capital - 1) if values else 0

        sharpe = 0.0
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = float(returns_arr.mean() / returns_arr.std() * np.sqrt(252))

        cumulative = np.cumprod(1 + returns_arr) if len(returns_arr) > 0 else np.array([1.0])
        peak = np.maximum.accumulate(cumulative)
        max_dd = float(((peak - cumulative) / peak).max()) if len(cumulative) > 0 else 0.0

        report = {
            "start_date": str(common_dates[0]),
            "end_date": str(common_dates[-1]),
            "trading_days": len(common_dates),
            "total_return_pct": total_return * 100,
            "annualized_sharpe": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "final_value": values[-1] if values else self.config.trading.initial_capital,
            "total_trades": sum(r["trades"] for r in results),
            "daily_results": results,
        }

        logger.info("=== Backtest Complete ===")
        logger.info("Return: %.2f%% | Sharpe: %.2f | Max DD: %.2f%% | Trades: %d",
                     report["total_return_pct"], report["annualized_sharpe"],
                     report["max_drawdown_pct"], report["total_trades"])

        return report
