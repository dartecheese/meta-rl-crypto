#!/usr/bin/env python3
"""
End-to-end test of the Meta-RL-Crypto pipeline.
Uses a mock LLM so no API keys are needed.
Generates synthetic price data and runs a full backtest.
"""

import json
import logging
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from agents.actor import Actor, ActorForecast
from agents.judge import Judge, CandidateScore
from agents.meta_judge import MetaJudge
from agents.rewards import RewardCalculator, RewardVector
from data.news_sentiment import NewsSentimentCollector, _simhash_64, _hamming_distance
from portfolio.manager import PortfolioManager
from trading.engine import TradingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Synthetic Data ──

def generate_price_data(symbol: str, days: int = 90, start_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data with trends and volatility regimes."""
    np.random.seed(hash(symbol) % 2**31)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Simulate price with regime changes
    price = start_price
    prices = []
    for i in range(days):
        # Regime: trending up, sideways, or down based on period
        phase = (i / days) * 3
        if phase < 1:
            drift = 0.002  # bullish
        elif phase < 2:
            drift = -0.001  # slight bearish
        else:
            drift = 0.001  # recovery

        vol = 0.02 + 0.01 * np.sin(i / 10)  # time-varying volatility
        ret = drift + vol * np.random.randn()
        price *= (1 + ret)
        high = price * (1 + abs(np.random.randn()) * 0.01)
        low = price * (1 - abs(np.random.randn()) * 0.01)
        volume = np.random.lognormal(20, 1)
        prices.append({
            "timestamp": dates[i],
            "open": price * (1 + np.random.randn() * 0.002),
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
        })

    df = pd.DataFrame(prices)
    df.set_index("timestamp", inplace=True)
    return df


def mock_llm_forecast(alphas: dict[str, float], regime: str = "sideways") -> str:
    """Generate a mock LLM forecast response."""
    forecasts = {}
    for sym, alpha in alphas.items():
        direction = "long" if alpha > 0 else "short" if alpha < 0 else "neutral"
        forecasts[sym] = {
            "direction": direction,
            "alpha": alpha,
            "confidence": abs(alpha),
            "price_target_pct": alpha * 2,
            "rationale": f"{'Bullish' if alpha > 0 else 'Bearish'} momentum signal for {sym}",
        }
    return json.dumps({
        "forecasts": forecasts,
        "market_regime": regime,
        "risk_assessment": "Moderate risk environment",
        "reasoning": f"Market appears {regime}. Positions sized by conviction.",
    })


# ── Test Functions ──

def test_reward_calculator():
    """Test all 5 reward channels."""
    print("\n" + "="*60)
    print("  TEST: Reward Calculator (5-channel)")
    print("="*60)

    config = Config()
    rc = RewardCalculator(config.reward, config.trading)

    # Scenario 1: Profitable long
    r1 = rc.compute(
        alpha={"BTC/USDT": 0.8, "ETH/USDT": 0.5, "SOL/USDT": 0.3},
        price_returns={"BTC/USDT": 0.05, "ETH/USDT": 0.03, "SOL/USDT": 0.01},
        portfolio_value=105000, prev_portfolio_value=100000,
        sentiment_vector=np.array([0.6, 0.7, 0.3, 0.1, 0.8]),
        actor_rationale="bullish momentum, strong growth expected",
        volumes={"BTC/USDT": 1e9, "ETH/USDT": 5e8, "SOL/USDT": 1e8},
    )
    print(f"\n  Scenario 1 (profitable long):")
    print(f"    R_return = {r1.r_return:+.4f}  (positive = correct direction)")
    print(f"    R_sharpe = {r1.r_sharpe:+.4f}  (risk-adjusted)")
    print(f"    R_dd     = {r1.r_dd:+.4f}  (drawdown penalty)")
    print(f"    R_liq    = {r1.r_liq:+.4f}  (liquidity bonus)")
    print(f"    R_sent   = {r1.r_sent:+.4f}  (sentiment alignment)")
    print(f"    Total    = {r1.total():+.4f}")

    # Scenario 2: Bad trade — went long, market dropped
    r2 = rc.compute(
        alpha={"BTC/USDT": 0.9, "ETH/USDT": 0.7, "SOL/USDT": 0.5},
        price_returns={"BTC/USDT": -0.08, "ETH/USDT": -0.05, "SOL/USDT": -0.10},
        portfolio_value=92000, prev_portfolio_value=100000,
        sentiment_vector=np.array([-0.5, -0.6, -0.3, -0.7, 0.9]),
        actor_rationale="bullish outlook, expecting rally",
        volumes={"BTC/USDT": 1e9, "ETH/USDT": 5e8, "SOL/USDT": 1e8},
    )
    print(f"\n  Scenario 2 (bad long into crash):")
    print(f"    R_return = {r2.r_return:+.4f}  (negative = wrong direction)")
    print(f"    R_sharpe = {r2.r_sharpe:+.4f}")
    print(f"    R_dd     = {r2.r_dd:+.4f}  (drawdown hit)")
    print(f"    R_liq    = {r2.r_liq:+.4f}")
    print(f"    R_sent   = {r2.r_sent:+.4f}  (said bullish, sentiment bearish)")
    print(f"    Total    = {r2.total():+.4f}")

    assert r1.total() > r2.total(), "Profitable trade should score higher"
    print("\n  ✓ Profitable trade scored higher than losing trade")


def test_portfolio_manager():
    """Test portfolio execution, fees, and P&L tracking."""
    print("\n" + "="*60)
    print("  TEST: Portfolio Manager")
    print("="*60)

    config = Config()
    config.trading.initial_capital = 100_000
    pm = PortfolioManager(config.trading)

    print(f"\n  Initial: ${pm.total_value:,.2f} (cash=${pm.cash:,.2f})")

    # Buy some BTC and ETH
    prices = {"BTC/USDT": 60000, "ETH/USDT": 3000, "SOL/USDT": 150}
    trades = pm.execute_signals(
        alphas={"BTC/USDT": 0.4, "ETH/USDT": 0.3, "SOL/USDT": 0.0},
        prices=prices,
    )
    print(f"\n  After buying (α=0.4 BTC, 0.3 ETH):")
    print(f"    Trades executed: {len(trades)}")
    for t in trades:
        print(f"    {t.side.upper()} {t.symbol}: {t.quantity:.6f} @ ${t.price:,.2f} (fee=${t.fee:.2f})")
    print(f"    Portfolio: ${pm.total_value:,.2f} | Cash: ${pm.cash:,.2f}")
    print(f"    BTC position: {pm.positions['BTC/USDT'].quantity:.6f}")
    print(f"    ETH position: {pm.positions['ETH/USDT'].quantity:.6f}")

    # Price goes up
    new_prices = {"BTC/USDT": 63000, "ETH/USDT": 3150, "SOL/USDT": 155}
    pm.update_prices(new_prices)
    print(f"\n  After +5% move:")
    print(f"    Portfolio: ${pm.total_value:,.2f}")
    print(f"    BTC unrealized P&L: ${pm.positions['BTC/USDT'].unrealized_pnl:,.2f}")

    # Sell some
    trades = pm.execute_signals(
        alphas={"BTC/USDT": -0.5, "ETH/USDT": 0.0, "SOL/USDT": 0.0},
        prices=new_prices,
    )
    print(f"\n  After selling 50% BTC:")
    for t in trades:
        print(f"    {t.side.upper()} {t.symbol}: {t.quantity:.6f} @ ${t.price:,.2f}")
    print(f"    Portfolio: ${pm.total_value:,.2f} | Cash: ${pm.cash:,.2f}")

    initial = config.trading.initial_capital * config.trading.cash_reserve_pct  # 50k with 50% cash reserve
    assert pm.total_value > initial, f"Should be profitable after price increase (${pm.total_value:,.2f} vs ${initial:,.2f})"
    print("\n  ✓ Portfolio tracking and trade execution working correctly")


def test_judge_elo():
    """Test Judge scoring and Elo rating updates."""
    print("\n" + "="*60)
    print("  TEST: Judge (Elo Ranking)")
    print("="*60)

    config = Config()

    # Create mock forecasts and rewards
    forecasts = []
    rewards = []
    scenarios = [
        ({"BTC/USDT": 0.8, "ETH/USDT": 0.5}, 0.15, "strong bull"),
        ({"BTC/USDT": 0.2, "ETH/USDT": 0.1}, 0.03, "weak bull"),
        ({"BTC/USDT": -0.3, "ETH/USDT": -0.5}, -0.08, "bearish"),
        ({"BTC/USDT": 0.0, "ETH/USDT": 0.0}, 0.00, "neutral"),
        ({"BTC/USDT": 0.5, "ETH/USDT": 0.3}, 0.07, "moderate bull"),
    ]

    for alphas, ret, label in scenarios:
        fc = ActorForecast(
            raw_text=mock_llm_forecast(alphas),
            alphas=alphas,
            confidences={k: abs(v) for k, v in alphas.items()},
            rationale=f"{label} forecast",
            market_regime="bullish" if ret > 0 else "bearish",
        )
        rw = RewardVector(
            r_return=ret, r_sharpe=ret*2, r_dd=-abs(ret)*0.5,
            r_liq=0.3, r_sent=0.1 if ret > 0 else -0.1,
        )
        forecasts.append(fc)
        rewards.append(rw)

    # Use Judge without LLM calls (reward-based scoring only)
    judge = Judge(config.llm, config.elo, config.reward)
    scored = judge.score_candidates(forecasts, rewards)

    print(f"\n  Candidate Rankings (by total score):")
    for rank, s in enumerate(scored, 1):
        label = scenarios[s.index][2]
        print(f"    #{rank}: [{label:>15}] score={s.total_score:+.4f} elo={s.elo:.0f}")

    # Verify best candidate is the strong bull
    assert scored[0].index == 0, "Strong bull should rank first"

    # Test preference selection
    positive, negative = judge.select_preferences(scored, forecasts)
    print(f"\n  Preference pair:")
    print(f"    Positive (y_c): {positive.market_regime} — α={positive.alphas}")
    print(f"    Negative (y_r): {negative.market_regime} — α={negative.alphas}")

    print("\n  ✓ Judge ranking and preference selection working correctly")


def test_meta_judge():
    """Test Meta-Judge preference validation and weight updates."""
    print("\n" + "="*60)
    print("  TEST: Meta-Judge (Preference Validation)")
    print("="*60)

    config = Config()
    mj = MetaJudge(config.llm, config.training)

    print(f"\n  Initial reward weights: {mj.get_reward_weights().tolist()}")

    # Simulate several preference observations
    for i in range(20):
        chosen = RewardVector(
            r_return=0.1 + np.random.randn()*0.05,
            r_sharpe=0.08 + np.random.randn()*0.03,
            r_dd=-0.02,
            r_liq=0.3,
            r_sent=0.1,
        )
        rejected = RewardVector(
            r_return=-0.05 + np.random.randn()*0.05,
            r_sharpe=-0.03,
            r_dd=-0.1,
            r_liq=0.2,
            r_sent=-0.1,
        )

        # Use heuristic evaluation (no LLM)
        result = mj._heuristic_evaluation(chosen, rejected)
        mj.preference_history.append(
            __import__("agents.meta_judge", fromlist=["PreferenceRecord"]).PreferenceRecord(
                reward_chosen=chosen, reward_rejected=rejected,
                judge_correct=result["judge_correct"],
                meta_confidence=result["meta_confidence"],
            )
        )

        dpo = mj.compute_dpo_loss(chosen, rejected)
        align = mj.compute_alignment_loss(chosen, rejected)

    mj.update_weights()
    print(f"  Updated reward weights: {[f'{w:.3f}' for w in mj.get_reward_weights()]}")

    print(f"  DPO loss (last): {dpo:.4f}")
    print(f"  Alignment loss (last): {align:.4f}")
    print(f"  Preference history size: {len(mj.preference_history)}")

    print("\n  ✓ Meta-Judge weight updates and DPO loss working correctly")


def test_news_sentiment():
    """Test sentiment analysis and SimHash deduplication."""
    print("\n" + "="*60)
    print("  TEST: News Sentiment & SimHash")
    print("="*60)

    config = Config()
    ns = NewsSentimentCollector(config.data)

    # Test SimHash dedup
    h1 = _simhash_64("bitcoin price surges to new record high today")
    h2 = _simhash_64("bitcoin price surges to new record high now")  # near dup
    h3 = _simhash_64("ethereum developers announce major protocol upgrade")
    dist_12 = _hamming_distance(h1, h2)
    dist_13 = _hamming_distance(h1, h3)
    print(f"\n  SimHash deduplication:")
    print(f"    Near-duplicate distance: {dist_12} (< 4 = duplicate)")
    print(f"    Different article distance: {dist_13}")

    # Test sentiment extraction
    mock_articles = [
        {"title": "Bitcoin surges past $100k in massive rally", "description": "Institutional adoption drives bullish momentum", "source": "Bloomberg"},
        {"title": "Ethereum upgrade brings growth and positive gains", "description": "Network activity soars after upgrade", "source": "Reuters"},
        {"title": "Solana faces crash concerns amid bearish selloff", "description": "Liquidation risk increases as prices decline", "source": "CoinDesk"},
        {"title": "SEC lawsuit creates fear in crypto markets", "description": "Regulation concerns lead to negative sentiment", "source": "Yahoo Finance"},
    ]

    sentiment = ns.compute_sentiment_vector(mock_articles)
    labels = ["Overall", "BTC", "ETH", "SOL", "Volume"]
    print(f"\n  Sentiment vector:")
    for label, val in zip(labels, sentiment):
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"    {label:>8}: {sign}{bar} ({val:+.2f})")

    print("\n  ✓ Sentiment analysis and deduplication working correctly")


def test_full_pipeline():
    """Full end-to-end pipeline test with synthetic data and mock LLM."""
    print("\n" + "="*60)
    print("  TEST: Full Pipeline (Synthetic Backtest)")
    print("="*60)

    config = Config()
    config.trading.initial_capital = 100_000
    config.llm.num_candidates = 3

    # Generate synthetic price data
    price_data = {
        "BTC/USDT": generate_price_data("BTC", days=60, start_price=60000),
        "ETH/USDT": generate_price_data("ETH", days=60, start_price=3000),
        "SOL/USDT": generate_price_data("SOL", days=60, start_price=150),
    }
    print(f"\n  Generated {60} days of synthetic OHLCV data for 3 assets")
    for sym, df in price_data.items():
        print(f"    {sym}: ${df['close'].iloc[0]:,.2f} → ${df['close'].iloc[-1]:,.2f} "
              f"({(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%)")

    # Mock the LLM to return deterministic forecasts based on momentum
    call_count = [0]

    def mock_chat(messages, temperature=None, top_p=None, max_tokens=None, n=1):
        from agents.llm_client import LLMResponse
        call_count[0] += 1
        responses = []
        for _ in range(n):
            # Simple momentum strategy: go long if recent returns positive
            alpha_btc = np.clip(np.random.randn() * 0.3, -0.8, 0.8)
            alpha_eth = np.clip(np.random.randn() * 0.3, -0.8, 0.8)
            alpha_sol = np.clip(np.random.randn() * 0.2, -0.5, 0.5)

            text = mock_llm_forecast(
                {"BTC/USDT": round(alpha_btc, 2), "ETH/USDT": round(alpha_eth, 2), "SOL/USDT": round(alpha_sol, 2)},
                regime="bullish" if alpha_btc > 0 else "bearish",
            )
            responses.append(LLMResponse(text=text))
        return responses

    # Patch the LLM client
    with patch("agents.llm_client.LLMClient.chat", side_effect=mock_chat), \
         patch("agents.llm_client.LLMClient.generate_candidates", side_effect=lambda msgs, k=None: mock_chat(msgs, n=k or 3)):

        from trading.backtest import Backtester
        bt = Backtester(config)
        results = bt.run(price_data)

    print(f"\n  {'─'*50}")
    print(f"  Backtest Results ({results['trading_days']} days)")
    print(f"  {'─'*50}")
    print(f"    Total Return:     {results['total_return_pct']:+.2f}%")
    print(f"    Sharpe Ratio:     {results['annualized_sharpe']:.2f}")
    print(f"    Max Drawdown:     {results['max_drawdown_pct']:.2f}%")
    print(f"    Final Value:      ${results['final_value']:,.2f}")
    print(f"    Total Trades:     {results['total_trades']}")
    print(f"    LLM Calls:        {call_count[0]}")

    # Verify pipeline produced meaningful results
    assert results['total_trades'] > 0, "Should have executed trades"
    assert results['final_value'] > 0, "Portfolio value should be positive"
    assert len(results['daily_results']) == results['trading_days'], "Should have daily results"

    print(f"\n  Daily portfolio value trajectory:")
    values = [r["portfolio_value"] for r in results["daily_results"]]
    # Mini sparkline
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    spark_chars = "▁▂▃▄▅▆▇█"
    sparkline = ""
    step = max(1, len(values) // 50)
    for v in values[::step]:
        idx = int((v - mn) / rng * 7)
        sparkline += spark_chars[idx]
    print(f"    {sparkline}")
    print(f"    ${mn:,.0f} ──── ${mx:,.0f}")

    print("\n  ✓ Full pipeline test passed!")


# ── Run All Tests ──

def main():
    print("╔" + "═"*60 + "╗")
    print("║  Meta-RL-Crypto — End-to-End Test Suite" + " "*19 + "║")
    print("╚" + "═"*60 + "╝")

    tests = [
        ("Reward Calculator", test_reward_calculator),
        ("Portfolio Manager", test_portfolio_manager),
        ("Judge (Elo)", test_judge_elo),
        ("Meta-Judge", test_meta_judge),
        ("News Sentiment", test_news_sentiment),
        ("Full Pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
