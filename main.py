#!/usr/bin/env python3
"""
Meta-RL-Crypto — Self-Improving Crypto Trading Agent
Based on: "Meta-Learning Reinforcement Learning for Crypto-Return Prediction"

Usage:
    # Dry-run (no real trades):
    python main.py trade --dry-run

    # Live trading:
    python main.py trade --live

    # Single step:
    python main.py trade --once

    # Backtest with CSV data:
    python main.py backtest --data-dir ./historical_data/

    # Show performance:
    python main.py report
"""

import argparse
import json
import logging
import os
import sys

from config import Config
from trading.engine import TradingEngine
from trading.live import LiveTrader
from trading.backtest import Backtester


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | None = None) -> Config:
    """Load configuration, overlaying environment variables."""
    config = Config()

    # Override from env vars
    if os.getenv("LLM_API_BASE"):
        config.llm.api_base = os.getenv("LLM_API_BASE")
    if os.getenv("LLM_API_KEY"):
        config.llm.api_key = os.getenv("LLM_API_KEY")
    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")
    if os.getenv("CMC_API_KEY"):
        config.data.cmc_api_key = os.getenv("CMC_API_KEY")
    if os.getenv("GNEWS_API_KEY"):
        config.data.gnews_api_key = os.getenv("GNEWS_API_KEY")
    if os.getenv("DUNE_API_KEY"):
        config.data.dune_api_key = os.getenv("DUNE_API_KEY")
    if os.getenv("EXCHANGE_ID"):
        config.data.exchange_id = os.getenv("EXCHANGE_ID")
    if os.getenv("EXCHANGE_API_KEY"):
        config.data.exchange_api_key = os.getenv("EXCHANGE_API_KEY")
    if os.getenv("EXCHANGE_SECRET"):
        config.data.exchange_secret = os.getenv("EXCHANGE_SECRET")
    if os.getenv("INITIAL_CAPITAL"):
        config.trading.initial_capital = float(os.getenv("INITIAL_CAPITAL"))

    # Override from JSON config file
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = json.load(f)
        # Apply nested overrides
        for section, values in overrides.items():
            if hasattr(config, section):
                obj = getattr(config, section)
                for key, val in values.items():
                    if hasattr(obj, key):
                        setattr(obj, key, val)

    return config


def cmd_trade(args, config: Config):
    """Run the trading agent."""
    dry_run = not args.live
    trader = LiveTrader(config, dry_run=dry_run)

    if args.once:
        summary = trader.run_once()
        print(json.dumps(summary, indent=2, default=str))
    else:
        interval = args.interval * 3600  # hours to seconds
        trader.run_loop(interval_seconds=interval)


def cmd_backtest(args, config: Config):
    """Run a backtest on historical data."""
    import pandas as pd

    data_dir = args.data_dir
    price_data = {}

    for sym in config.trading.assets:
        filename = sym.replace("/", "_") + ".csv"
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: No data file found for {sym} at {filepath}")
            continue

        df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
        # Ensure required columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            print(f"Error: {filepath} must have columns: {required}")
            continue

        price_data[sym] = df
        print(f"Loaded {len(df)} rows for {sym}")

    if not price_data:
        print("No price data loaded. Create CSV files with columns: timestamp, open, high, low, close, volume")
        print(f"Expected files in {data_dir}/: " + ", ".join(s.replace("/", "_") + ".csv" for s in config.trading.assets))
        sys.exit(1)

    backtester = Backtester(config)
    results = backtester.run(
        price_data=price_data,
        start_date=args.start,
        end_date=args.end,
    )

    # Save results
    output_path = args.output or "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Backtest Results: {results['start_date']} to {results['end_date']}")
    print(f"{'='*50}")
    print(f"  Total Return:    {results['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:    {results['annualized_sharpe']:.2f}")
    print(f"  Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
    print(f"  Final Value:     ${results['final_value']:,.2f}")
    print(f"  Trading Days:    {results['trading_days']}")
    print(f"  Total Trades:    {results['total_trades']}")
    print(f"{'='*50}")


def cmd_report(args, config: Config):
    """Show current performance report."""
    engine = TradingEngine(config)

    # Try to load latest portfolio state
    from pathlib import Path
    state_dir = Path("state")
    if state_dir.exists():
        steps = sorted(state_dir.glob("step_*"))
        if steps:
            latest = steps[-1]
            portfolio_path = latest / "portfolio.json"
            if portfolio_path.exists():
                engine.portfolio.load_state(str(portfolio_path))

            summary_path = latest / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    last_summary = json.load(f)
                print(f"Last step: {last_summary.get('step')} at {last_summary.get('timestamp')}")

    report = engine.get_performance_report()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Meta-RL-Crypto Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Run the trading agent")
    trade_parser.add_argument("--live", action="store_true", help="Enable live trading (real orders)")
    trade_parser.add_argument("--once", action="store_true", help="Run a single step and exit")
    trade_parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run (default)")
    trade_parser.add_argument("--interval", type=float, default=24, help="Trading interval in hours (default: 24)")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run a backtest")
    bt_parser.add_argument("--data-dir", type=str, required=True, help="Directory with CSV price data")
    bt_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("--output", type=str, help="Output file for results")

    # Report command
    subparsers.add_parser("report", help="Show performance report")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)
    config = load_config(args.config)

    if args.command == "trade":
        cmd_trade(args, config)
    elif args.command == "backtest":
        cmd_backtest(args, config)
    elif args.command == "report":
        cmd_report(args, config)


if __name__ == "__main__":
    main()
