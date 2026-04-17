"""
Live Trading Runner — executes real trades on an exchange via CCXT.

Runs the Meta-RL-Crypto engine on a schedule, placing real orders.
"""

import logging
import signal
import time
from datetime import datetime

import ccxt

from config import Config
from trading.engine import TradingEngine

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Runs the trading engine in live mode.
    Places real orders on the configured exchange.
    """

    def __init__(self, config: Config, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.engine = TradingEngine(config)
        self._running = False

        # Exchange for order placement
        if not dry_run:
            exchange_class = getattr(ccxt, config.data.exchange_id)
            self.exchange = exchange_class({
                "apiKey": config.data.exchange_api_key,
                "secret": config.data.exchange_secret,
                "enableRateLimit": True,
            })
        else:
            self.exchange = None

    def run_once(self) -> dict:
        """Execute a single trading step."""
        logger.info("Running trading step (dry_run=%s)", self.dry_run)

        summary = self.engine.run_step()

        if not self.dry_run and self.exchange:
            self._place_orders(summary)

        # Print report
        report = self.engine.get_performance_report()
        logger.info("Portfolio: $%.2f | Return: %.2f%% | Sharpe: %.2f | Trades: %d",
                     report.get("portfolio_value", 0),
                     report.get("total_return_pct", 0),
                     report.get("sharpe_ratio", 0),
                     report.get("total_trades", 0))

        return summary

    def run_loop(self, interval_seconds: int = 86400):
        """Run continuously at the specified interval (default: daily)."""
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Starting live trading loop (interval=%ds, dry_run=%s)",
                     interval_seconds, self.dry_run)

        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error("Trading step failed: %s", e, exc_info=True)

            if self._running:
                next_run = datetime.utcnow().timestamp() + interval_seconds
                logger.info("Next step at %s",
                            datetime.utcfromtimestamp(next_run).isoformat())
                # Sleep in small increments so we can respond to signals
                remaining = interval_seconds
                while remaining > 0 and self._running:
                    time.sleep(min(remaining, 10))
                    remaining -= 10

    def _place_orders(self, summary: dict):
        """Place real orders on the exchange based on engine output."""
        alphas = summary.get("best_alphas", {})

        for sym, alpha in alphas.items():
            if abs(alpha) < 0.01:
                continue

            try:
                ticker = self.exchange.fetch_ticker(sym)
                price = ticker["last"]

                if alpha > 0:
                    # Market buy
                    balance = self.exchange.fetch_balance()
                    quote = sym.split("/")[1]
                    available = balance.get(quote, {}).get("free", 0)
                    order_value = available * abs(alpha) * 0.95  # 5% buffer
                    qty = order_value / price

                    if qty * price < 10:  # min order size
                        continue

                    logger.info("LIVE ORDER: BUY %s qty=%.6f @ ~$%.2f", sym, qty, price)
                    order = self.exchange.create_market_buy_order(sym, qty)
                    logger.info("Order placed: %s", order.get("id"))

                elif alpha < 0:
                    # Market sell
                    base = sym.split("/")[0]
                    balance = self.exchange.fetch_balance()
                    available = balance.get(base, {}).get("free", 0)
                    qty = available * abs(alpha)

                    if qty * price < 10:
                        continue

                    logger.info("LIVE ORDER: SELL %s qty=%.6f @ ~$%.2f", sym, qty, price)
                    order = self.exchange.create_market_sell_order(sym, qty)
                    logger.info("Order placed: %s", order.get("id"))

            except Exception as e:
                logger.error("Failed to place order for %s: %s", sym, e)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received — finishing current step...")
        self._running = False
