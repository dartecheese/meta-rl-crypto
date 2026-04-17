"""
Portfolio Manager — tracks positions, executes trades, manages risk.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0

    @property
    def value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.quantity * (self.current_price - self.avg_entry_price)


@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    fee: float
    slippage: float
    alpha: float  # the signal that triggered this trade


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position]
    total_value: float
    trades: list[Trade] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "total_value": self.total_value,
            "positions": {
                sym: {
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "value": pos.value,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for sym, pos in self.positions.items()
            },
        }


class PortfolioManager:
    """Manages portfolio state, executes alpha signals, tracks P&L."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.cash = config.initial_capital * config.cash_reserve_pct
        self.positions: dict[str, Position] = {
            sym: Position(symbol=sym) for sym in config.assets
        }
        # Allocate initial capital equally across assets
        allocation_per_asset = (config.initial_capital * (1 - config.cash_reserve_pct)) / len(config.assets)
        self._initial_allocation = allocation_per_asset
        self.trade_history: list[Trade] = []
        self._value_history: list[float] = [config.initial_capital]

    def update_prices(self, prices: dict[str, float]):
        """Update current market prices for all positions."""
        for sym, price in prices.items():
            if sym in self.positions:
                self.positions[sym].current_price = price

    def execute_signals(
        self,
        alphas: dict[str, float],
        prices: dict[str, float],
        timestamp: datetime | None = None,
    ) -> list[Trade]:
        """
        Execute trading signals. Alpha > 0 = buy, alpha < 0 = sell.
        Returns list of executed trades.
        """
        timestamp = timestamp or datetime.utcnow()
        self.update_prices(prices)
        trades = []

        for sym, alpha in alphas.items():
            if abs(alpha) < 0.01:  # dead zone — not worth the fees
                continue

            pos = self.positions.get(sym)
            if pos is None:
                continue

            price = prices.get(sym, 0)
            if price <= 0:
                continue

            fee_rate = self.config.fee_bps / 10_000
            slippage_std = self.config.slippage.get(sym, 0.0005)
            slippage = np.random.normal(0, slippage_std)
            exec_price = price * (1 + slippage)

            if alpha > 0:
                # Buy — use proportion of available cash
                max_buy_value = self.cash * abs(alpha)
                if max_buy_value < 1.0:  # minimum trade size
                    continue
                qty = max_buy_value / exec_price
                fee = max_buy_value * fee_rate
                if max_buy_value + fee > self.cash:
                    max_buy_value = self.cash / (1 + fee_rate)
                    qty = max_buy_value / exec_price
                    fee = max_buy_value * fee_rate

                # Update position
                total_cost = pos.quantity * pos.avg_entry_price + qty * exec_price
                pos.quantity += qty
                pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0
                self.cash -= (qty * exec_price + fee)

                trades.append(Trade(
                    timestamp=timestamp, symbol=sym, side="buy",
                    quantity=qty, price=exec_price, fee=fee,
                    slippage=slippage, alpha=alpha,
                ))

            elif alpha < 0:
                # Sell — reduce position proportionally
                sell_qty = pos.quantity * abs(alpha)
                if sell_qty * exec_price < 1.0:  # minimum trade size
                    continue
                sell_qty = min(sell_qty, pos.quantity)
                proceeds = sell_qty * exec_price
                fee = proceeds * fee_rate

                pos.quantity -= sell_qty
                if pos.quantity < 1e-10:
                    pos.quantity = 0.0
                    pos.avg_entry_price = 0.0
                self.cash += (proceeds - fee)

                trades.append(Trade(
                    timestamp=timestamp, symbol=sym, side="sell",
                    quantity=sell_qty, price=exec_price, fee=fee,
                    slippage=slippage, alpha=alpha,
                ))

        self.trade_history.extend(trades)
        self._value_history.append(self.total_value)
        return trades

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.value for p in self.positions.values())

    @property
    def returns_history(self) -> list[float]:
        if len(self._value_history) < 2:
            return []
        vals = np.array(self._value_history)
        return list(np.diff(vals) / vals[:-1])

    def get_state(self) -> PortfolioState:
        return PortfolioState(
            cash=self.cash,
            positions=dict(self.positions),
            total_value=self.total_value,
            trades=list(self.trade_history),
        )

    def get_daily_returns(self, symbol: str) -> dict[str, float]:
        """Get price return for each asset from last two values."""
        # This gets called with actual price data externally
        return {}

    def save_state(self, path: str):
        """Persist portfolio state to JSON."""
        state = self.get_state().to_dict()
        state["trade_count"] = len(self.trade_history)
        state["value_history"] = self._value_history
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str):
        """Restore portfolio state from JSON."""
        with open(path) as f:
            state = json.load(f)
        self.cash = state["cash"]
        for sym, pos_data in state.get("positions", {}).items():
            if sym in self.positions:
                self.positions[sym].quantity = pos_data["quantity"]
                self.positions[sym].avg_entry_price = pos_data["avg_entry_price"]
                self.positions[sym].current_price = pos_data["current_price"]
        self._value_history = state.get("value_history", [self.config.initial_capital])
