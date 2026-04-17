"""
Market data collection from exchanges and CoinMarketCap.
Provides OHLCV price data and market metrics.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests

from config import DataConfig

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collects OHLCV and market cap data from exchanges."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir) / "market"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange connection
        exchange_class = getattr(ccxt, config.exchange_id)
        exchange_params = {"enableRateLimit": True}
        if config.exchange_api_key:
            exchange_params["apiKey"] = config.exchange_api_key
            exchange_params["secret"] = config.exchange_secret
        self.exchange = exchange_class(exchange_params)

        # CoinMarketCap
        self.cmc_key = config.cmc_api_key or os.getenv("CMC_API_KEY", "")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: datetime | None = None,
        limit: int = 90,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from the exchange."""
        since_ms = int(since.timestamp() * 1000) if since else None
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except ccxt.BaseError as e:
            logger.warning("Exchange fetch failed for %s: %s — trying cache", symbol, e)
            return self._load_cache(symbol, timeframe)

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["symbol"] = symbol

        # Cache
        cache_path = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        df.to_parquet(cache_path)
        return df

    def fetch_market_metrics(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch market cap and volume from CoinMarketCap."""
        if not self.cmc_key:
            logger.info("No CMC API key — skipping market metrics")
            return {}

        slug_map = {
            "BTC/USDT": "bitcoin",
            "ETH/USDT": "ethereum",
            "SOL/USDT": "solana",
        }
        slugs = ",".join(slug_map.get(s, s.split("/")[0].lower()) for s in symbols)

        headers = {"X-CMC_PRO_API_KEY": self.cmc_key}
        url = f"{self.config.cmc_base_url}/v2/cryptocurrency/quotes/latest"
        try:
            resp = requests.get(url, headers=headers, params={"slug": slugs}, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", {})
        except Exception as e:
            logger.warning("CMC fetch failed: %s", e)
            return {}

        metrics = {}
        for _id, info in data.items():
            sym = info.get("symbol", "")
            quote = info.get("quote", {}).get("USD", {})
            metrics[sym] = {
                "market_cap": quote.get("market_cap", 0),
                "fdv": quote.get("fully_diluted_market_cap", 0),
                "volume_24h": quote.get("volume_24h", 0),
                "pct_change_24h": quote.get("percent_change_24h", 0),
            }
        return metrics

    def get_latest_snapshot(self, symbols: list[str]) -> dict[str, dict]:
        """Get a combined snapshot of price + market data for all assets."""
        snapshot = {}
        for sym in symbols:
            df = self.fetch_ohlcv(sym, limit=30)
            if df.empty:
                continue
            latest = df.iloc[-1]
            snapshot[sym] = {
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "volume": float(latest["volume"]),
                "returns_1d": float(df["close"].pct_change().iloc[-1]) if len(df) > 1 else 0.0,
                "returns_7d": float(df["close"].pct_change(7).iloc[-1]) if len(df) > 7 else 0.0,
                "volatility_30d": float(df["close"].pct_change().std()) if len(df) > 5 else 0.0,
            }

        # Overlay CMC metrics
        cmc = self.fetch_market_metrics(symbols)
        for sym in snapshot:
            base = sym.split("/")[0]
            if base in cmc:
                snapshot[sym].update(cmc[base])

        return snapshot

    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame()
