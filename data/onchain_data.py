"""
On-chain data collection from Dune Analytics.
Provides transaction counts, active wallets, gas metrics.
"""

import os
import logging
from datetime import datetime

import requests

from config import DataConfig

logger = logging.getLogger(__name__)

# Pre-built Dune query IDs for common on-chain metrics
# Replace these with your own Dune query IDs or use the API to create them
DEFAULT_QUERIES = {
    "ethereum": {
        "tx_count": 0,       # placeholder — set your Dune query ID
        "active_wallets": 0,
        "gas_metrics": 0,
    },
    "solana": {
        "tx_count": 0,
        "active_wallets": 0,
    },
    "bitcoin": {
        "tx_count": 0,
        "active_wallets": 0,
    },
}


class OnChainDataCollector:
    """Collects on-chain metrics from Dune Analytics."""

    def __init__(self, config: DataConfig):
        self.api_key = config.dune_api_key or os.getenv("DUNE_API_KEY", "")
        self.base_url = "https://api.dune.com/api/v1"
        self.headers = {"X-Dune-API-Key": self.api_key}

    def _execute_query(self, query_id: int, params: dict | None = None) -> list[dict]:
        """Execute a Dune query and return results."""
        if not self.api_key or query_id == 0:
            return []

        try:
            # Trigger execution
            exec_url = f"{self.base_url}/query/{query_id}/execute"
            resp = requests.post(exec_url, headers=self.headers, json=params or {}, timeout=30)
            resp.raise_for_status()
            execution_id = resp.json()["execution_id"]

            # Poll for results (simplified — in production use async)
            import time
            results_url = f"{self.base_url}/execution/{execution_id}/results"
            for _ in range(30):
                resp = requests.get(results_url, headers=self.headers, timeout=15)
                data = resp.json()
                if data.get("state") == "QUERY_STATE_COMPLETED":
                    return data.get("result", {}).get("rows", [])
                time.sleep(2)
        except Exception as e:
            logger.warning("Dune query %d failed: %s", query_id, e)
        return []

    def get_onchain_metrics(self, chain: str = "ethereum") -> dict:
        """Get latest on-chain metrics for a given chain."""
        queries = DEFAULT_QUERIES.get(chain, {})

        metrics = {
            "chain": chain,
            "tx_count_24h": 0,
            "active_wallets_24h": 0,
            "total_value_transferred_usd": 0,
            "mean_gas_gwei": 0,
            "median_gas_gwei": 0,
            "total_gas_used": 0,
        }

        if not self.api_key:
            logger.info("No Dune API key — returning placeholder on-chain metrics")
            return self._placeholder_metrics(chain)

        # Fetch tx count
        rows = self._execute_query(queries.get("tx_count", 0))
        if rows:
            metrics["tx_count_24h"] = rows[0].get("tx_count", 0)
            metrics["total_value_transferred_usd"] = rows[0].get("value_usd", 0)

        # Fetch active wallets
        rows = self._execute_query(queries.get("active_wallets", 0))
        if rows:
            metrics["active_wallets_24h"] = rows[0].get("active_wallets", 0)

        # Fetch gas metrics (ETH only)
        if chain == "ethereum":
            rows = self._execute_query(queries.get("gas_metrics", 0))
            if rows:
                metrics["mean_gas_gwei"] = rows[0].get("mean_gas", 0)
                metrics["median_gas_gwei"] = rows[0].get("median_gas", 0)
                metrics["total_gas_used"] = rows[0].get("total_gas", 0)

        return metrics

    def _placeholder_metrics(self, chain: str) -> dict:
        """Return reasonable placeholder metrics when no API key is set."""
        placeholders = {
            "bitcoin": {
                "chain": "bitcoin",
                "tx_count_24h": 350_000,
                "active_wallets_24h": 800_000,
                "total_value_transferred_usd": 15_000_000_000,
                "mean_gas_gwei": 0,
                "median_gas_gwei": 0,
                "total_gas_used": 0,
            },
            "ethereum": {
                "chain": "ethereum",
                "tx_count_24h": 1_200_000,
                "active_wallets_24h": 500_000,
                "total_value_transferred_usd": 5_000_000_000,
                "mean_gas_gwei": 25,
                "median_gas_gwei": 20,
                "total_gas_used": 100_000_000_000,
            },
            "solana": {
                "chain": "solana",
                "tx_count_24h": 40_000_000,
                "active_wallets_24h": 1_500_000,
                "total_value_transferred_usd": 2_000_000_000,
                "mean_gas_gwei": 0,
                "median_gas_gwei": 0,
                "total_gas_used": 0,
            },
        }
        return placeholders.get(chain, {"chain": chain})

    def get_all_chain_metrics(self) -> dict[str, dict]:
        """Get metrics for all tracked chains."""
        chains = {"BTC/USDT": "bitcoin", "ETH/USDT": "ethereum", "SOL/USDT": "solana"}
        return {symbol: self.get_onchain_metrics(chain) for symbol, chain in chains.items()}
