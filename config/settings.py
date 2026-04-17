"""
Meta-RL-Crypto Configuration
Based on: "Meta-Learning Reinforcement Learning for Crypto-Return Prediction"
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM configuration for Actor/Judge/Meta-Judge roles."""
    model: str = "gemini-2.0-flash"  # set via env var or override
    api_key: str = ""  # set via env var GEMINI_API_KEY
    temperature: float = 0.7
    top_p: float = 0.9
    num_candidates: int = 5    # K candidate responses per step
    max_tokens: int = 1024


@dataclass
class TradingConfig:
    """Portfolio and trading parameters."""
    initial_capital: float = 100_000.0
    cash_reserve_pct: float = 0.50  # 50% cash reserve
    assets: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    fee_bps: float = 10.0  # 10 basis points
    slippage: dict[str, float] = field(default_factory=lambda: {
        "BTC/USDT": 0.0005,   # 0.05% std dev
        "ETH/USDT": 0.0005,
        "SOL/USDT": 0.0012,   # 0.12% std dev
    })
    max_position_pct: float = 1.0  # alpha in [-1, 1]
    rebalance_frequency: str = "daily"


@dataclass
class RewardConfig:
    """Multi-reward channel parameters."""
    sharpe_window: int = 30  # exponentially-weighted variance window
    sharpe_decay: float = 0.94  # EW decay factor
    drawdown_lookback: int = 1  # intra-day
    slippage_threshold: float = 0.001  # 10 bps max expected slippage
    score_partition_rho: float = 0.3  # rho for top/low tier partitioning


@dataclass
class EloConfig:
    """Elo rating system for Judge."""
    k_base: float = 32.0
    sigma_max: float = 0.05  # max daily volatility normalizer
    initial_elo: float = 1500.0


@dataclass
class DataConfig:
    """Data source configuration."""
    # CoinMarketCap
    cmc_api_key: str = ""  # set via env var CMC_API_KEY
    cmc_base_url: str = "https://pro-api.coinmarketcap.com"

    # News
    gnews_api_key: str = ""  # set via env var GNEWS_API_KEY

    # On-chain (Dune Analytics)
    dune_api_key: str = ""  # set via env var DUNE_API_KEY

    # Exchange (for live trading)
    exchange_id: str = "binance"  # any ccxt-supported exchange
    exchange_api_key: str = ""
    exchange_secret: str = ""

    # Cache
    cache_dir: str = "data_cache"


@dataclass
class TrainingConfig:
    """Meta-learning training parameters."""
    dpo_beta: float = 0.1  # temperature for DPO loss
    learning_rate: float = 1e-5
    judge_alignment_weight: float = 0.5
    meta_judge_epochs: int = 3
    save_preferences: bool = True
    preferences_dir: str = "preferences"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
