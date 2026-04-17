# Meta-RL-Crypto

A self-improving crypto trading agent based on the paper [*Meta-Learning Reinforcement Learning for Crypto-Return Prediction*](https://arxiv.org/abs/2509.09751).

The system uses a **triple-loop architecture** where a single LLM adopts three roles — Actor, Judge, and Meta-Judge — to continuously generate, evaluate, and refine trading signals for BTC, ETH, and SOL without human supervision.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Market Data                        │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │  Exchange  │  │  On-Chain  │  │  News/Sentiment  │ │
│  │  (CCXT)   │  │  (Dune)   │  │  (GNews)         │ │
│  └─────┬─────┘  └─────┬─────┘  └────────┬─────────┘ │
└────────┼──────────────┼─────────────────┼───────────┘
         └──────────────┼─────────────────┘
                        ▼
              ┌─────────────────┐
              │      Actor      │  Generates K candidate
              │   (LLM Role)   │  forecasts with α ∈ [-1,1]
              └────────┬────────┘
                       │ K candidates
                       ▼
              ┌─────────────────┐
              │      Judge      │  Ranks via 5-channel rewards
              │   (LLM Role)   │  + dynamic Elo ratings
              └────────┬────────┘
                       │ preference pairs
                       ▼
              ┌─────────────────┐
              │   Meta-Judge    │  Validates preferences,
              │   (LLM Role)   │  prevents reward drift
              └────────┬────────┘
                       │ corrected weights
                       ▼
              ┌─────────────────┐
              │    Portfolio    │  Executes best signal,
              │    Manager     │  manages risk
              └─────────────────┘
```

### Reward Channels

The Judge scores candidates across five normalized reward dimensions (all clipped to [-1, 1]):

| Channel | Description |
|---------|-------------|
| **R_return** | Realized net % gain after fees and slippage |
| **R_sharpe** | Incremental Sharpe ratio (exponentially-weighted variance) |
| **R_dd** | Intra-day drawdown penalty proportional to position size |
| **R_liq** | Bonus for keeping expected slippage below threshold |
| **R_sent** | Cosine similarity between rationale and news sentiment |

The Meta-Judge dynamically re-weights these channels via a DPO-style loss function, preventing reward hacking and length bias.

## Quick Start

### 1. Install

```bash
git clone https://github.com/dartecheese/meta-rl-crypto.git
cd meta-rl-crypto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

You need at minimum:
- **LLM endpoint** — any OpenAI-compatible API (vLLM, Ollama, OpenAI, Together AI, etc.)
- **Exchange** — any [CCXT-supported exchange](https://github.com/ccxt/ccxt) for market data

Optional (improves signal quality):
- CoinMarketCap API key — market cap and volume data
- GNews API key — news headlines and sentiment
- Dune Analytics API key — on-chain metrics

### 3. Run

```bash
# Dry run — simulated trades, real market data
python main.py trade --once

# Continuous dry run (daily interval)
python main.py trade --dry-run

# Live trading (places real orders)
python main.py trade --live

# Backtest on historical data
python main.py backtest --data-dir ./historical_data/

# View performance report
python main.py report
```

## Configuration

All settings can be overridden via environment variables or a JSON config file:

```bash
python main.py --config my_config.json trade --once
```

Example `my_config.json`:

```json
{
  "llm": {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "api_base": "http://localhost:8000/v1",
    "num_candidates": 5,
    "temperature": 0.7
  },
  "trading": {
    "initial_capital": 50000,
    "assets": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "fee_bps": 10
  }
}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible API endpoint |
| `LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model to use for all three roles |
| `EXCHANGE_ID` | `binance` | CCXT exchange ID |
| `INITIAL_CAPITAL` | `100000` | Starting capital in USD |
| `num_candidates` | `5` | Number of forecast candidates per step |
| `fee_bps` | `10` | Transaction fee in basis points |

## Backtesting

Prepare CSV files with columns `timestamp, open, high, low, close, volume` for each asset:

```
historical_data/
├── BTC_USDT.csv
├── ETH_USDT.csv
└── SOL_USDT.csv
```

Run:

```bash
python main.py backtest --data-dir ./historical_data/ --start 2025-01-01 --end 2025-06-30 --output results.json
```

## Project Structure

```
meta-rl-crypto/
├── main.py                 # CLI entry point
├── config/
│   └── settings.py         # All configuration dataclasses
├── agents/
│   ├── llm_client.py       # OpenAI-compatible LLM client
│   ├── actor.py            # Actor — forecast generation
│   ├── judge.py            # Judge — Elo-based candidate ranking
│   ├── meta_judge.py       # Meta-Judge — preference validation
│   └── rewards.py          # 5-channel reward computation
├── data/
│   ├── market_data.py      # Exchange + CoinMarketCap data
│   ├── onchain_data.py     # Dune Analytics on-chain metrics
│   └── news_sentiment.py   # GNews + SimHash dedup + sentiment
├── portfolio/
│   └── manager.py          # Position tracking, trade execution, P&L
├── trading/
│   ├── engine.py           # Core triple-loop orchestration
│   ├── live.py             # Live trading runner with signal handling
│   └── backtest.py         # Historical replay engine
├── requirements.txt
└── .env.example
```

## How It Works

Each trading step follows this cycle:

1. **Data Collection** — Fetch OHLCV prices, on-chain metrics, and news
2. **Actor** — Generate K candidate forecasts via nucleus sampling (p=0.9, T=0.7)
3. **Reward Scoring** — Evaluate each candidate across 5 reward channels
4. **Judge** — Rank candidates using multi-objective scores and update Elo ratings with dynamic K-factor: `K_t = K_base × (1 + σ_t / σ_max)`
5. **Preference Selection** — Shortest top-tier candidate as positive, longest low-tier as negative
6. **Meta-Judge** — Validate Judge preferences, detect bias, update reward channel weights via DPO loss
7. **Execution** — Execute the winning forecast's alpha signals as portfolio trades
8. **Learning** — Preference pairs stored for continuous improvement

## Disclaimer

This is a research implementation for educational purposes. Cryptocurrency trading involves substantial risk of loss. Always test thoroughly in dry-run mode before using real funds. The authors of the paper and this implementation are not responsible for any financial losses.

## Reference

```
@article{meta-rl-crypto-2025,
  title={Meta-Learning Reinforcement Learning for Crypto-Return Prediction},
  year={2025},
  url={https://arxiv.org/abs/2509.09751}
}
```
