# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SPY (S&P 500 ETF) trading signal system that implements multiple technical analysis strategies with dynamic leverage. The project includes signal generation, backtesting, automated trading via Alpaca API, and an interactive Streamlit dashboard.

## Architecture

### Core Components

**Signal System** (`daily_signal_checker.py`)
- Calculates 5 independent technical signals:
  1. Trend Filter: Price > 50 DMA AND 200 DMA slope positive
  2. Golden Cross: 50 DMA > 200 DMA
  3. Momentum: Price > 20 DMA
  4. RSI OK: RSI < 70 (not overbought)
  5. Low Volatility: Annualized volatility < 25%
- Implements 5 distinct trading strategies (A-E) with different risk profiles
- Strategies use 0x (cash), 1x (SPY), 1.5x (50% SPY + 50% SSO), or 2x (SSO) leverage
- Strategy D (Conservative) is the recommended default - requires 3+ signals before entering

**Backtesting Suite**
- `backtest_spy_ma.py`: Original long/short MA strategy (deprecated approach)
- `backtest_spy_ma_v2.py`: Improved long-only variants with cash positions
- `backtest_spy_combined.py`: Full implementation of all 5 strategies with leverage
  - Calculates comprehensive metrics: CAGR, Sharpe, Calmar, max drawdown
  - Yearly breakdowns and position distribution analysis
  - Uses 10 years of historical data with 200+ day warmup period for indicators

**Live Trading** (`alpaca_trader.py`)
- Integrates with Alpaca API for automated execution
- Currently implements Strategy D (Conservative) only
- Requires environment variables: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`
- Paper trading mode by default - production code needs order sizing and error handling improvements

**Dashboard** (`dashboard.py`)
- Streamlit web interface for real-time signal monitoring
- Interactive backtesting with equity curves and performance metrics
- Visualizes price charts with MAs, RSI indicators, and signal status

## Common Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Daily Signal Check
```bash
python daily_signal_checker.py           # Full report with all strategies
python daily_signal_checker.py --mobile  # Compact mobile-friendly output
```

### Backtesting
```bash
python backtest_spy_combined.py    # Run all 5 strategies (recommended)
python backtest_spy_ma_v2.py       # Run improved long-only variants
```

### Dashboard
```bash
streamlit run dashboard.py
```

### Alpaca Trading (requires API keys)
```bash
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
python alpaca_trader.py
```

## Data & Indicators

All scripts use Yahoo Finance via `yfinance` for historical data. Key calculations:
- Moving averages: 20, 50, 200 day periods
- MA200 slope: 20-day rate of change of 200 DMA
- RSI: 14-day standard calculation
- Volatility: 20-day rolling standard deviation, annualized

**Important**: Scripts handle MultiIndex columns from yfinance and require sufficient historical data for indicator warmup (at least 200 days).

## Strategy Implementation Pattern

When adding or modifying strategies:
1. Calculate all technical indicators first (MAs, RSI, volatility)
2. Generate boolean signals for each condition
3. Combine signals to determine position size (0, 1.0, 1.5, 2.0)
4. Apply position on T+1 (shift signals by 1 day for realistic trading)
5. Calculate strategy returns as `position.shift(1) * daily_return`

## Key Design Decisions

- **Long-only with cash**: After backtesting, short positions underperform. All production strategies use long + cash.
- **Leverage via SSO**: 2x exposure achieved through SSO (ProShares Ultra S&P 500), not margin
- **Signal confirmation**: Conservative strategies require multiple aligned signals to reduce whipsaws
- **200 DMA slope**: Critical filter - positive slope indicates bull market, avoiding exposure in bear markets

## Testing & Validation

The codebase uses backtesting as the primary validation method. When testing changes:
- Run `backtest_spy_combined.py` and compare key metrics: Sharpe ratio, max drawdown, Calmar ratio
- Check yearly breakdown to ensure strategy performs across different market regimes
- Verify position distribution - excessive 2x leverage or cash sitting may indicate issues
- Compare against buy & hold baseline

## Dependencies

Core: `yfinance`, `pandas`, `numpy`
Visualization: `streamlit`, `plotly`
Trading: `alpaca-py` (optional, only for live trading)
