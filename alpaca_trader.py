```markdown
# Alpaca Auto-Trader

## Overview

Automated SPY Trading with Alpaca

## Setup

1. Create free account at alpaca.markets
2. Get API keys (paper trading first!)
3. pip install alpaca-trade-api
4. Set environment variables:
   export ALPACA_API_KEY="your-key"
   export ALPACA_SECRET_KEY="your-secret"
   export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading

## Strategies

### 1. Trend Rider

* Logic: Long SPY when 50-day MA > 200-day MA
* Leverage Range: 1.0 - 2.0
* Risk Profile: Medium

### 2. Golden Boost

* Logic: Long SPY when RSI < 30
* Leverage Range: 1.0 - 2.0
* Risk Profile: High

### 3. Signal Stacker

* Logic: Long SPY when 3 out of 5 signals are positive
* Leverage Range: 1.0 - 2.0
* Risk Profile: Medium

### 4. Steady Eddie

* Logic: Long SPY when 20-day MA > Close
* Leverage Range: 1.0 - 2.0
* Risk Profile: Medium

### 5. Full Send

* Logic: Long SPY when 50-day MA > Close
* Leverage Range: 1.0 - 2.0
* Risk Profile: High

### 6. RSI Bounce

* Logic: Long SPY when RSI > 70
* Leverage Range: 1.0 - 2.0
* Risk Profile: Medium

### 7. Dual Momentum

* Logic: Long SPY when both 50-day and 200-day MOM > 0
* Leverage Range: 1.0 - 2.0
* Risk Profile: Medium

### 8. Ensemble

* Logic: Long SPY when 3 out of 7 signals are positive
* Leverage Range: 1.0 - 2.0
* Risk Profile: Medium

### 9. Voting System

* Logic: Long SPY when 4 out of 7 signals are positive
* Leverage Range: 1.0 - 2.0
* Risk Profile: Medium

### 10. Vol Adaptive

* Logic: Long SPY when Vol < 0.25
* Leverage Range: 1.0 - 2.0
* Risk Profile: Low

### 11. Conservative

* Logic: Long SPY when 5 out of 11 signals are positive
* Leverage Range: 1.0 - 2.0
* Risk Profile: Low

## Dashboard Sections

* Overview
* Positions
* Trades
* Signals
* Settings
* Account Info
* Paper Trading
* Live Trading

## Strategy Implementation Pattern

The `get_strategy_position()` function in `dashboard.py` takes in the following parameters:
```python
def get_strategy_position(strategy: str, leverage: float) -> dict:
    # Implementation details...
```
This function returns a dictionary containing the strategy's position details.

## Getting Started

1. Run the `main.py` script to execute the auto-trader.
2. Set environment variables for Alpaca API keys and base URL.
3. Configure the dashboard to display the desired strategy and leverage.
4. Monitor the auto-trader's performance and adjust settings as needed.
```