#!/usr/bin/env python3
"""
Automated SPY Trading with Alpaca
Run daily via cron job to auto-execute trades.

Setup:
1. Create free account at alpaca.markets
2. Get API keys (paper trading first!)
3. pip install alpaca-trade-api
4. Set environment variables:
   export ALPACA_API_KEY="your-key"
   export ALPACA_SECRET_KEY="your-secret"
   export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Check if alpaca is installed
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_INSTALLED = True
except ImportError:
    ALPACA_INSTALLED = False
    print("⚠️  Alpaca not installed. Run: pip install alpaca-py")

def get_signal_conservative(ticker='SPY'):
    """Get Conservative strategy signal (D)."""
    data = yf.download(ticker, period='1y', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    data['MA200_Slope'] = data['MA200'].pct_change(periods=20)
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain/loss))
    data['Vol'] = data['Close'].pct_change().rolling(20).std() * (252 ** 0.5)
    
    latest = data.iloc[-1]
    
    signals = [
        (latest['Close'] > latest['MA50']) and (latest['MA200_Slope'] > 0),
        latest['MA50'] > latest['MA200'],
        latest['Close'] > latest['MA20'],
        latest['RSI'] < 70,
        latest['Vol'] < 0.25
    ]
    
    count = sum(signals)
    
    if count == 5:
        return 2.0, count
    elif count >= 4:
        return 1.5, count
    elif count >= 3:
        return 1.0, count
    else:
        return 0.0, count

def execute_trade(target_leverage):
    """
    Execute trades to achieve target leverage.
    Uses SPY for 1x, SSO for 2x exposure.
    """
    if not ALPACA_INSTALLED:
        print("Alpaca not installed - cannot execute trades")
        return
    
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return
    
    # Initialize client
    client = TradingClient(api_key, secret_key, paper=True)
    
    # Get account info
    account = client.get_account()
    equity = float(account.equity)
    
    print(f"Account Equity: ${equity:,.2f}")
    print(f"Target Leverage: {target_leverage}x")
    
    # Get current positions
    positions = client.get_all_positions()
    current_spy = 0
    current_sso = 0
    
    for pos in positions:
        if pos.symbol == 'SPY':
            current_spy = float(pos.market_value)
        elif pos.symbol == 'SSO':
            current_sso = float(pos.market_value)
    
    current_exposure = current_spy + (current_sso * 2)  # SSO is 2x
    current_leverage = current_exposure / equity if equity > 0 else 0
    
    print(f"Current Leverage: {current_leverage:.2f}x")
    
    # Calculate target allocations
    if target_leverage == 0:
        # Sell everything
        target_spy = 0
        target_sso = 0
    elif target_leverage == 1.0:
        # 100% SPY
        target_spy = equity
        target_sso = 0
    elif target_leverage == 1.5:
        # 50% SPY + 50% SSO = 1.5x exposure
        target_spy = equity * 0.5
        target_sso = equity * 0.5
    elif target_leverage == 2.0:
        # 100% SSO
        target_spy = 0
        target_sso = equity
    
    # Execute trades
    print(f"\nTarget: SPY=${target_spy:,.0f}, SSO=${target_sso:,.0f}")
    
    # This is a simplified example - production code would need:
    # - Proper order sizing
    # - Slippage handling
    # - Error handling
    # - Partial fills
    
    print("\n⚠️  Trade execution is SIMULATED")
    print("   Enable live trading only after extensive paper trading!")

def main():
    print("=" * 60)
    print("🤖 ALPACA AUTO-TRADER - Conservative Strategy")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Get signal
    target_lev, signal_count = get_signal_conservative('SPY')
    
    print(f"Signal Count: {signal_count}/5")
    print(f"Target Position: {target_lev}x")
    print()
    
    # Execute (in paper mode)
    execute_trade(target_lev)

if __name__ == "__main__":
    main()
