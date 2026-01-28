#!/usr/bin/env python3
"""
Daily Signal Checker for ALL 5 SPY Trading Strategies
Run this each morning before market open to get your trading signals.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def get_signals(ticker='SPY'):
    """Calculate all signals and return position for all 5 strategies."""
    
    # Fetch data
    data = yf.download(ticker, period='1y', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Calculate indicators
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    data['MA200_Slope'] = data['MA200'].pct_change(periods=20)
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    data['Vol'] = data['Close'].pct_change().rolling(20).std() * (252 ** 0.5)
    
    # Get latest values
    latest = data.iloc[-1]
    price = latest['Close']
    ma20 = latest['MA20']
    ma50 = latest['MA50']
    ma200 = latest['MA200']
    ma200_slope = latest['MA200_Slope']
    rsi = latest['RSI']
    vol = latest['Vol']
    
    # Calculate 5 base signals
    sig_trend = (price > ma50) and (ma200_slope > 0)
    sig_golden = ma50 > ma200
    sig_momentum = price > ma20
    sig_rsi = rsi < 70
    sig_lowvol = vol < 0.25
    
    signal_count = sum([sig_trend, sig_golden, sig_momentum, sig_rsi, sig_lowvol])
    
    # ============================================================
    # ALL 5 STRATEGIES
    # ============================================================
    
    strategies = {}
    
    # Strategy A: Trend Filter
    # Long when price > 50 DMA AND 200 DMA slope positive
    if sig_trend:
        strategies['A'] = {'position': 1.0, 'action': 'LONG 1x', 'etf': 'SPY'}
    else:
        strategies['A'] = {'position': 0.0, 'action': 'CASH', 'etf': 'SGOV/BIL'}
    
    # Strategy B: TF + Golden Cross Leverage
    # 1x when Trend Filter, 1.5x when Golden Cross also active
    if sig_trend:
        if sig_golden:
            strategies['B'] = {'position': 1.5, 'action': 'LONG 1.5x', 'etf': '50% SPY + 50% SSO'}
        else:
            strategies['B'] = {'position': 1.0, 'action': 'LONG 1x', 'etf': 'SPY'}
    else:
        strategies['B'] = {'position': 0.0, 'action': 'CASH', 'etf': 'SGOV/BIL'}
    
    # Strategy C: Signal Stack
    # 1x base (requires Trend Filter), 1.5x with 4 signals, 2x with 5 signals
    if sig_trend:
        if signal_count == 5:
            strategies['C'] = {'position': 2.0, 'action': 'LONG 2x', 'etf': 'SSO'}
        elif signal_count >= 4:
            strategies['C'] = {'position': 1.5, 'action': 'LONG 1.5x', 'etf': '50% SPY + 50% SSO'}
        else:
            strategies['C'] = {'position': 1.0, 'action': 'LONG 1x', 'etf': 'SPY'}
    else:
        strategies['C'] = {'position': 0.0, 'action': 'CASH', 'etf': 'SGOV/BIL'}
    
    # Strategy D: Conservative
    # Requires 3+ signals to enter; scales 1x→1.5x→2x
    if signal_count == 5:
        strategies['D'] = {'position': 2.0, 'action': 'LONG 2x', 'etf': 'SSO'}
    elif signal_count >= 4:
        strategies['D'] = {'position': 1.5, 'action': 'LONG 1.5x', 'etf': '50% SPY + 50% SSO'}
    elif signal_count >= 3:
        strategies['D'] = {'position': 1.0, 'action': 'LONG 1x', 'etf': 'SPY'}
    else:
        strategies['D'] = {'position': 0.0, 'action': 'CASH', 'etf': 'SGOV/BIL'}
    
    # Strategy E: Aggressive
    # 1x Trend Filter, 2x when TF + Golden Cross + Momentum align
    if sig_trend:
        if sig_golden and sig_momentum:
            strategies['E'] = {'position': 2.0, 'action': 'LONG 2x', 'etf': 'SSO'}
        else:
            strategies['E'] = {'position': 1.0, 'action': 'LONG 1x', 'etf': 'SPY'}
    else:
        strategies['E'] = {'position': 0.0, 'action': 'CASH', 'etf': 'SGOV/BIL'}
    
    return {
        'date': data.index[-1].strftime('%Y-%m-%d'),
        'ticker': ticker,
        'price': price,
        'ma20': ma20,
        'ma50': ma50,
        'ma200': ma200,
        'ma200_slope': ma200_slope,
        'rsi': rsi,
        'volatility': vol,
        'signals': {
            '1. Trend Filter': sig_trend,
            '2. Golden Cross': sig_golden,
            '3. Momentum': sig_momentum,
            '4. RSI OK': sig_rsi,
            '5. Low Vol': sig_lowvol,
        },
        'signal_count': signal_count,
        'strategies': strategies
    }

def print_report(result):
    """Print formatted daily report with all 5 strategies."""
    
    print("=" * 55)
    print(f"📊 DAILY TRADING SIGNALS - ALL 5 STRATEGIES")
    print(f"   {result['ticker']} | {result['date']}")
    print("=" * 55)
    print()
    
    # Price info
    print(f"💰 PRICE: ${result['price']:.2f}")
    print(f"   20 DMA: ${result['ma20']:.2f}")
    print(f"   50 DMA: ${result['ma50']:.2f}")
    print(f"   200 DMA: ${result['ma200']:.2f}")
    print(f"   RSI: {result['rsi']:.1f} | Vol: {result['volatility']*100:.1f}%")
    print()
    
    # Signal status
    print("📡 SIGNAL STATUS:")
    for name, status in result['signals'].items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {name}")
    print(f"\n   Total: {result['signal_count']}/5", end="")
    if result['signal_count'] == 5:
        print(" 🔥")
    elif result['signal_count'] >= 4:
        print(" 👍")
    elif result['signal_count'] >= 3:
        print(" 👌")
    else:
        print(" ⚠️")
    print()
    
    # All 5 strategies
    print("=" * 55)
    print("📈 STRATEGY SIGNALS")
    print("=" * 55)
    
    strategy_names = {
        'A': 'Trend Filter',
        'B': 'TF + GC Leverage', 
        'C': 'Signal Stack',
        'D': 'Conservative ⭐',
        'E': 'Aggressive'
    }
    
    strategy_desc = {
        'A': 'Long when Price>50DMA & 200DMA rising',
        'B': '1x base, 1.5x when Golden Cross adds',
        'C': '1x→1.5x→2x based on signal count',
        'D': 'Requires 3+ signals; best risk-adjusted',
        'E': '2x when Trend+Cross+Momentum align'
    }
    
    for key in ['A', 'B', 'C', 'D', 'E']:
        s = result['strategies'][key]
        name = strategy_names[key]
        desc = strategy_desc[key]
        
        # Position emoji
        if s['position'] == 0:
            pos_icon = "🔴"
        elif s['position'] == 1.0:
            pos_icon = "🟢"
        elif s['position'] == 1.5:
            pos_icon = "🟡"
        else:
            pos_icon = "🔥"
        
        print()
        print(f"{pos_icon} Strategy {key}: {name}")
        print(f"   {desc}")
        print(f"   ➤ Position: {s['action']}")
        print(f"   ➤ ETF: {s['etf']}")
    
    print()
    print("=" * 55)
    print("💡 QUICK REFERENCE")
    print("=" * 55)
    print("   CASH    = SGOV, BIL, or money market")
    print("   1x      = 100% SPY")
    print("   1.5x    = 50% SPY + 50% SSO")
    print("   2x      = 100% SSO")
    print()

def print_mobile_summary(result):
    """Print compact mobile-friendly summary."""
    
    print(f"📊 SPY Signals | {result['date']}")
    print(f"Price: ${result['price']:.2f}")
    print()
    
    # Signals one-liner
    sigs = result['signals']
    sig_str = ""
    sig_str += "✅" if sigs['1. Trend Filter'] else "❌"
    sig_str += "✅" if sigs['2. Golden Cross'] else "❌"
    sig_str += "✅" if sigs['3. Momentum'] else "❌"
    sig_str += "✅" if sigs['4. RSI OK'] else "❌"
    sig_str += "✅" if sigs['5. Low Vol'] else "❌"
    print(f"Signals: {sig_str} ({result['signal_count']}/5)")
    print()
    
    print("Strategy Positions:")
    for key in ['A', 'B', 'C', 'D', 'E']:
        s = result['strategies'][key]
        star = "⭐" if key == 'D' else ""
        print(f"  {key}: {s['action']:<10} {star}")
    
    print()
    # Highlight recommended (D)
    d = result['strategies']['D']
    print(f"💡 Recommended (D): {d['action']}")
    print(f"   Buy: {d['etf']}")

if __name__ == "__main__":
    import sys
    
    result = get_signals('SPY')
    
    # Check for --mobile flag
    if len(sys.argv) > 1 and sys.argv[1] == '--mobile':
        print_mobile_summary(result)
    else:
        print_report(result)
