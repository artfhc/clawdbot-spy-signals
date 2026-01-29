#!/usr/bin/env python3
"""
Signal Alerts - Detects when strategy signals change and saves state.
Run daily via cron job. Outputs changes for alerting.
"""

import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent / "signal_state.json"

def load_data():
    """Load SPY and international data."""
    spy = yf.download('SPY', period='1y', progress=False)
    veu = yf.download('VEU', period='1y', progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(veu.columns, pd.MultiIndex):
        veu.columns = veu.columns.get_level_values(0)
    
    # Calculate indicators
    spy['MA20'] = spy['Close'].rolling(20).mean()
    spy['MA50'] = spy['Close'].rolling(50).mean()
    spy['MA200'] = spy['Close'].rolling(200).mean()
    spy['MA200_Slope'] = spy['MA200'].pct_change(periods=20)
    
    delta = spy['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    spy['RSI'] = 100 - (100 / (1 + gain/loss))
    
    gain5 = delta.where(delta > 0, 0).rolling(5).mean()
    loss5 = (-delta.where(delta < 0, 0)).rolling(5).mean()
    spy['RSI5'] = 100 - (100 / (1 + gain5/loss5))
    
    spy['Vol'] = spy['Close'].pct_change().rolling(20).std() * (252 ** 0.5)
    spy['Mom12M'] = spy['Close'].pct_change(periods=252)
    
    veu['Mom12M'] = veu['Close'].pct_change(periods=252)
    
    return spy, veu

def get_all_signals(spy, veu):
    """Get current signals for all strategies."""
    row = spy.iloc[-1]
    
    # Base signals
    tf = (row['Close'] > row['MA50']) and (row['MA200_Slope'] > 0)
    gc = row['MA50'] > row['MA200']
    mom = row['Close'] > row['MA20']
    rsi_ok = row['RSI'] < 70
    low_vol = row['Vol'] < 0.25
    count = sum([tf, gc, mom, rsi_ok, low_vol])
    
    above_200 = row['Close'] > row['MA200']
    rsi5 = row['RSI5']
    rsi_oversold = rsi5 < 30
    vol = row['Vol']
    
    spy_mom = row['Mom12M'] if pd.notna(row['Mom12M']) else 0
    veu_mom = veu['Mom12M'].iloc[-1] if len(veu) > 0 and pd.notna(veu['Mom12M'].iloc[-1]) else 0
    
    signals = {}
    
    # Original strategies
    signals['Trend Rider'] = 'LONG 1x' if tf else 'CASH'
    signals['Golden Boost'] = 'LONG 1.5x' if (tf and gc) else ('LONG 1x' if tf else 'CASH')
    
    if count >= 5:
        signals['Signal Stacker'] = 'LONG 2x'
    elif count >= 4:
        signals['Signal Stacker'] = 'LONG 1.5x'
    elif count >= 3:
        signals['Signal Stacker'] = 'LONG 1x'
    else:
        signals['Signal Stacker'] = 'CASH'
    
    signals['Steady Eddie'] = signals['Signal Stacker']  # Same logic
    signals['Full Send'] = 'LONG 2x' if (tf and gc and mom) else 'CASH'
    
    # Research strategies
    signals['200 DMA Monthly'] = 'LONG 1x' if above_200 else 'CASH'
    signals['RSI Bounce'] = 'BUY SIGNAL' if (above_200 and rsi_oversold) else 'CASH'
    
    if spy_mom > veu_mom and spy_mom > 0:
        signals['Dual Momentum'] = 'LONG SPY'
    elif veu_mom > spy_mom and veu_mom > 0:
        signals['Dual Momentum'] = 'LONG VEU'
    else:
        signals['Dual Momentum'] = 'BONDS'
    
    # Hybrid strategies
    if count >= 4 and above_200:
        signals['Voting System'] = 'LONG 1.5x'
    elif count >= 3 and above_200:
        signals['Voting System'] = 'LONG 1x'
    else:
        signals['Voting System'] = 'CASH'
    
    if tf and gc:
        if vol < 0.15:
            signals['Vol Adaptive'] = 'LONG 2x'
        elif vol < 0.20:
            signals['Vol Adaptive'] = 'LONG 1.5x'
        elif vol < 0.25:
            signals['Vol Adaptive'] = 'LONG 1x'
        else:
            signals['Vol Adaptive'] = 'LONG 0.5x'
    elif above_200:
        signals['Vol Adaptive'] = 'LONG 0.5x' if vol < 0.25 else 'CASH'
    else:
        signals['Vol Adaptive'] = 'CASH'
    
    # Ensemble
    trend_pos = tf
    rsi_pos = above_200 and rsi_oversold
    if trend_pos and rsi_pos:
        signals['Ensemble'] = 'LONG 1.5x'
    elif trend_pos or rsi_pos:
        signals['Ensemble'] = 'LONG 1x'
    else:
        signals['Ensemble'] = 'CASH'
    
    return signals, float(row['Close'])

def load_previous_state():
    """Load previous signal state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(signals, price):
    """Save current signal state."""
    state = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'price': price,
        'signals': signals
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def detect_changes(old_signals, new_signals):
    """Detect which signals have changed."""
    changes = []
    for strategy, new_signal in new_signals.items():
        old_signal = old_signals.get(strategy, '')
        if old_signal and old_signal != new_signal:
            changes.append({
                'strategy': strategy,
                'from': old_signal,
                'to': new_signal
            })
    return changes

def main():
    print("Loading market data...")
    spy, veu = load_data()
    
    print("Calculating signals...")
    current_signals, price = get_all_signals(spy, veu)
    
    print("Checking for changes...")
    previous = load_previous_state()
    old_signals = previous.get('signals', {})
    
    changes = detect_changes(old_signals, current_signals)
    
    # Save new state
    save_state(current_signals, price)
    
    # Output results
    print(f"\n{'='*50}")
    print(f"📊 SIGNAL CHECK - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"SPY Price: ${price:.2f}")
    print(f"{'='*50}")
    
    if changes:
        print(f"\n🚨 {len(changes)} SIGNAL CHANGE(S) DETECTED:\n")
        for change in changes:
            print(f"  ⚡ {change['strategy']}")
            print(f"     {change['from']} → {change['to']}")
            print()
    else:
        print("\n✅ No signal changes since last check.")
    
    print("\n📋 CURRENT SIGNALS:")
    for strategy, signal in current_signals.items():
        emoji = '🟢' if 'LONG' in signal or 'BUY' in signal else '⚪'
        print(f"  {emoji} {strategy}: {signal}")
    
    # Return changes for external use (e.g., alerting)
    return changes, current_signals, price

if __name__ == "__main__":
    changes, signals, price = main()
