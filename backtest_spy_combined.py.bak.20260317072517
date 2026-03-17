#!/usr/bin/env python3
"""
SPY Combined Strategy with Dynamic Leverage
Base: Trend Filter
Leverage up when multiple signals align
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fetch 10 years of SPY data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10 + 250)

print("Fetching SPY data...")
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print(f"Data range: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")

# Calculate indicators
spy['MA20'] = spy['Close'].rolling(window=20).mean()
spy['MA50'] = spy['Close'].rolling(window=50).mean()
spy['MA200'] = spy['Close'].rolling(window=200).mean()
spy['MA200_Slope'] = spy['MA200'].pct_change(periods=20)

# RSI (14-day)
delta = spy['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
spy['RSI'] = 100 - (100 / (1 + rs))

# Volatility (20-day rolling std of returns)
spy['Volatility'] = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

# Drop NaN
spy = spy.dropna()

# Trim to 10 years
ten_years_ago = end_date - timedelta(days=365*10)
spy = spy[spy.index >= ten_years_ago]

print(f"Backtest period: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
print(f"Trading days: {len(spy)}")
print()

# ============================================================
# SIGNAL COMPONENTS
# ============================================================

# Signal 1: Trend Filter (price > 50 DMA AND 200 DMA slope positive)
spy['Sig_TrendFilter'] = ((spy['Close'] > spy['MA50']) & (spy['MA200_Slope'] > 0)).astype(int)

# Signal 2: Golden Cross (50 DMA > 200 DMA)
spy['Sig_GoldenCross'] = (spy['MA50'] > spy['MA200']).astype(int)

# Signal 3: Price Momentum (price > 20 DMA)
spy['Sig_Momentum'] = (spy['Close'] > spy['MA20']).astype(int)

# Signal 4: RSI not overbought (RSI < 70) - avoid buying at extremes
spy['Sig_RSI_OK'] = (spy['RSI'] < 70).astype(int)

# Signal 5: Low volatility regime (vol < 25% annualized)
spy['Sig_LowVol'] = (spy['Volatility'] < 0.25).astype(int)

# ============================================================
# COMBINED STRATEGIES
# ============================================================

# Strategy A: Trend Filter Only (baseline from before)
spy['Position_A'] = spy['Sig_TrendFilter']

# Strategy B: Trend Filter + Leverage on Golden Cross
# 1x when Trend Filter, 1.5x when Trend Filter + Golden Cross
spy['Position_B'] = spy['Sig_TrendFilter'] * 1.0
spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Sig_GoldenCross'] == 1), 'Position_B'] = 1.5

# Strategy C: Full Signal Stack
# 1x base, add 0.25x for each additional signal (max 2x)
spy['Signal_Count'] = (spy['Sig_TrendFilter'] + spy['Sig_GoldenCross'] + 
                       spy['Sig_Momentum'] + spy['Sig_RSI_OK'] + spy['Sig_LowVol'])
spy['Position_C'] = 0.0
spy.loc[spy['Sig_TrendFilter'] == 1, 'Position_C'] = 1.0  # Base position
# Add leverage for additional confirming signals
spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Signal_Count'] >= 4), 'Position_C'] = 1.5
spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Signal_Count'] == 5), 'Position_C'] = 2.0

# Strategy D: Conservative - Require 3+ signals for any position
spy['Position_D'] = 0.0
spy.loc[spy['Signal_Count'] >= 3, 'Position_D'] = 1.0
spy.loc[spy['Signal_Count'] >= 4, 'Position_D'] = 1.5
spy.loc[spy['Signal_Count'] == 5, 'Position_D'] = 2.0

# Strategy E: Aggressive Momentum
# 2x when all signals align, 1x when trend filter only
spy['Position_E'] = spy['Sig_TrendFilter'] * 1.0
spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Sig_GoldenCross'] == 1) & 
        (spy['Sig_Momentum'] == 1), 'Position_E'] = 2.0

# Calculate returns
spy['Daily_Return'] = spy['Close'].pct_change()

spy['Return_BH'] = spy['Daily_Return']
spy['Return_A'] = spy['Position_A'].shift(1) * spy['Daily_Return']
spy['Return_B'] = spy['Position_B'].shift(1) * spy['Daily_Return']
spy['Return_C'] = spy['Position_C'].shift(1) * spy['Daily_Return']
spy['Return_D'] = spy['Position_D'].shift(1) * spy['Daily_Return']
spy['Return_E'] = spy['Position_E'].shift(1) * spy['Daily_Return']

# Cumulative returns
for col in ['BH', 'A', 'B', 'C', 'D', 'E']:
    spy[f'Cum_{col}'] = (1 + spy[f'Return_{col}']).cumprod()

# Metrics function
def calc_metrics(returns, positions=None):
    returns = returns.dropna()
    cum = (1 + returns).cumprod()
    total = cum.iloc[-1] - 1
    years = len(returns) / 252
    ann_ret = (1 + total) ** (1/years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()
    
    calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0
    
    # Average leverage if positions provided
    avg_lev = positions.mean() if positions is not None else 1.0
    
    return {
        'total': total,
        'ann_ret': ann_ret,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'avg_lev': avg_lev
    }

# Calculate all metrics
metrics = {
    'Buy & Hold': calc_metrics(spy['Return_BH']),
    'A: Trend Filter': calc_metrics(spy['Return_A'], spy['Position_A']),
    'B: TF + GC Lev': calc_metrics(spy['Return_B'], spy['Position_B']),
    'C: Signal Stack': calc_metrics(spy['Return_C'], spy['Position_C']),
    'D: Conservative': calc_metrics(spy['Return_D'], spy['Position_D']),
    'E: Aggressive': calc_metrics(spy['Return_E'], spy['Position_E']),
}

# Print results
print("=" * 95)
print("COMBINED STRATEGY BACKTEST: SPY with Dynamic Leverage")
print("=" * 95)
print()
print("STRATEGY DEFINITIONS:")
print("  A: Trend Filter      — 1x when price > 50 DMA AND 200 DMA slope positive")
print("  B: TF + GC Leverage  — 1x Trend Filter, 1.5x when Golden Cross also active")
print("  C: Signal Stack      — 1x base, 1.5x with 4 signals, 2x with all 5 signals")
print("  D: Conservative      — Requires 3+ signals; scales 1x→1.5x→2x")
print("  E: Aggressive        — 1x Trend Filter, 2x when TF + GC + Momentum align")
print()
print("SIGNALS USED:")
print("  1. Trend Filter (Price > 50 DMA & 200 DMA rising)")
print("  2. Golden Cross (50 DMA > 200 DMA)")
print("  3. Momentum (Price > 20 DMA)")
print("  4. RSI OK (RSI < 70)")
print("  5. Low Volatility (Ann. Vol < 25%)")
print()

# Main comparison table
print("-" * 95)
header = f"{'Strategy':<18} {'Total':>10} {'Ann.Ret':>10} {'Vol':>10} {'Sharpe':>8} {'MaxDD':>10} {'Calmar':>8} {'AvgLev':>8}"
print(header)
print("-" * 95)

for name, m in metrics.items():
    print(f"{name:<18} {m['total']*100:>9.1f}% {m['ann_ret']*100:>9.1f}% {m['vol']*100:>9.1f}% {m['sharpe']:>8.2f} {m['max_dd']*100:>9.1f}% {m['calmar']:>8.2f} {m['avg_lev']:>7.2f}x")

print("-" * 95)
print()

# Final values
print("FINAL VALUES (Starting: $10,000):")
initial = 10000
for col, name in [('BH', 'Buy & Hold'), ('A', 'Trend Filter'), 
                   ('B', 'TF + GC Lev'), ('C', 'Signal Stack'),
                   ('D', 'Conservative'), ('E', 'Aggressive')]:
    final = initial * spy[f'Cum_{col}'].iloc[-1]
    print(f"  {name:<18}: ${final:>12,.0f}")
print()

# Position distribution
print("POSITION DISTRIBUTION (% of days at each leverage level):")
print("-" * 70)
for col, name in [('A', 'Trend Filter'), ('B', 'TF + GC Lev'), 
                   ('C', 'Signal Stack'), ('D', 'Conservative'), ('E', 'Aggressive')]:
    pos = spy[f'Position_{col}']
    cash = (pos == 0).sum() / len(pos) * 100
    lev_1 = (pos == 1).sum() / len(pos) * 100
    lev_15 = (pos == 1.5).sum() / len(pos) * 100
    lev_2 = (pos == 2).sum() / len(pos) * 100
    print(f"  {name:<15}: Cash {cash:>5.1f}% | 1x {lev_1:>5.1f}% | 1.5x {lev_15:>5.1f}% | 2x {lev_2:>5.1f}%")
print()

# Yearly returns
print("YEARLY RETURNS:")
print("-" * 95)
spy['Year'] = spy.index.year
yearly = spy.groupby('Year').agg({
    'Return_BH': lambda x: (1 + x).prod() - 1,
    'Return_A': lambda x: (1 + x).prod() - 1,
    'Return_B': lambda x: (1 + x).prod() - 1,
    'Return_C': lambda x: (1 + x).prod() - 1,
    'Return_D': lambda x: (1 + x).prod() - 1,
    'Return_E': lambda x: (1 + x).prod() - 1,
})

print(f"{'Year':<6} {'B&H':>10} {'TrendFlt':>10} {'TF+GC':>10} {'Stack':>10} {'Conserv':>10} {'Aggress':>10}")
print("-" * 95)
for year, row in yearly.iterrows():
    bh = row['Return_BH']*100
    a = row['Return_A']*100
    b = row['Return_B']*100
    c = row['Return_C']*100
    d = row['Return_D']*100
    e = row['Return_E']*100
    print(f"{year:<6} {bh:>9.1f}% {a:>9.1f}% {b:>9.1f}% {c:>9.1f}% {d:>9.1f}% {e:>9.1f}%")
print("-" * 95)
print()

# Drawdown comparison
print("DRAWDOWN ANALYSIS:")
print("-" * 70)
for col, name in [('BH', 'Buy & Hold'), ('A', 'Trend Filter'), 
                   ('B', 'TF + GC Lev'), ('C', 'Signal Stack'),
                   ('D', 'Conservative'), ('E', 'Aggressive')]:
    cum = spy[f'Cum_{col}']
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()
    max_dd_date = dd.idxmin()
    peak_date = cum.loc[:max_dd_date].idxmax()
    
    # Recovery date
    recovery_mask = (cum.index > max_dd_date) & (cum >= rolling_max.loc[max_dd_date])
    if recovery_mask.any():
        recovery_date = cum[recovery_mask].index[0]
        recovery_days = (recovery_date - max_dd_date).days
        recovery_str = f"{recovery_days} days"
    else:
        recovery_str = "Not recovered"
    
    print(f"  {name:<15}: {max_dd*100:>6.1f}% | {peak_date.strftime('%Y-%m-%d')} → {max_dd_date.strftime('%Y-%m-%d')} | Recovery: {recovery_str}")

print()
print("=" * 95)
print("WINNER BY CATEGORY:")
print("=" * 95)

# Find winners
best_return = max(metrics.items(), key=lambda x: x[1]['total'])
best_sharpe = max(metrics.items(), key=lambda x: x[1]['sharpe'])
best_calmar = max(metrics.items(), key=lambda x: x[1]['calmar'])
best_dd = max(metrics.items(), key=lambda x: x[1]['max_dd'])  # max because negative

print(f"  🏆 Highest Return:     {best_return[0]} ({best_return[1]['total']*100:.1f}%)")
print(f"  🏆 Best Sharpe:        {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
print(f"  🏆 Best Calmar:        {best_calmar[0]} ({best_calmar[1]['calmar']:.2f})")
print(f"  🏆 Smallest Drawdown:  {best_dd[0]} ({best_dd[1]['max_dd']*100:.1f}%)")
print()

# Recommendation
print("=" * 95)
print("RECOMMENDATION:")
print("=" * 95)
print()
print("  For GROWTH with managed risk:  Strategy C (Signal Stack)")
print("    → Good returns with intelligent leverage scaling")
print()
print("  For CAPITAL PRESERVATION:      Strategy D (Conservative)")  
print("    → Requires strong confirmation before taking risk")
print()
print("  For AGGRESSIVE GROWTH:         Strategy E (Aggressive)")
print("    → 2x leverage when trend + cross + momentum align")
print()
