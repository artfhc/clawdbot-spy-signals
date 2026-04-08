#!/usr/bin/env python3
"""
SPY Moving Average Strategy Backtest - 3 Improved Variants
1. Long only > 50 DMA, cash otherwise
2. Golden Cross / Death Cross (50/200 crossover)
3. Trend filter - long only when 200 DMA slope is positive
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fetch 10 years of SPY data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10 + 250)  # Extra days for MA warmup

print("Fetching SPY data...")
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

# Flatten multi-level columns if present
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print(f"Data range: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")

# Calculate moving averages
spy['MA50'] = spy['Close'].rolling(window=50).mean()
spy['MA200'] = spy['Close'].rolling(window=200).mean()

# Calculate 200 DMA slope (20-day rate of change of 200 DMA)
spy['MA200_Slope'] = spy['MA200'].pct_change(periods=20)

# Drop NaN rows (warmup period)
spy = spy.dropna()

# Trim to exactly 10 years
ten_years_ago = end_date - timedelta(days=365*10)
spy = spy[spy.index >= ten_years_ago]

print(f"Backtest period: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
print(f"Trading days in backtest: {len(spy)}")
print()

# ============================================================
# STRATEGY 1: Long only > 50 DMA, cash otherwise
# ============================================================
spy['Signal_1'] = 0
spy.loc[spy['Close'] > spy['MA50'], 'Signal_1'] = 1

# ============================================================
# STRATEGY 2: Golden Cross / Death Cross
# Long when 50 DMA > 200 DMA, cash otherwise
# ============================================================
spy['Signal_2'] = 0
spy.loc[spy['MA50'] > spy['MA200'], 'Signal_2'] = 1

# ============================================================
# STRATEGY 3: Trend Filter
# Long when: Close > 50 DMA AND 200 DMA slope is positive
# ============================================================
spy['Signal_3'] = 0
spy.loc[(spy['Close'] > spy['MA50']) & (spy['MA200_Slope'] > 0), 'Signal_3'] = 1

# Calculate daily returns
spy['Daily_Return'] = spy['Close'].pct_change()

# Strategy returns (position from previous day applied to today's return)
spy['Strategy_1_Return'] = spy['Signal_1'].shift(1) * spy['Daily_Return']
spy['Strategy_2_Return'] = spy['Signal_2'].shift(1) * spy['Daily_Return']
spy['Strategy_3_Return'] = spy['Signal_3'].shift(1) * spy['Daily_Return']

# Cumulative returns
spy['BuyHold_Cum'] = (1 + spy['Daily_Return']).cumprod()
spy['Strategy_1_Cum'] = (1 + spy['Strategy_1_Return']).cumprod()
spy['Strategy_2_Cum'] = (1 + spy['Strategy_2_Return']).cumprod()
spy['Strategy_3_Cum'] = (1 + spy['Strategy_3_Return']).cumprod()

# Calculate metrics function
def calculate_metrics(returns, name):
    returns = returns.dropna()
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1/years) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calculate number of trades (signal changes)
    return {
        'name': name,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
    }

# Get metrics for all strategies
metrics_bh = calculate_metrics(spy['Daily_Return'], 'Buy & Hold')
metrics_1 = calculate_metrics(spy['Strategy_1_Return'], 'Long > 50 DMA')
metrics_2 = calculate_metrics(spy['Strategy_2_Return'], 'Golden Cross')
metrics_3 = calculate_metrics(spy['Strategy_3_Return'], 'Trend Filter')

# Count exposure (days in market)
exposure_1 = (spy['Signal_1'] == 1).sum() / len(spy) * 100
exposure_2 = (spy['Signal_2'] == 1).sum() / len(spy) * 100
exposure_3 = (spy['Signal_3'] == 1).sum() / len(spy) * 100

# Count trades (signal changes)
trades_1 = (spy['Signal_1'].diff() != 0).sum()
trades_2 = (spy['Signal_2'].diff() != 0).sum()
trades_3 = (spy['Signal_3'].diff() != 0).sum()

# Print results
print("=" * 80)
print("BACKTEST RESULTS: SPY Moving Average Strategies (Improved)")
print("=" * 80)
print()

print("STRATEGY DESCRIPTIONS:")
print("  1. Long > 50 DMA    : Buy when price above 50 DMA, cash otherwise")
print("  2. Golden Cross     : Buy when 50 DMA > 200 DMA (crossover)")
print("  3. Trend Filter     : Buy when price > 50 DMA AND 200 DMA slope is positive")
print()

print("-" * 80)
print(f"{'Metric':<22} {'Buy & Hold':>13} {'Long>50DMA':>13} {'GoldenCross':>13} {'TrendFilter':>13}")
print("-" * 80)
print(f"{'Total Return':<22} {metrics_bh['total_return']*100:>12.1f}% {metrics_1['total_return']*100:>12.1f}% {metrics_2['total_return']*100:>12.1f}% {metrics_3['total_return']*100:>12.1f}%")
print(f"{'Annualized Return':<22} {metrics_bh['annualized_return']*100:>12.1f}% {metrics_1['annualized_return']*100:>12.1f}% {metrics_2['annualized_return']*100:>12.1f}% {metrics_3['annualized_return']*100:>12.1f}%")
print(f"{'Volatility (Ann.)':<22} {metrics_bh['volatility']*100:>12.1f}% {metrics_1['volatility']*100:>12.1f}% {metrics_2['volatility']*100:>12.1f}% {metrics_3['volatility']*100:>12.1f}%")
print(f"{'Sharpe Ratio':<22} {metrics_bh['sharpe']:>13.2f} {metrics_1['sharpe']:>13.2f} {metrics_2['sharpe']:>13.2f} {metrics_3['sharpe']:>13.2f}")
print(f"{'Max Drawdown':<22} {metrics_bh['max_drawdown']*100:>12.1f}% {metrics_1['max_drawdown']*100:>12.1f}% {metrics_2['max_drawdown']*100:>12.1f}% {metrics_3['max_drawdown']*100:>12.1f}%")
print(f"{'Market Exposure':<22} {'100.0%':>13} {exposure_1:>12.1f}% {exposure_2:>12.1f}% {exposure_3:>12.1f}%")
print(f"{'# of Trades':<22} {'—':>13} {trades_1:>13} {trades_2:>13} {trades_3:>13}")
print("-" * 80)
print()

# Risk-adjusted metrics
print("RISK-ADJUSTED PERFORMANCE:")
print("-" * 80)

# Return per unit of drawdown
ret_dd_bh = abs(metrics_bh['annualized_return'] / metrics_bh['max_drawdown'])
ret_dd_1 = abs(metrics_1['annualized_return'] / metrics_1['max_drawdown'])
ret_dd_2 = abs(metrics_2['annualized_return'] / metrics_2['max_drawdown'])
ret_dd_3 = abs(metrics_3['annualized_return'] / metrics_3['max_drawdown'])

print(f"{'Return/MaxDD Ratio':<22} {ret_dd_bh:>13.2f} {ret_dd_1:>13.2f} {ret_dd_2:>13.2f} {ret_dd_3:>13.2f}")

# Calmar Ratio (annualized return / max drawdown)
print(f"{'Calmar Ratio':<22} {ret_dd_bh:>13.2f} {ret_dd_1:>13.2f} {ret_dd_2:>13.2f} {ret_dd_3:>13.2f}")
print("-" * 80)
print()

# Final values
initial = 10000
print("FINAL VALUES (Starting: $10,000):")
print(f"  Buy & Hold:      ${initial * spy['BuyHold_Cum'].iloc[-1]:>12,.0f}")
print(f"  Long > 50 DMA:   ${initial * spy['Strategy_1_Cum'].iloc[-1]:>12,.0f}")
print(f"  Golden Cross:    ${initial * spy['Strategy_2_Cum'].iloc[-1]:>12,.0f}")
print(f"  Trend Filter:    ${initial * spy['Strategy_3_Cum'].iloc[-1]:>12,.0f}")
print()

# Yearly breakdown
print("YEARLY RETURNS:")
print("-" * 80)
spy['Year'] = spy.index.year
yearly = spy.groupby('Year').agg({
    'Daily_Return': lambda x: (1 + x).prod() - 1,
    'Strategy_1_Return': lambda x: (1 + x).prod() - 1,
    'Strategy_2_Return': lambda x: (1 + x).prod() - 1,
    'Strategy_3_Return': lambda x: (1 + x).prod() - 1,
})
yearly.columns = ['Buy&Hold', 'Long>50DMA', 'GoldenCross', 'TrendFilter']

print(f"{'Year':<8} {'Buy&Hold':>12} {'Long>50DMA':>12} {'GoldenCross':>12} {'TrendFilter':>12}")
print("-" * 80)
for year, row in yearly.iterrows():
    # Highlight winning strategy each year
    best = max(row['Long>50DMA'], row['GoldenCross'], row['TrendFilter'])
    bh = row['Buy&Hold']
    l50 = row['Long>50DMA']
    gc = row['GoldenCross']
    tf = row['TrendFilter']
    
    # Mark with * if strategy beat buy & hold
    l50_mark = '*' if l50 > bh else ' '
    gc_mark = '*' if gc > bh else ' '
    tf_mark = '*' if tf > bh else ' '
    
    print(f"{year:<8} {bh*100:>11.1f}% {l50*100:>10.1f}%{l50_mark} {gc*100:>10.1f}%{gc_mark} {tf*100:>10.1f}%{tf_mark}")
print("-" * 80)
print("* = Beat Buy & Hold that year")
print()

# Drawdown analysis
print("WORST DRAWDOWN PERIODS:")
print("-" * 60)

for name, cum_col in [('Buy & Hold', 'BuyHold_Cum'), 
                       ('Long > 50 DMA', 'Strategy_1_Cum'),
                       ('Golden Cross', 'Strategy_2_Cum'),
                       ('Trend Filter', 'Strategy_3_Cum')]:
    rolling_max = spy[cum_col].cummax()
    drawdown = (spy[cum_col] - rolling_max) / rolling_max
    worst_dd = drawdown.min()
    worst_dd_date = drawdown.idxmin()
    
    # Find peak before worst drawdown
    peak_date = spy.loc[:worst_dd_date, cum_col].idxmax()
    
    print(f"  {name:<15}: {worst_dd*100:>6.1f}% (Peak: {peak_date.strftime('%Y-%m-%d')} → Trough: {worst_dd_date.strftime('%Y-%m-%d')})")

print()
print("=" * 80)
print("SUMMARY: Best Strategy by Metric")
print("=" * 80)
strategies = ['Buy&Hold', 'Long>50DMA', 'GoldenCross', 'TrendFilter']
all_metrics = [metrics_bh, metrics_1, metrics_2, metrics_3]

# Best total return
best_ret = max(all_metrics, key=lambda x: x['total_return'])
print(f"  Highest Return:      {best_ret['name']}")

# Best Sharpe
best_sharpe = max(all_metrics, key=lambda x: x['sharpe'])
print(f"  Best Sharpe Ratio:   {best_sharpe['name']}")

# Smallest drawdown
best_dd = max(all_metrics, key=lambda x: x['max_drawdown'])  # max because DD is negative
print(f"  Smallest Drawdown:   {best_dd['name']}")

# Best risk-adjusted (Calmar)
calmars = [(m['name'], abs(m['annualized_return']/m['max_drawdown'])) for m in all_metrics]
best_calmar = max(calmars, key=lambda x: x[1])
print(f"  Best Calmar Ratio:   {best_calmar[0]}")
