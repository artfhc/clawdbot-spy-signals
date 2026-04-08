#!/usr/bin/env python3
"""
SPY Moving Average Strategy Backtest
- Long when price > 50 DMA
- Short when price < 200 DMA
- Flat otherwise (between 50 DMA and 200 DMA when 50 < 200)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fetch 10 years of SPY data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10 + 200)  # Extra days for MA warmup

print("Fetching SPY data...")
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

# Flatten multi-level columns if present
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print(f"Data range: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
print(f"Total trading days: {len(spy)}")

# Calculate moving averages
spy['MA50'] = spy['Close'].rolling(window=50).mean()
spy['MA200'] = spy['Close'].rolling(window=200).mean()

# Drop NaN rows (warmup period)
spy = spy.dropna()

# Trim to exactly 10 years
ten_years_ago = end_date - timedelta(days=365*10)
spy = spy[spy.index >= ten_years_ago]

print(f"Backtest period: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
print(f"Trading days in backtest: {len(spy)}")
print()

# Strategy signals
# Position: 1 = Long, -1 = Short, 0 = Flat
spy['Signal'] = 0
spy.loc[spy['Close'] > spy['MA50'], 'Signal'] = 1   # Long above 50 DMA
spy.loc[spy['Close'] < spy['MA200'], 'Signal'] = -1  # Short below 200 DMA

# Calculate daily returns
spy['Daily_Return'] = spy['Close'].pct_change()

# Strategy returns (position from previous day applied to today's return)
spy['Strategy_Return'] = spy['Signal'].shift(1) * spy['Daily_Return']

# Buy and hold returns
spy['BuyHold_Cumulative'] = (1 + spy['Daily_Return']).cumprod()
spy['Strategy_Cumulative'] = (1 + spy['Strategy_Return']).cumprod()

# Calculate metrics
def calculate_metrics(returns, name):
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    # Annualized return
    years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1/years) - 1
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    return {
        'name': name,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

# Get metrics
strategy_metrics = calculate_metrics(spy['Strategy_Return'].dropna(), 'MA Strategy')
buyhold_metrics = calculate_metrics(spy['Daily_Return'].dropna(), 'Buy & Hold')

# Count positions
long_days = (spy['Signal'] == 1).sum()
short_days = (spy['Signal'] == -1).sum()
flat_days = (spy['Signal'] == 0).sum()
total_days = len(spy)

# Print results
print("=" * 60)
print("BACKTEST RESULTS: SPY Moving Average Strategy")
print("Strategy: Long > 50 DMA | Short < 200 DMA | Flat otherwise")
print("=" * 60)
print()

print("POSITION BREAKDOWN:")
print(f"  Long days:   {long_days:,} ({100*long_days/total_days:.1f}%)")
print(f"  Short days:  {short_days:,} ({100*short_days/total_days:.1f}%)")
print(f"  Flat days:   {flat_days:,} ({100*flat_days/total_days:.1f}%)")
print()

print("PERFORMANCE COMPARISON:")
print("-" * 60)
print(f"{'Metric':<25} {'MA Strategy':>15} {'Buy & Hold':>15}")
print("-" * 60)
print(f"{'Total Return':<25} {strategy_metrics['total_return']*100:>14.1f}% {buyhold_metrics['total_return']*100:>14.1f}%")
print(f"{'Annualized Return':<25} {strategy_metrics['annualized_return']*100:>14.1f}% {buyhold_metrics['annualized_return']*100:>14.1f}%")
print(f"{'Volatility (Ann.)':<25} {strategy_metrics['volatility']*100:>14.1f}% {buyhold_metrics['volatility']*100:>14.1f}%")
print(f"{'Sharpe Ratio':<25} {strategy_metrics['sharpe']:>15.2f} {buyhold_metrics['sharpe']:>15.2f}")
print(f"{'Max Drawdown':<25} {strategy_metrics['max_drawdown']*100:>14.1f}% {buyhold_metrics['max_drawdown']*100:>14.1f}%")
print(f"{'Win Rate (daily)':<25} {strategy_metrics['win_rate']*100:>14.1f}% {buyhold_metrics['win_rate']*100:>14.1f}%")
print("-" * 60)
print()

# Final values (starting with $10,000)
initial = 10000
final_strategy = initial * spy['Strategy_Cumulative'].iloc[-1]
final_buyhold = initial * spy['BuyHold_Cumulative'].iloc[-1]

print("FINAL VALUES (Starting: $10,000):")
print(f"  MA Strategy:  ${final_strategy:,.0f}")
print(f"  Buy & Hold:   ${final_buyhold:,.0f}")
print()

# Yearly breakdown
print("YEARLY RETURNS:")
print("-" * 45)
spy['Year'] = spy.index.year
yearly = spy.groupby('Year').agg({
    'Strategy_Return': lambda x: (1 + x).prod() - 1,
    'Daily_Return': lambda x: (1 + x).prod() - 1
})
yearly.columns = ['Strategy', 'Buy&Hold']

print(f"{'Year':<10} {'Strategy':>15} {'Buy & Hold':>15}")
print("-" * 45)
for year, row in yearly.iterrows():
    print(f"{year:<10} {row['Strategy']*100:>14.1f}% {row['Buy&Hold']*100:>14.1f}%")
print("-" * 45)
print()

# Short performance specifically
short_returns = spy[spy['Signal'].shift(1) == -1]['Strategy_Return']
if len(short_returns) > 0:
    short_total = (1 + short_returns).prod() - 1
    print(f"SHORT POSITION PERFORMANCE:")
    print(f"  Total return from shorts: {short_total*100:.1f}%")
    print(f"  Avg daily return (short): {short_returns.mean()*100:.3f}%")
    print(f"  Short win rate: {(short_returns > 0).sum() / len(short_returns) * 100:.1f}%")
