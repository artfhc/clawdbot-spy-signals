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

def fetch_spy_data(start_date, end_date):
    """
    Fetch 10 years of SPY data.
    
    Args:
        start_date (datetime): Start date of the data range.
        end_date (datetime): End date of the data range.
    
    Returns:
        pd.DataFrame: SPY data.
    """
    print("Fetching SPY data...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    return spy

def flatten_columns(spy):
    """
    Flatten multi-level columns if present.
    
    Args:
        spy (pd.DataFrame): DataFrame with potential multi-level columns.
    
    Returns:
        pd.DataFrame: DataFrame with flattened columns.
    """
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy

def calculate_moving_averages(spy):
    """
    Calculate moving averages.
    
    Args:
        spy (pd.DataFrame): DataFrame with Close column.
    
    Returns:
        pd.DataFrame: DataFrame with MA50 and MA200 columns.
    """
    spy['MA50'] = spy['Close'].rolling(window=50).mean()
    spy['MA200'] = spy['Close'].rolling(window=200).mean()
    return spy

def calculate_dma_slope(spy):
    """
    Calculate 200 DMA slope (20-day rate of change of 200 DMA).
    
    Args:
        spy (pd.DataFrame): DataFrame with MA200 column.
    
    Returns:
        pd.DataFrame: DataFrame with MA200_Slope column.
    """
    spy['MA200_Slope'] = spy['MA200'].pct_change(periods=20)
    return spy

def trim_data(spy, start_date):
    """
    Trim data to exactly 10 years.
    
    Args:
        spy (pd.DataFrame): DataFrame to trim.
        start_date (datetime): Start date of the trimmed data range.
    
    Returns:
        pd.DataFrame: Trimmed DataFrame.
    """
    spy = spy.dropna()
    spy = spy[spy.index >= start_date]
    return spy

def generate_signals(spy):
    """
    Generate signals for the strategies.
    
    Args:
        spy (pd.DataFrame): DataFrame with Close, MA50, MA200, and MA200_Slope columns.
    
    Returns:
        pd.DataFrame: DataFrame with Signal_1, Signal_2, and Signal_3 columns.
    """
    spy['Signal_1'] = 0
    spy.loc[(spy['Close'] > spy['MA50']), 'Signal_1'] = 1
    
    spy['Signal_2'] = 0
    spy.loc[(spy['MA50'] > spy['MA200']), 'Signal_2'] = 1
    
    spy['Signal_3'] = 0
    spy.loc[(spy['Close'] > spy['MA50']) & (spy['MA200_Slope'] > 0), 'Signal_3'] = 1
    return spy

def calculate_returns(spy):
    """
    Calculate daily returns and strategy returns.
    
    Args:
        spy (pd.DataFrame): DataFrame with Signal_1, Signal_2, and Signal_3 columns.
    
    Returns:
        pd.DataFrame: DataFrame with Daily_Return, Strategy_1_Return, Strategy_2_Return, and Strategy_3_Return columns.
    """
    spy['Daily_Return'] = spy['Close'].pct_change()
    
    spy['Strategy_1_Return'] = spy['Signal_1'].shift(1) * spy['Daily_Return']
    spy['Strategy_2_Return'] = spy['Signal_2'].shift(1) * spy['Daily_Return']
    spy['Strategy_3_Return'] = spy['Signal_3'].shift(1) * spy['Daily_Return']
    return spy

def calculate_cumulative_returns(spy):
    """
    Calculate cumulative returns.
    
    Args:
        spy (pd.DataFrame): DataFrame with Daily_Return, Strategy_1_Return, Strategy_2_Return, and Strategy_3_Return columns.
    
    Returns:
        pd.DataFrame: DataFrame with BuyHold_Cum, Strategy_1_Cum, Strategy_2_Cum, and Strategy_3_Cum columns.
    """
    spy['BuyHold_Cum'] = (1 + spy['Daily_Return']).cumprod()
    spy['Strategy_1_Cum'] = (1 + spy['Strategy_1_Return']).cumprod()
    spy['Strategy_2_Cum'] = (1 + spy['Strategy_2_Return']).cumprod()
    spy['Strategy_3_Cum'] = (1 + spy['Strategy_3_Return']).cumprod()
    return spy

def calculate_metrics(returns, name):
    """
    Calculate metrics for a given return series.
    
    Args:
        returns (pd.Series): Return series.
        name (str): Name of the strategy.
    
    Returns:
        dict: Dictionary with metrics.
    """
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
    
    return {
        'name': name,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
    }

def get_exposure(spy, signal):
    """
    Calculate exposure for a given signal.
    
    Args:
        spy (pd.DataFrame): DataFrame with signal column.
        signal (str): Name of the signal.
    
    Returns:
        float: Exposure as a percentage.
    """
    return (spy[signal] == 1).sum() / len(spy) * 100

def get_trades(spy, signal):
    """
    Calculate number of trades for a given signal.
    
    Args:
        spy (pd.DataFrame): DataFrame with signal column.
        signal (str): Name of the signal.
    
    Returns:
        int: Number of trades.
    """
    return (spy[signal].diff() != 0).sum()

def print_results(metrics_bh, metrics_1, metrics_2, metrics_3, exposure_1, exposure_2, exposure_3, trades_1, trades_2, trades_3):
    """
    Print results.
    
    Args:
        metrics_bh (dict): Metrics for Buy & Hold.
        metrics_1 (dict): Metrics for Long > 50 DMA.
        metrics_2 (dict): Metrics for Golden Cross.
        metrics_3 (dict): Metrics for Trend Filter.
        exposure_1 (float): Exposure for Long > 50 DMA.
        exposure_2 (float): Exposure for Golden Cross.
        exposure_3 (float): Exposure for Trend Filter.
        trades_1 (int): Number of trades for Long > 50 DMA.
        trades_2 (int): Number of trades for Golden Cross.
        trades_3 (int): Number of trades for Trend Filter.
    """
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
    print(f"{'Sharpe Ratio':<22}