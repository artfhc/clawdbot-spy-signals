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

def fetch_spy_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
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
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    print(f"Data range: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
    return spy

def calculate_indicators(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate indicators.

    Args:
        spy (pd.DataFrame): SPY data.

    Returns:
        pd.DataFrame: SPY data with indicators.
    """
    spy['MA20'] = spy['Close'].rolling(window=20).mean()
    spy['MA50'] = spy['Close'].rolling(window=50).mean()
    spy['MA200'] = spy['Close'].rolling(window=200).mean()
    spy['MA200_Slope'] = spy['MA200'].pct_change(periods=20)

    delta = spy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    spy['RSI'] = 100 - (100 / (1 + rs))

    spy['Volatility'] = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    spy = spy.dropna()
    ten_years_ago = end_date - timedelta(days=365*10)
    spy = spy[spy.index >= ten_years_ago]
    return spy

def calculate_signals(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate signals.

    Args:
        spy (pd.DataFrame): SPY data with indicators.

    Returns:
        pd.DataFrame: SPY data with signals.
    """
    spy['Sig_TrendFilter'] = ((spy['Close'] > spy['MA50']) & (spy['MA200_Slope'] > 0)).astype(int)
    spy['Sig_GoldenCross'] = (spy['MA50'] > spy['MA200']).astype(int)
    spy['Sig_Momentum'] = (spy['Close'] > spy['MA20']).astype(int)
    spy['Sig_RSI_OK'] = (spy['RSI'] < 70).astype(int)
    spy['Sig_LowVol'] = (spy['Volatility'] < 0.25).astype(int)
    return spy

def calculate_positions(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate positions.

    Args:
        spy (pd.DataFrame): SPY data with signals.

    Returns:
        pd.DataFrame: SPY data with positions.
    """
    spy['Position_A'] = spy['Sig_TrendFilter']
    spy['Position_B'] = spy['Sig_TrendFilter'] * 1.0
    spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Sig_GoldenCross'] == 1), 'Position_B'] = 1.5
    spy['Signal_Count'] = (spy['Sig_TrendFilter'] + spy['Sig_GoldenCross'] + 
                           spy['Sig_Momentum'] + spy['Sig_RSI_OK'] + spy['Sig_LowVol'])
    spy['Position_C'] = 0.0
    spy.loc[spy['Sig_TrendFilter'] == 1, 'Position_C'] = 1.0
    spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Signal_Count'] >= 4), 'Position_C'] = 1.5
    spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Signal_Count'] == 5), 'Position_C'] = 2.0
    spy['Position_D'] = 0.0
    spy.loc[spy['Signal_Count'] >= 3, 'Position_D'] = 1.0
    spy.loc[spy['Signal_Count'] >= 4, 'Position_D'] = 1.5
    spy.loc[spy['Signal_Count'] == 5, 'Position_D'] = 2.0
    spy['Position_E'] = spy['Sig_TrendFilter'] * 1.0
    spy.loc[(spy['Sig_TrendFilter'] == 1) & (spy['Sig_GoldenCross'] == 1) & 
            (spy['Sig_Momentum'] == 1), 'Position_E'] = 2.0
    return spy

def calculate_returns(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns.

    Args:
        spy (pd.DataFrame): SPY data with positions.

    Returns:
        pd.DataFrame: SPY data with returns.
    """
    spy['Daily_Return'] = spy['Close'].pct_change()
    spy['Return_BH'] = spy['Daily_Return']
    spy['Return_A'] = spy['Position_A'].shift(1) * spy['Daily_Return']
    spy['Return_B'] = spy['Position_B'].shift(1) * spy['Daily_Return']
    spy['Return_C'] = spy['Position_C'].shift(1) * spy['Daily_Return']
    spy['Return_D'] = spy['Position_D'].shift(1) * spy['Daily_Return']
    spy['Return_E'] = spy['Position_E'].shift(1) * spy['Daily_Return']
    return spy

def calculate_cumulative_returns(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative returns.

    Args:
        spy (pd.DataFrame): SPY data with returns.

    Returns:
        pd.DataFrame: SPY data with cumulative returns.
    """
    for col in ['BH', 'A', 'B', 'C', 'D', 'E']:
        spy[f'Cum_{col}'] = (1 + spy[f'Return_{col}']).cumprod()
    return spy

def calculate_metrics(returns: pd.Series, positions: pd.Series = None) -> dict:
    """
    Calculate metrics.

    Args:
        returns (pd.Series): Returns.
        positions (pd.Series, optional): Positions. Defaults to None.

    Returns:
        dict: Metrics.
    """
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

def main():
    global end_date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10 + 250)
    spy = fetch_spy_data(start_date, end_date)
    spy = calculate_indicators(spy)
    spy = calculate_signals(spy)
    spy = calculate_positions(spy)
    spy = calculate_returns(spy)
    spy = calculate_cumulative_returns(spy)

    metrics = {
        'Buy & Hold': calculate_metrics(spy['Return_BH']),
        'A: Trend Filter': calculate_metrics(spy['Return_A'], spy['Position_A']),
        'B: TF + GC Lev': calculate_metrics(spy['Return_B'], spy['Position_B']),
        'C: Signal Stack': calculate_metrics(spy['Return_C'], spy['Position_C']),
        'D: Conservative': calculate_metrics(spy['Return_D'], spy['Position_D']),
        'E: Aggressive': calculate_metrics(spy['Return_E'], spy['Position_E']),
    }

    print("=" * 95)
    print("COMBINED STRATEGY BACKTEST: SPY with Dynamic Leverage")
    print("=" * 95)
    print()
    print("STRATEGY DEFINITIONS:")
    print("  A: Trend Filter      — 1x when price > 50 DMA AND 200 DMA slope positive")
    print("  B: TF + GC Leverage  — 1x Trend Filter, 1.5x when Golden Cross also active")
    print("  C: Signal Stack      — 1x base, 1.5x with 4 signals, 2x with all 5 signals")
    print("  D: Conservative      — Requires