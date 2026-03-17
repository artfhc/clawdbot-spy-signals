"""
Shared pytest fixtures for the SPY trading signal test suite.

Provides synthetic price DataFrames that mimic yfinance output without
making any real network calls.
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(close_prices: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a Close price array."""
    n = len(close_prices)
    # Construct Open/High/Low from Close with small offsets
    open_prices = close_prices * (1 + np.random.uniform(-0.005, 0.005, n))
    high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.uniform(0, 0.005, n)))
    low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.uniform(0, 0.005, n)))
    volume = np.random.randint(50_000_000, 150_000_000, n).astype(float)

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture(scope="session")
def business_dates() -> pd.DatetimeIndex:
    """300 business days ending at a fixed date for reproducibility."""
    return pd.bdate_range(end="2024-12-31", periods=300)


@pytest.fixture
def spy_prices(business_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    300-row synthetic DataFrame with an upward-trending Close price series.

    Start price ~400, gentle random walk with a small positive drift so
    that MAs will trend upward and golden-cross conditions can be established
    with enough data.
    """
    rng = np.random.default_rng(42)
    n = len(business_dates)
    daily_returns = rng.normal(loc=0.0004, scale=0.010, size=n)
    close_prices = 400.0 * np.cumprod(1 + daily_returns)
    return _make_ohlcv(close_prices, business_dates)


@pytest.fixture
def flat_prices(business_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    300-row DataFrame where every Close price is identical (500.0).

    Used to test RSI edge cases where the denominator (average loss) is zero.
    """
    close_prices = np.full(len(business_dates), 500.0)
    return _make_ohlcv(close_prices, business_dates)


@pytest.fixture
def bear_prices(business_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    300-row DataFrame with a downward-trending Close price series.

    Start price ~500 with a negative drift so that trend signals should
    be False and strategies should go to cash.
    """
    rng = np.random.default_rng(7)
    n = len(business_dates)
    daily_returns = rng.normal(loc=-0.0006, scale=0.010, size=n)
    close_prices = 500.0 * np.cumprod(1 + daily_returns)
    return _make_ohlcv(close_prices, business_dates)
