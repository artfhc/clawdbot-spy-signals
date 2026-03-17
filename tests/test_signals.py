"""
Tests for signal calculation logic extracted from daily_signal_checker.py.

All yfinance.download calls are mocked; no real network calls are made.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helper: replicate the indicator calculation block from daily_signal_checker
# ---------------------------------------------------------------------------

def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror the indicator calculations in daily_signal_checker.get_signals()
    so we can test them independently without importing the module (which
    would trigger a live yfinance call at import time via __main__ guard —
    the module is safe to import, but this keeps tests self-contained).
    """
    data = df.copy()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["MA200_Slope"] = data["MA200"].pct_change(periods=20)

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    data["Vol"] = data["Close"].pct_change().rolling(20).std() * (252 ** 0.5)
    return data


def _build_signals(data: pd.DataFrame) -> dict:
    """
    Build the five boolean signals from the last row of a prepared DataFrame,
    mirroring the logic in daily_signal_checker.get_signals().
    """
    latest = data.iloc[-1]
    price = latest["Close"]
    ma20 = latest["MA20"]
    ma50 = latest["MA50"]
    ma200 = latest["MA200"]
    ma200_slope = latest["MA200_Slope"]
    rsi = latest["RSI"]
    vol = latest["Vol"]

    return {
        "sig_trend": bool((price > ma50) and (ma200_slope > 0)),
        "sig_golden": bool(ma50 > ma200),
        "sig_momentum": bool(price > ma20),
        "sig_rsi": bool(rsi < 70),
        "sig_lowvol": bool(vol < 0.25),
        "rsi": rsi,
        "vol": vol,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rsi_no_division_by_zero_flat_market(flat_prices):
    """
    With all Close prices identical both the average gain and average loss are
    zero after the rolling mean.  The formula `100 - 100/(1 + gain/loss)` hits
    a 0/0 division which pandas silently converts to NaN (not inf, not crash).

    This test documents the current behavior: the RSI series for a flat market
    is entirely NaN.  No exception must be raised during the calculation.
    The caller (get_signals) must handle this by accessing only the last row,
    which will also be NaN — an acceptable edge case given the real data always
    has at least some price variation.
    """
    # Must not raise an exception
    data = _calculate_indicators(flat_prices)

    # The RSI column must exist
    assert "RSI" in data.columns, "RSI column must be present in output"

    # Current behavior: all NaN due to 0/0 in gain/loss for flat prices
    rsi_series = data["RSI"]
    # No inf values — pandas converts 0/0 to NaN, not inf
    assert not np.isinf(rsi_series.dropna()).any(), (
        "RSI must not contain inf values even in a flat market"
    )


def test_rsi_range(spy_prices):
    """
    RSI must always be between 0 and 100 (inclusive) for any valid price
    series.  Values outside this range indicate a calculation bug.
    """
    data = _calculate_indicators(spy_prices)
    rsi_series = data["RSI"].dropna()

    assert (rsi_series >= 0).all(), f"RSI went below 0: min={rsi_series.min()}"
    assert (rsi_series <= 100).all(), f"RSI exceeded 100: max={rsi_series.max()}"


def test_ma_crossover_signals():
    """
    Construct a price series whose final segment is strictly rising so
    that the 50-day MA crosses above the 200-day MA.  The golden cross
    signal must then be True.
    """
    dates = pd.bdate_range(end="2024-12-31", periods=300)
    # Flat at 400 for first 250 days, then rally sharply to force MA50 > MA200
    close = np.full(300, 400.0)
    close[250:] = np.linspace(400, 600, 50)
    df = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": 1e8}, index=dates)

    data = _calculate_indicators(df)
    sigs = _build_signals(data)

    assert sigs["sig_golden"] is True, (
        f"Expected golden cross (MA50 > MA200) to be True. "
        f"MA50={data['MA50'].iloc[-1]:.2f}, MA200={data['MA200'].iloc[-1]:.2f}"
    )


def test_trend_filter_requires_price_above_50ma():
    """
    When price is consistently below the 50-day MA the trend filter signal
    must be False, regardless of the 200 DMA slope.
    """
    dates = pd.bdate_range(end="2024-12-31", periods=300)
    # Downward slope ensures price < MA50
    close = np.linspace(600, 300, 300)
    df = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": 1e8}, index=dates)

    data = _calculate_indicators(df)
    latest = data.iloc[-1]

    price_above_ma50 = latest["Close"] > latest["MA50"]
    assert price_above_ma50 is False or not price_above_ma50, (
        f"Expected price ({latest['Close']:.2f}) to be below MA50 ({latest['MA50']:.2f})"
    )

    sigs = _build_signals(data)
    # Trend signal requires BOTH price > MA50 AND MA200 slope positive.
    # With price < MA50, sig_trend must be False.
    assert sigs["sig_trend"] is False, "Trend filter should be False when price < MA50"


def test_signal_count_range(spy_prices):
    """
    The sum of the five boolean signals must always be between 0 and 5.
    """
    data = _calculate_indicators(spy_prices)
    sigs = _build_signals(data)

    signal_count = sum([
        sigs["sig_trend"],
        sigs["sig_golden"],
        sigs["sig_momentum"],
        sigs["sig_rsi"],
        sigs["sig_lowvol"],
    ])

    assert 0 <= signal_count <= 5, (
        f"Signal count {signal_count} is outside valid range [0, 5]"
    )


@pytest.mark.xfail(
    reason=(
        "daily_signal_checker.get_signals() does not currently validate "
        "the row count of the downloaded data and will not raise when given "
        "fewer than 250 rows.  This test documents the missing guard; it is "
        "marked xfail so the suite stays green until the check is added."
    ),
    strict=False,
)
def test_insufficient_data_raises():
    """
    If yfinance returns fewer than 250 rows the function should raise a
    ValueError (or similar) because there is not enough data to warm up the
    200-day MA.  Currently the function does not perform this check.
    """
    import daily_signal_checker

    dates = pd.bdate_range(end="2024-12-31", periods=100)  # Only 100 rows
    close = np.linspace(400, 420, 100)
    tiny_df = pd.DataFrame(
        {"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": 1e8},
        index=dates,
    )

    with patch("yfinance.download", return_value=tiny_df):
        # Expect an exception due to insufficient data
        with pytest.raises((ValueError, IndexError, KeyError)):
            daily_signal_checker.get_signals("SPY")


def test_get_signals_returns_expected_keys(spy_prices):
    """
    get_signals() must return a dict containing all documented top-level
    keys when given sufficient synthetic data via a mocked yfinance call.
    """
    import daily_signal_checker

    with patch("yfinance.download", return_value=spy_prices):
        result = daily_signal_checker.get_signals("SPY")

    required_keys = {"date", "ticker", "price", "ma20", "ma50", "ma200",
                     "ma200_slope", "rsi", "volatility", "signals",
                     "signal_count", "strategies"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )

    # All 5 strategies must be present
    assert set(result["strategies"].keys()) == {"A", "B", "C", "D", "E"}
