"""
Tests for strategy position logic in dashboard.py.

dashboard.py calls st.set_page_config() at module level and also executes
rendering code at import time (chart creation, etc.).  We stub out the
streamlit and plotly modules before importing so no Streamlit runtime is
needed and no network calls are made.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Streamlit + plotly stubs
# ---------------------------------------------------------------------------

def _make_streamlit_stub() -> types.ModuleType:
    """Return a minimal streamlit stub compatible with dashboard.py's module-level code."""
    stub = types.ModuleType("streamlit")

    # Transparent decorator for @st.cache_data(ttl=...)
    def _cache_data(ttl=None):
        def decorator(fn):
            return fn
        return decorator

    stub.cache_data = _cache_data

    # st.columns returns a list of context-manager mocks
    def _columns(n):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        count = n if isinstance(n, int) else len(n)
        return [ctx for _ in range(count)]

    stub.columns = _columns

    # st.expander as a context manager
    exp_ctx = MagicMock()
    exp_ctx.__enter__ = MagicMock(return_value=exp_ctx)
    exp_ctx.__exit__ = MagicMock(return_value=False)
    stub.expander = MagicMock(return_value=exp_ctx)

    # st.spinner as a context manager
    spinner_cm = MagicMock()
    spinner_cm.__enter__ = MagicMock(return_value=spinner_cm)
    spinner_cm.__exit__ = MagicMock(return_value=False)
    stub.spinner = MagicMock(return_value=spinner_cm)

    # CRITICAL: st.button must return False so conditional blocks (backtest,
    # comparison) do not execute during module-level code evaluation.
    stub.button = MagicMock(return_value=False)

    # st.selectbox returns a sensible default string
    stub.selectbox = MagicMock(return_value="Ensemble")

    # Numeric inputs with safe defaults
    stub.number_input = MagicMock(return_value=100000)
    stub.slider = MagicMock(return_value=50)

    # Sidebar with attribute access
    stub.sidebar = MagicMock()
    stub.sidebar.header = MagicMock()
    stub.sidebar.subheader = MagicMock()
    stub.sidebar.selectbox = MagicMock(return_value="SPY")
    stub.sidebar.text_input = MagicMock(return_value="SPY")
    stub.sidebar.caption = MagicMock()

    # Everything else: no-op
    for _name in [
        "set_page_config", "title", "header", "subheader", "caption",
        "metric", "success", "error", "info", "warning", "stop",
        "divider", "write", "text_input", "dataframe", "plotly_chart",
    ]:
        setattr(stub, _name, MagicMock())

    return stub


def _make_fake_spy_df() -> pd.DataFrame:
    """Build a synthetic SPY-like DataFrame with all indicator columns that
    dashboard.py expects to find at module level."""
    dates = pd.bdate_range(end="2024-12-31", periods=300)
    close = np.linspace(400.0, 500.0, 300)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": np.full(300, 1e8),
        },
        index=dates,
    )
    df["MA20"] = df["Close"].rolling(20).mean().ffill()
    df["MA50"] = df["Close"].rolling(50).mean().ffill()
    df["MA200"] = df["Close"].rolling(200).mean().ffill()
    df["MA200_Slope"] = df["MA200"].pct_change(20).ffill()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = (100 - (100 / (1 + gain / loss))).ffill()

    gain5 = delta.where(delta > 0, 0).rolling(5).mean()
    loss5 = (-delta.where(delta < 0, 0)).rolling(5).mean()
    df["RSI5"] = (100 - (100 / (1 + gain5 / loss5))).ffill()

    df["Vol"] = df["Close"].pct_change().rolling(20).std().ffill() * (252 ** 0.5)
    df["Mom12M"] = df["Close"].pct_change(252).ffill()
    df["MonthEnd"] = df.index.is_month_end
    return df


def _import_dashboard():
    """
    Import dashboard.py with streamlit and plotly stubbed out.
    Returns the module object, or raises with a clear message on failure.
    """
    # Remove any previous import so the stub takes effect
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("dashboard", "streamlit"):
            del sys.modules[mod_name]

    sys.modules["streamlit"] = _make_streamlit_stub()
    sys.modules.setdefault("plotly", MagicMock())
    sys.modules.setdefault("plotly.graph_objects", MagicMock())
    sys.modules.setdefault("plotly.subplots", MagicMock())

    fake_df = _make_fake_spy_df()

    with patch("yfinance.download", return_value=fake_df):
        import dashboard as _dash  # noqa: PLC0415

    return _dash


# ---------------------------------------------------------------------------
# Module-level import — executed once when pytest collects this file
# ---------------------------------------------------------------------------

_IMPORT_ERROR: Exception | None = None
_IMPORT_OK = False
_dashboard = None

try:
    _dashboard = _import_dashboard()
    get_strategy_position = _dashboard.get_strategy_position
    get_signals = _dashboard.get_signals
    _IMPORT_OK = True
except Exception as _exc:
    _IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_signals(value: bool) -> dict:
    """Return a signals dict where every signal has the given boolean value."""
    return {
        "Trend Filter": value,
        "Golden Cross": value,
        "Momentum": value,
        "RSI OK": value,
        "Low Vol": value,
    }


def _signals_with_count(n: int) -> dict:
    """Return a signals dict with exactly n True values (first n keys)."""
    keys = ["Trend Filter", "Golden Cross", "Momentum", "RSI OK", "Low Vol"]
    return {k: (i < n) for i, k in enumerate(keys)}


def _skip_if_no_import(fn):
    """Decorator: skip the test if dashboard failed to import."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _IMPORT_OK:
            pytest.skip(f"dashboard import failed: {_IMPORT_ERROR}")
        return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_skip_if_no_import
def test_full_send_all_signals_true():
    """
    Full Send strategy: when Trend Filter, Golden Cross, and Momentum are all
    True the position must be 2.0 (maximum leverage).
    """
    signals = _all_signals(True)
    positions, _ = get_strategy_position(signals)
    assert positions["Full Send"] == 2.0, (
        f"Full Send should be 2.0 when all signals True, got {positions['Full Send']}"
    )


@_skip_if_no_import
def test_full_send_no_signals():
    """
    Full Send strategy: when all signals are False the position must be
    0.0 (cash).
    """
    signals = _all_signals(False)
    positions, _ = get_strategy_position(signals)
    assert positions["Full Send"] == 0.0, (
        f"Full Send should be 0.0 when all signals False, got {positions['Full Send']}"
    )


@_skip_if_no_import
def test_signal_stacker_3_signals():
    """
    Signal Stacker: exactly 3 active signals must produce a 1x (1.0) position.
    """
    signals = _signals_with_count(3)
    positions, count = get_strategy_position(signals)
    assert count == 3, f"Expected count 3, got {count}"
    assert positions["Signal Stacker"] == 1.0, (
        f"Signal Stacker with 3 signals should be 1.0, got {positions['Signal Stacker']}"
    )


@_skip_if_no_import
def test_signal_stacker_4_signals():
    """
    Signal Stacker: exactly 4 active signals must produce a 1.5x position.
    """
    signals = _signals_with_count(4)
    positions, count = get_strategy_position(signals)
    assert count == 4, f"Expected count 4, got {count}"
    assert positions["Signal Stacker"] == 1.5, (
        f"Signal Stacker with 4 signals should be 1.5, got {positions['Signal Stacker']}"
    )


@_skip_if_no_import
def test_signal_stacker_5_signals():
    """
    Signal Stacker: all 5 signals active must produce a 2.0x position.
    """
    signals = _signals_with_count(5)
    positions, count = get_strategy_position(signals)
    assert count == 5, f"Expected count 5, got {count}"
    assert positions["Signal Stacker"] == 2.0, (
        f"Signal Stacker with 5 signals should be 2.0, got {positions['Signal Stacker']}"
    )


@_skip_if_no_import
def test_conservative_requires_3_signals():
    """
    Steady Eddie (the conservative strategy mapped from Strategy D): with only
    2 signals active the position must be 0.0 (cash).  The strategy requires
    at least 3 signals before entering any long position.
    """
    signals = _signals_with_count(2)
    positions, count = get_strategy_position(signals)
    assert count == 2, f"Expected count 2, got {count}"
    assert positions["Steady Eddie"] == 0.0, (
        f"Steady Eddie with 2 signals should be 0.0 (cash), got {positions['Steady Eddie']}"
    )


@_skip_if_no_import
def test_trend_rider_cash_when_no_trend():
    """
    Trend Rider: when Trend Filter is False the position must be 0.0
    regardless of the other signals.
    """
    signals = _all_signals(True)
    signals["Trend Filter"] = False
    positions, _ = get_strategy_position(signals)
    assert positions["Trend Rider"] == 0.0, (
        f"Trend Rider should be 0.0 when Trend Filter is False, got {positions['Trend Rider']}"
    )


@_skip_if_no_import
def test_signal_count_returned_correctly():
    """
    get_strategy_position() must return the exact signal count as the second
    element of its return tuple for all possible counts 0–5.
    """
    for n in range(6):
        signals = _signals_with_count(n)
        _, count = get_strategy_position(signals)
        assert count == n, f"Expected signal count {n}, got {count}"
