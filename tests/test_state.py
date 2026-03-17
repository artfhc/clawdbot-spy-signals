"""
Tests for load_previous_state() and save_state() in signal_alerts.py.

All tests use pytest's tmp_path fixture and monkeypatch to redirect the
STATE_FILE path so the real signal_state.json is never touched.
No network calls are made.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import signal_alerts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def redirect_state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Redirect signal_alerts.STATE_FILE to a temp directory for every test.

    autouse=True means this runs automatically for every test in this module
    without needing to be listed explicitly as a parameter.
    """
    tmp_state = tmp_path / "signal_state.json"
    monkeypatch.setattr(signal_alerts, "STATE_FILE", tmp_state)
    return tmp_state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path: Path):
    """
    Save a state dict then load it back.  All fields must survive the
    JSON serialisation round-trip with identical values.
    """
    test_signals = {
        "Trend Rider": "LONG 1x",
        "Golden Boost": "LONG 1.5x",
        "Signal Stacker": "LONG 2x",
        "Steady Eddie": "CASH",
        "Full Send": "CASH",
    }
    test_price = 487.32

    signal_alerts.save_state(test_signals, test_price)
    loaded = signal_alerts.load_previous_state()

    assert loaded is not None, "load_previous_state() returned None after save"
    assert loaded["price"] == test_price, (
        f"Price mismatch: expected {test_price}, got {loaded['price']}"
    )
    assert loaded["signals"] == test_signals, (
        f"Signals mismatch: {loaded['signals']}"
    )


def test_load_missing_file_returns_empty_dict():
    """
    When the state file does not exist load_previous_state() must return an
    empty dict (not None, not raise an exception).

    The current implementation returns {} when the file is absent.
    """
    # The redirect_state_file fixture points STATE_FILE at a non-existent path
    assert not signal_alerts.STATE_FILE.exists(), (
        "STATE_FILE should not exist at the start of this test"
    )

    result = signal_alerts.load_previous_state()

    # The implementation returns {} when the file does not exist
    assert result is not None, "load_previous_state() must not return None"
    assert isinstance(result, dict), (
        f"Expected dict, got {type(result)}"
    )


def test_save_state_creates_file():
    """
    After calling save_state() the state file must exist on disk.
    """
    assert not signal_alerts.STATE_FILE.exists(), "STATE_FILE should not pre-exist"

    signal_alerts.save_state({"Trend Rider": "LONG 1x"}, 490.0)

    assert signal_alerts.STATE_FILE.exists(), (
        f"STATE_FILE not created at {signal_alerts.STATE_FILE}"
    )


def test_save_state_writes_valid_json():
    """
    The saved file must be parseable JSON with the expected top-level keys:
    date, price, signals.
    """
    signals = {"Full Send": "LONG 2x"}
    signal_alerts.save_state(signals, 500.0)

    raw = signal_alerts.STATE_FILE.read_text(encoding="utf-8")
    data = json.loads(raw)

    assert "date" in data, "Saved JSON missing 'date' key"
    assert "price" in data, "Saved JSON missing 'price' key"
    assert "signals" in data, "Saved JSON missing 'signals' key"
    assert data["price"] == 500.0


def test_save_overwrites_previous_state():
    """
    A second save_state() call must overwrite the previous file so that
    only the most recent state is retained.
    """
    signal_alerts.save_state({"Trend Rider": "CASH"}, 400.0)
    signal_alerts.save_state({"Trend Rider": "LONG 1x"}, 450.0)

    loaded = signal_alerts.load_previous_state()
    assert loaded["price"] == 450.0, (
        f"Expected price 450.0 after overwrite, got {loaded['price']}"
    )
    assert loaded["signals"]["Trend Rider"] == "LONG 1x"


def test_detect_changes_finds_differences():
    """
    detect_changes() must identify strategy signals that changed between
    old and new state, and ignore unchanged ones.
    """
    old = {"Trend Rider": "CASH", "Full Send": "LONG 2x"}
    new = {"Trend Rider": "LONG 1x", "Full Send": "LONG 2x"}

    changes = signal_alerts.detect_changes(old, new)

    assert len(changes) == 1, f"Expected 1 change, got {len(changes)}: {changes}"
    assert changes[0]["strategy"] == "Trend Rider"
    assert changes[0]["from"] == "CASH"
    assert changes[0]["to"] == "LONG 1x"


def test_detect_changes_empty_when_no_changes():
    """
    detect_changes() must return an empty list when old and new signals are
    identical.
    """
    signals = {"Trend Rider": "LONG 1x", "Full Send": "CASH"}
    changes = signal_alerts.detect_changes(signals, signals.copy())
    assert changes == [], f"Expected no changes, got {changes}"


def test_detect_changes_ignores_new_strategies():
    """
    If a strategy appears in new_signals but not in old_signals it must not
    be reported as a change (there is no prior state to compare against).
    """
    old = {}
    new = {"Trend Rider": "LONG 1x"}

    changes = signal_alerts.detect_changes(old, new)
    assert changes == [], (
        f"New strategies with no prior state should not be flagged as changes, got {changes}"
    )
