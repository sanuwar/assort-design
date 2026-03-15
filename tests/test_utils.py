"""Tests for app.utils — env_int and env_float helpers."""
from __future__ import annotations

import pytest

from app.utils import env_int, env_float


# ── env_int ──────────────────────────────────────────────────────────────────

def test_env_int_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_INT_VAR", raising=False)
    assert env_int("TEST_INT_VAR", 42) == 42


def test_env_int_parses_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "100")
    assert env_int("TEST_INT_VAR", 0) == 100


def test_env_int_parses_negative(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "-7")
    assert env_int("TEST_INT_VAR", 0) == -7


def test_env_int_returns_default_on_invalid(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "not_a_number")
    assert env_int("TEST_INT_VAR", 5) == 5


def test_env_int_returns_default_on_empty_string(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "   ")
    assert env_int("TEST_INT_VAR", 5) == 5


def test_env_int_strips_whitespace(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "  20  ")
    assert env_int("TEST_INT_VAR", 0) == 20


def test_env_int_logs_warning_on_invalid(monkeypatch, caplog):
    monkeypatch.setenv("TEST_INT_VAR", "oops")
    import logging
    with caplog.at_level(logging.WARNING, logger="app.utils"):
        result = env_int("TEST_INT_VAR", 99)
    assert result == 99
    assert "TEST_INT_VAR" in caplog.text


# ── env_float ────────────────────────────────────────────────────────────────

def test_env_float_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_FLOAT_VAR", raising=False)
    assert env_float("TEST_FLOAT_VAR", 3.14) == pytest.approx(3.14)


def test_env_float_parses_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_VAR", "2.5")
    assert env_float("TEST_FLOAT_VAR", 0.0) == pytest.approx(2.5)


def test_env_float_parses_integer_string(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_VAR", "10")
    assert env_float("TEST_FLOAT_VAR", 0.0) == pytest.approx(10.0)


def test_env_float_parses_negative(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_VAR", "-0.5")
    assert env_float("TEST_FLOAT_VAR", 0.0) == pytest.approx(-0.5)


def test_env_float_returns_default_on_invalid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_VAR", "abc")
    assert env_float("TEST_FLOAT_VAR", 1.0) == pytest.approx(1.0)


def test_env_float_returns_default_on_empty_string(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_VAR", "")
    assert env_float("TEST_FLOAT_VAR", 7.7) == pytest.approx(7.7)


def test_env_float_logs_warning_on_invalid(monkeypatch, caplog):
    monkeypatch.setenv("TEST_FLOAT_VAR", "bad")
    import logging
    with caplog.at_level(logging.WARNING, logger="app.utils"):
        result = env_float("TEST_FLOAT_VAR", 0.0)
    assert result == pytest.approx(0.0)
    assert "TEST_FLOAT_VAR" in caplog.text
