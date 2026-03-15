"""Tests for app.llm — mock-mode helpers (no API key required)."""
from __future__ import annotations

import pytest

import app.llm as llm_module
from app.llm import (
    has_api_key,
    is_mock_mode,
    route_audience,
    generate_content,
    evaluate_content,
)


# ── Mode detection ────────────────────────────────────────────────────────────

def test_is_mock_mode_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert is_mock_mode() is True
    assert has_api_key() is False


def test_not_mock_mode_when_api_key_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
    assert has_api_key() is True
    assert is_mock_mode() is False


# ── _mock_route (via route_audience in mock mode) ─────────────────────────────

def test_route_audience_mock_commercial(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = route_audience("router prompt", "market sales commercial pricing strategy")
    assert result["audience"] == "commercial"
    assert result["confidence"] > 0


def test_route_audience_mock_medical_affairs(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = route_audience("router prompt", "clinical patient safety evidence trials")
    assert result["audience"] == "medical_affairs"


def test_route_audience_mock_rnd(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = route_audience("router prompt", "experiment assay protocol method laboratory")
    assert result["audience"] == "r_and_d"


def test_route_audience_mock_cross_functional_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = route_audience("router prompt", "the quick brown fox jumps over")
    assert result["audience"] == "cross_functional"
    assert "candidates" in result


def test_route_audience_mock_returns_required_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = route_audience("prompt", "some text")
    for key in ("audience", "confidence", "reasons", "candidates"):
        assert key in result


# ── _mock_generate (via generate_content in mock mode) ───────────────────────

REQUIRED = ["Executive Summary", "Market Opportunity", "Value Proposition"]

def test_generate_content_mock_returns_required_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_content(
        system_prompt="system",
        generation_prompt="generate",
        document_text="This is a document about sales strategy.",
        required_sections=REQUIRED,
        max_words=200,
    )
    for key in ("one_line_summary", "tags", "key_clues", "decision_bullets", "mind_map"):
        assert key in result


def test_generate_content_mock_bullets_include_sections(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_content(
        system_prompt="sys",
        generation_prompt="gen",
        document_text="Document content here.",
        required_sections=REQUIRED,
        max_words=150,
    )
    bullets_text = " ".join(result["decision_bullets"])
    for section in REQUIRED:
        assert section in bullets_text


def test_generate_content_mock_max_5_bullets(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    many_sections = [f"Section {i}" for i in range(10)]
    result = generate_content(
        system_prompt="sys",
        generation_prompt="gen",
        document_text="Text",
        required_sections=many_sections,
        max_words=300,
    )
    assert len(result["decision_bullets"]) <= 5


def test_generate_content_mock_mind_map_is_string(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_content("s", "g", "doc", REQUIRED, 100)
    assert isinstance(result["mind_map"], str)
    assert "mindmap" in result["mind_map"]


# ── _mock_evaluate (via evaluate_content in mock mode) ───────────────────────

def test_evaluate_content_mock_passes_valid_input(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = evaluate_content(
        evaluator_prompt="evaluate",
        one_line_summary="Drug reduces fibrosis by 45%.",
        decision_bullets=[
            "Executive Summary: strong results",
            "Market Opportunity: large addressable market",
            "Value Proposition: differentiated product",
        ],
        required_sections=["Executive Summary", "Market Opportunity", "Value Proposition"],
        max_words=500,
    )
    assert result["pass"] is True
    assert result["missing_sections"] == []
    assert result["fail_reasons"] == []


def test_evaluate_content_mock_fails_missing_section(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = evaluate_content(
        evaluator_prompt="evaluate",
        one_line_summary="Short summary.",
        decision_bullets=[
            "Executive Summary: here",
            "Market Opportunity: here",
            "Value Proposition: here",
        ],
        required_sections=["Executive Summary", "Market Opportunity", "Value Proposition", "Risk Factors"],
        max_words=500,
    )
    assert result["pass"] is False
    assert "Risk Factors" in result["missing_sections"]


def test_evaluate_content_mock_fails_too_few_bullets(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = evaluate_content(
        evaluator_prompt="evaluate",
        one_line_summary="Summary.",
        decision_bullets=["Executive Summary: only one bullet"],
        required_sections=["Executive Summary"],
        max_words=500,
    )
    assert result["pass"] is False
    assert any("bullet" in r.lower() for r in result["fail_reasons"])


def test_evaluate_content_mock_fails_too_many_bullets(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    bullets = [f"Section {i}: text" for i in range(6)]
    result = evaluate_content(
        evaluator_prompt="evaluate",
        one_line_summary="Summary.",
        decision_bullets=bullets,
        required_sections=[],
        max_words=500,
    )
    assert result["pass"] is False


def test_evaluate_content_mock_returns_fix_instructions_on_failure(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = evaluate_content(
        evaluator_prompt="evaluate",
        one_line_summary="ok",
        decision_bullets=["only one"],
        required_sections=[],
        max_words=500,
    )
    assert result["pass"] is False
    assert isinstance(result["fix_instructions"], list)
    assert len(result["fix_instructions"]) > 0


def test_evaluate_content_mock_returns_empty_fix_on_pass(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = evaluate_content(
        evaluator_prompt="evaluate",
        one_line_summary="ok",
        decision_bullets=["S1: a", "S2: b", "S3: c"],
        required_sections=["S1", "S2", "S3"],
        max_words=500,
    )
    assert result["pass"] is True
    assert result["fix_instructions"] == []
