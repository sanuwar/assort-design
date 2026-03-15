"""Tests for app.tools — citation_finder and risk_checker."""
from __future__ import annotations

import pytest

from app.tools import (
    CitationResult,
    citation_finder,
    count_supported_citations,
    risk_checker,
)


SOURCE = (
    "The drug demonstrated a 45% reduction in fibrosis scores in Phase 3 trials. "
    "Safety data showed patients experienced mild adverse events. "
    "The study enrolled 966 patients over 52 weeks."
)


# ── citation_finder ───────────────────────────────────────────────────────────

def test_citation_exact_match():
    results = citation_finder(SOURCE, ["45% reduction in fibrosis scores"])
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.9)
    assert "45%" in results[0].quote_text


def test_citation_token_overlap_match():
    results = citation_finder(SOURCE, ["fibrosis Phase 3"])
    assert len(results) == 1
    assert results[0].confidence >= 0.4
    assert results[0].quote_text != ""


def test_citation_no_match_returns_low_confidence():
    results = citation_finder(SOURCE, ["quantum computing breakthrough"])
    assert len(results) == 1
    assert results[0].confidence < 0.4
    assert results[0].quote_text == ""
    assert results[0].source_start is None


def test_citation_empty_claims_returns_empty():
    results = citation_finder(SOURCE, [])
    assert results == []


def test_citation_empty_source_returns_low_confidence():
    results = citation_finder("", ["some claim"])
    assert len(results) == 1
    assert results[0].confidence < 0.4


def test_citation_none_source_handled():
    results = citation_finder(None, ["some claim"])
    assert len(results) == 1


def test_citation_caps_at_8_results():
    claims = [f"claim {i}" for i in range(20)]
    results = citation_finder(SOURCE, claims)
    assert len(results) <= 8


def test_citation_result_fields_present():
    results = citation_finder(SOURCE, ["966 patients"])
    assert len(results) == 1
    r = results[0]
    assert hasattr(r, "claim_text")
    assert hasattr(r, "quote_text")
    assert hasattr(r, "source_start")
    assert hasattr(r, "source_end")
    assert hasattr(r, "confidence")


# ── count_supported_citations ────────────────────────────────────────────────

def test_count_supported_citations_counts_high_confidence():
    citations = [
        CitationResult("c1", "some quote", 0, 10, 0.9),
        CitationResult("c2", "another quote", 10, 20, 0.5),
        CitationResult("c3", "", None, None, 0.2),   # no quote → not counted
        CitationResult("c4", "low conf", 20, 30, 0.3),  # below threshold
    ]
    assert count_supported_citations(citations) == 2


def test_count_supported_citations_empty():
    assert count_supported_citations([]) == 0


# ── risk_checker ─────────────────────────────────────────────────────────────

def test_risk_checker_empty_text():
    flags = risk_checker("")
    # Empty text → no efficacy found, no safety found; possibly a low flag
    assert isinstance(flags, list)


def test_risk_checker_absolute_efficacy_high():
    flags = risk_checker("This drug will cure all patients.")
    categories = [f.category for f in flags]
    assert "absolute efficacy claim" in categories
    high_flags = [f for f in flags if f.severity == "high"]
    assert len(high_flags) >= 1


def test_risk_checker_100_percent_high():
    flags = risk_checker("We saw 100% response rate in this cohort.")
    severities = {f.severity for f in flags}
    assert "high" in severities or "medium" in severities


def test_risk_checker_off_label_high():
    flags = risk_checker("The compound is used off-label for pediatric populations.")
    categories = [f.category for f in flags]
    assert "off-label promotion" in categories


def test_risk_checker_safety_minimisation_high():
    flags = risk_checker("Patients experienced only minor side effects.")
    categories = [f.category for f in flags]
    assert "safety minimisation" in categories


def test_risk_checker_over_certain_language_high():
    flags = risk_checker("This therapy always works for NASH patients.")
    categories = [f.category for f in flags]
    assert "over-certain language" in categories


def test_risk_checker_comparative_medium():
    flags = risk_checker("Our drug outperforms the current standard of care.")
    categories = [f.category for f in flags]
    assert "unsupported comparative claim" in categories
    medium_flags = [f for f in flags if f.severity == "medium"]
    assert len(medium_flags) >= 1


def test_risk_checker_missing_limitation_low():
    # Text with no limitation language (no may/might/could/etc.) and no safety language
    text = "The drug is effective and produces a positive outcome."
    flags = risk_checker(text)
    categories = [f.category for f in flags]
    assert "missing limitation language" in categories


def test_risk_checker_missing_safety_low():
    text = "The drug shows excellent efficacy and strong response rates."
    flags = risk_checker(text)
    categories = [f.category for f in flags]
    assert "missing safety language" in categories


def test_risk_checker_deduplicates_same_span():
    # The same phrase should not produce two identical flags
    text = "always always always"
    flags = risk_checker(text)
    keys = [f"{f.category}:{f.text_span[:50].lower()}" for f in flags]
    assert len(keys) == len(set(keys))


def test_risk_checker_caps_at_12():
    # Construct text with many trigger phrases
    text = (
        "always never guarantees best-in-class superior outperforms "
        "off-label 100% response patients reported will cure "
        "minor side effects proven that"
    )
    flags = risk_checker(text)
    assert len(flags) <= 12


def test_risk_checker_safe_text_no_high_flags():
    text = (
        "Data suggest the drug may reduce fibrosis scores. "
        "Adverse events were observed in 12% of patients. "
        "Further studies are needed to confirm these preliminary findings."
    )
    flags = risk_checker(text)
    high_flags = [f for f in flags if f.severity == "high"]
    assert len(high_flags) == 0


def test_risk_checker_conflict_of_interest_low():
    text = "The study was supported by a key opinion leader on the advisory board."
    flags = risk_checker(text)
    categories = [f.category for f in flags]
    assert "conflict of interest / disclosure" in categories
