"""Tests for app.tag_intel — tagging, canonicalisation, domain inference, Jaccard."""
from __future__ import annotations

import json

import pytest
from sqlmodel import Session

from app.models import DocumentTagSummary, TagAlias
from app.tag_intel import (
    canonicalize_tags,
    compute_bridge_tags,
    compute_cooccurrence_pairs,
    compute_jaccard,
    count_domains,
    count_tags,
    infer_domain,
    load_alias_map,
    normalize_tag,
    parse_summary_tags,
    persist_tag_summary,
)


# ── normalize_tag ─────────────────────────────────────────────────────────────

def test_normalize_tag_lowercases():
    assert normalize_tag("MASH") == "mash"


def test_normalize_tag_strips_whitespace():
    assert normalize_tag("  fibrosis  ") == "fibrosis"


def test_normalize_tag_collapses_internal_whitespace():
    assert normalize_tag("clinical  trial") == "clinical trial"


def test_normalize_tag_strips_trailing_punctuation():
    assert normalize_tag("safety,") == "safety"
    assert normalize_tag("efficacy.") == "efficacy"


def test_normalize_tag_normalizes_smart_apostrophe():
    assert normalize_tag("patient\u2019s") == "patient's"


def test_normalize_tag_empty_string():
    assert normalize_tag("") == ""


def test_normalize_tag_only_whitespace():
    assert normalize_tag("   ") == ""


# ── canonicalize_tags ─────────────────────────────────────────────────────────

def test_canonicalize_tags_applies_alias():
    alias_map = {"rct": "clinical trial"}
    result = canonicalize_tags(["RCT"], alias_map)
    assert result == ["clinical trial"]


def test_canonicalize_tags_deduplicates():
    alias_map = {"rct": "clinical trial", "randomized trial": "clinical trial"}
    result = canonicalize_tags(["RCT", "randomized trial"], alias_map)
    assert result.count("clinical trial") == 1


def test_canonicalize_tags_passthrough_unknown():
    alias_map = {}
    result = canonicalize_tags(["novel biomarker"], alias_map)
    assert result == ["novel biomarker"]


def test_canonicalize_tags_skips_empty():
    alias_map = {}
    result = canonicalize_tags(["", "  ", "mash"], alias_map)
    assert result == ["mash"]


def test_canonicalize_tags_preserves_order():
    alias_map = {}
    tags = ["fibrosis", "mash", "sirna"]
    result = canonicalize_tags(tags, alias_map)
    assert result == tags


# ── compute_jaccard ───────────────────────────────────────────────────────────

def test_jaccard_identical_sets():
    assert compute_jaccard(["a", "b"], ["a", "b"]) == pytest.approx(1.0)


def test_jaccard_disjoint_sets():
    assert compute_jaccard(["a", "b"], ["c", "d"]) == pytest.approx(0.0)


def test_jaccard_partial_overlap():
    # intersection=1, union=3 → 1/3
    score = compute_jaccard(["a", "b"], ["b", "c"])
    assert score == pytest.approx(1 / 3)


def test_jaccard_both_empty():
    assert compute_jaccard([], []) == pytest.approx(0.0)


def test_jaccard_one_empty():
    assert compute_jaccard(["a"], []) == pytest.approx(0.0)


# ── infer_domain ──────────────────────────────────────────────────────────────

def test_infer_domain_clinical_tags():
    tags = ["clinical trial", "fibrosis", "endpoint", "safety"]
    domain = infer_domain(tags)
    assert domain == "Clinical & Medical Strategy"


def test_infer_domain_rnd_tags():
    tags = ["sirna", "gene silencing", "pharmacology"]
    domain = infer_domain(tags)
    assert domain == "Translational Science & Drug R&D"


def test_infer_domain_regulatory_tags():
    tags = ["fda approval", "launch"]
    domain = infer_domain(tags)
    assert domain == "Regulatory, Launch & Market Strategy"


def test_infer_domain_corporate_tags():
    tags = ["nasdaq", "equity awards"]
    domain = infer_domain(tags)
    assert domain == "Corporate & Investor Updates"


def test_infer_domain_empty_tags():
    assert infer_domain([]) == "General / Other"


def test_infer_domain_unknown_tags():
    assert infer_domain(["unrelated topic", "random words"]) == "General / Other"


# ── parse_summary_tags ────────────────────────────────────────────────────────

def _make_summary(tags: list[str]) -> DocumentTagSummary:
    return DocumentTagSummary(
        document_id=1,
        job_id=None,
        domain="Clinical & Medical Strategy",
        canonical_tags_json=json.dumps(tags),
    )


def test_parse_summary_tags_returns_list():
    s = _make_summary(["fibrosis", "mash"])
    assert parse_summary_tags(s) == ["fibrosis", "mash"]


def test_parse_summary_tags_empty_json():
    s = _make_summary([])
    assert parse_summary_tags(s) == []


def test_parse_summary_tags_invalid_json():
    s = DocumentTagSummary(
        document_id=1, job_id=None, domain="x", canonical_tags_json="NOT JSON"
    )
    assert parse_summary_tags(s) == []


def test_parse_summary_tags_normalizes_values():
    s = _make_summary(["FIBROSIS", "  mash  "])
    result = parse_summary_tags(s)
    assert "fibrosis" in result
    assert "mash" in result


# ── count_domains ─────────────────────────────────────────────────────────────

def test_count_domains_counts_correctly():
    summaries = [
        _make_summary(["fibrosis"]),  # domain = Clinical
        _make_summary(["sirna"]),     # same domain placeholder
    ]
    # Override domain directly
    summaries[0].domain = "Clinical & Medical Strategy"
    summaries[1].domain = "Translational Science & Drug R&D"
    result = count_domains(summaries)
    domain_names = [d for d, _ in result]
    assert "Clinical & Medical Strategy" in domain_names
    assert "Translational Science & Drug R&D" in domain_names


def test_count_domains_sorted_by_count_descending():
    summaries = []
    for _ in range(3):
        s = _make_summary([])
        s.domain = "Clinical & Medical Strategy"
        summaries.append(s)
    s2 = _make_summary([])
    s2.domain = "Corporate & Investor Updates"
    summaries.append(s2)

    result = count_domains(summaries)
    assert result[0] == ("Clinical & Medical Strategy", 3)


# ── count_tags ────────────────────────────────────────────────────────────────

def test_count_tags_aggregates_all_summaries():
    s1 = _make_summary(["fibrosis", "mash"])
    s2 = _make_summary(["mash", "sirna"])
    counter = count_tags([s1, s2])
    assert counter["mash"] == 2
    assert counter["fibrosis"] == 1
    assert counter["sirna"] == 1


# ── compute_cooccurrence_pairs ────────────────────────────────────────────────

def test_cooccurrence_pairs_returns_pairs():
    s = _make_summary(["fibrosis", "mash", "clinical trial"])
    pairs = compute_cooccurrence_pairs([s])
    pair_keys = [p for p, _ in pairs]
    assert ("clinical trial", "fibrosis") in pair_keys or ("fibrosis", "mash") in pair_keys


def test_cooccurrence_pairs_single_tag_no_pairs():
    s = _make_summary(["fibrosis"])
    pairs = compute_cooccurrence_pairs([s])
    assert pairs == []


# ── compute_bridge_tags ───────────────────────────────────────────────────────

def test_bridge_tags_detects_cross_domain_tag():
    s1 = _make_summary(["mash", "fibrosis"])
    s1.domain = "Clinical & Medical Strategy"
    s2 = _make_summary(["mash", "sirna"])
    s2.domain = "Translational Science & Drug R&D"
    bridges = compute_bridge_tags([s1, s2])
    bridge_names = [t for t, _ in bridges]
    assert "mash" in bridge_names


def test_bridge_tags_single_domain_no_bridge():
    s1 = _make_summary(["fibrosis"])
    s1.domain = "Clinical & Medical Strategy"
    s2 = _make_summary(["efficacy"])
    s2.domain = "Clinical & Medical Strategy"
    bridges = compute_bridge_tags([s1, s2])
    assert bridges == []


# ── load_alias_map (DB integration) ──────────────────────────────────────────

def test_load_alias_map_includes_builtins(db_session):
    alias_map = load_alias_map(db_session)
    # Built-in alias from BUILTIN_ALIAS_MAP
    assert alias_map.get("rct") == "clinical trial"


def test_load_alias_map_includes_db_aliases(db_session):
    db_session.add(TagAlias(alias="nash", canonical="mash"))
    db_session.commit()
    alias_map = load_alias_map(db_session)
    assert alias_map.get("nash") == "mash"


# ── persist_tag_summary ───────────────────────────────────────────────────────

def _make_doc_and_job(db_session):
    """Helper: create a persisted Document + Job pair and return (doc, job)."""
    from app.models import Document, Job
    doc = Document(content="test", source_type="text")
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    job = Job(document_id=doc.id, selected_audience="auto")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return doc, job


def test_persist_tag_summary_creates_record(db_session):
    doc, job = _make_doc_and_job(db_session)
    result = persist_tag_summary(db_session, doc.id, job.id, ["fibrosis", "mash"])
    db_session.commit()

    assert result["domain"] != ""
    assert "fibrosis" in result["canonical_tags"] or "mash" in result["canonical_tags"]


def test_persist_tag_summary_empty_tags_defaults_to_general(db_session):
    doc, job = _make_doc_and_job(db_session)
    result = persist_tag_summary(db_session, doc.id, job.id, [])
    db_session.commit()
    assert "general" in result["canonical_tags"]


def test_persist_tag_summary_raises_on_none_job_id(db_session):
    doc, _ = _make_doc_and_job(db_session)
    with pytest.raises(ValueError, match="job_id"):
        persist_tag_summary(db_session, doc.id, None, ["fibrosis"])
