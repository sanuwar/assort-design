"""Extended HTTP route tests — broader coverage across web endpoints."""
from __future__ import annotations

from typing import Any, Dict

import app.graph as app_graph
from app.models import Document, Job, JobAttempt, Tag, TagAlias


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_route(*args, **kwargs) -> Dict[str, Any]:
    return {"audience": "commercial", "confidence": 0.9, "reasons": ["mocked"], "candidates": ["commercial"]}


def _mock_generate(*args, **kwargs) -> Dict[str, Any]:
    return {
        "one_line_summary": "Test summary.",
        "tags": ["fibrosis", "mash"],
        "key_clues": ["key clue"],
        "decision_bullets": [
            "Executive Summary: summary",
            "Market Opportunity: market",
            "Value Proposition: value",
        ],
        "mind_map": "mindmap\n  root((Summary))\n    Executive Summary",
    }


def _mock_evaluate_pass(*args, **kwargs) -> Dict[str, Any]:
    return {
        "pass": True,
        "word_count": 20,
        "missing_sections": [],
        "fail_reasons": [],
        "fix_instructions": [],
    }


def _create_completed_job(client, db_session, monkeypatch) -> int:
    """Create a document + run job to completion; returns job_id."""
    monkeypatch.setattr(app_graph, "route_audience", _mock_route)
    monkeypatch.setattr(app_graph, "generate_content", _mock_generate)
    monkeypatch.setattr(app_graph, "evaluate_content", _mock_evaluate_pass)

    resp = client.post(
        "/web/documents",
        data={"input_text": "Test document content about clinical trials.", "audience": "auto"},
        follow_redirects=False,
    )
    job_id = int(resp.headers["location"].rstrip("/").split("/")[-1])
    client.post(f"/web/jobs/{job_id}/run", follow_redirects=False)
    return job_id


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_returns_json_with_status(client):
    resp = client.get("/health")
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"


# ── Home page ─────────────────────────────────────────────────────────────────

def test_home_page_200(client):
    resp = client.get("/web")
    assert resp.status_code == 200


def test_home_page_audience_filter(client):
    resp = client.get("/web?audience=commercial")
    assert resp.status_code == 200


# ── Redirect routes ───────────────────────────────────────────────────────────

def test_insights_redirects(client):
    resp = client.get("/web/insights", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)


def test_library_redirects(client):
    resp = client.get("/web/library", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)


# ── About page ────────────────────────────────────────────────────────────────

def test_about_page_200(client):
    resp = client.get("/web/about")
    assert resp.status_code == 200


# ── Watchlist page ────────────────────────────────────────────────────────────

def test_watchlist_page_200(client):
    resp = client.get("/web/watchlist")
    assert resp.status_code == 200


def test_watchlist_page_pagination(client):
    resp = client.get("/web/watchlist?page=2")
    assert resp.status_code == 200


# ── Documents index ───────────────────────────────────────────────────────────

def test_documents_index_200(client):
    resp = client.get("/web/documents")
    assert resp.status_code == 200


def test_documents_index_search(client):
    resp = client.get("/web/documents?q=fibrosis")
    assert resp.status_code == 200


def test_documents_index_filter_by_audience(client):
    resp = client.get("/web/documents?audience=commercial")
    assert resp.status_code == 200


# ── Create document ───────────────────────────────────────────────────────────

def test_create_document_redirects_to_job(client):
    resp = client.post(
        "/web/documents",
        data={"input_text": "Document about MASH clinical trial outcomes.", "audience": "auto"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    assert "/web/jobs/" in resp.headers["location"]


def test_create_document_with_selected_audience(client, db_session):
    resp = client.post(
        "/web/documents",
        data={"input_text": "Sales strategy document.", "audience": "commercial"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    job_id = int(resp.headers["location"].rstrip("/").split("/")[-1])
    job = db_session.get(Job, job_id)
    assert job.selected_audience == "commercial"


def test_create_document_empty_text_rejected(client):
    resp = client.post(
        "/web/documents",
        data={"input_text": "", "audience": "auto"},
        follow_redirects=False,
    )
    # Should not create a job — either 400 or redirect back to home
    assert resp.status_code != 201


# ── Job detail ────────────────────────────────────────────────────────────────

def test_job_detail_pending_200(client, db_session):
    resp = client.post(
        "/web/documents",
        data={"input_text": "Pending job document.", "audience": "auto"},
        follow_redirects=False,
    )
    job_id = int(resp.headers["location"].rstrip("/").split("/")[-1])
    resp2 = client.get(f"/web/jobs/{job_id}")
    assert resp2.status_code == 200


def test_job_detail_completed_200(client, db_session, monkeypatch):
    job_id = _create_completed_job(client, db_session, monkeypatch)
    resp = client.get(f"/web/jobs/{job_id}")
    assert resp.status_code == 200


def test_job_detail_nonexistent_404(client):
    resp = client.get("/web/jobs/999999")
    assert resp.status_code == 404


# ── Document detail ───────────────────────────────────────────────────────────

def test_document_detail_200(client, db_session, monkeypatch):
    job_id = _create_completed_job(client, db_session, monkeypatch)
    db_session.refresh(db_session.get(Job, job_id))
    job = db_session.get(Job, job_id)
    resp = client.get(f"/web/documents/{job.document_id}")
    assert resp.status_code == 200


def test_document_detail_nonexistent_404(client):
    resp = client.get("/web/documents/999999")
    assert resp.status_code == 404


# ── Run job ───────────────────────────────────────────────────────────────────

def test_run_job_redirects(client, db_session, monkeypatch):
    monkeypatch.setattr(app_graph, "route_audience", _mock_route)
    monkeypatch.setattr(app_graph, "generate_content", _mock_generate)
    monkeypatch.setattr(app_graph, "evaluate_content", _mock_evaluate_pass)

    resp = client.post(
        "/web/documents",
        data={"input_text": "Some content.", "audience": "auto"},
        follow_redirects=False,
    )
    job_id = int(resp.headers["location"].rstrip("/").split("/")[-1])
    run_resp = client.post(f"/web/jobs/{job_id}/run", follow_redirects=False)
    assert run_resp.status_code in (302, 303)


def test_run_job_already_completed_redirects(client, db_session, monkeypatch):
    """Running a completed job should redirect without re-running."""
    import app.main as app_main
    app_main._RATE_LIMIT_BUCKETS.clear()
    job_id = _create_completed_job(client, db_session, monkeypatch)
    # Clear again before the second run so the rate limiter doesn't block it
    app_main._RATE_LIMIT_BUCKETS.clear()
    resp = client.post(f"/web/jobs/{job_id}/run", follow_redirects=False)
    assert resp.status_code in (302, 303)


def test_run_job_nonexistent_redirects_not_500(client):
    resp = client.post("/web/jobs/999999/run", follow_redirects=False)
    assert resp.status_code != 500


# ── Tag aliases ───────────────────────────────────────────────────────────────

def test_tag_aliases_page_200(client):
    resp = client.get("/web/tags/aliases")
    assert resp.status_code == 200


def test_create_tag_alias_redirects(client, db_session):
    resp = client.post(
        "/web/tags/aliases",
        data={"alias": "nash_test", "canonical": "mash"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)


def test_create_tag_alias_duplicate_shows_error(client, db_session):
    client.post(
        "/web/tags/aliases",
        data={"alias": "dup_alias", "canonical": "mash"},
        follow_redirects=False,
    )
    resp2 = client.post(
        "/web/tags/aliases",
        data={"alias": "dup_alias", "canonical": "mash"},
        follow_redirects=True,
    )
    assert resp2.status_code == 200  # rendered page with error


def test_create_tag_alias_too_long_rejected(client):
    long_alias = "x" * 201
    resp = client.post(
        "/web/tags/aliases",
        data={"alias": long_alias, "canonical": "mash"},
        follow_redirects=True,
    )
    # Should not crash — either redirect with error or 4xx
    assert resp.status_code != 500


# ── Tag insights ──────────────────────────────────────────────────────────────

def test_tag_insights_page_200(client):
    resp = client.get("/web/insights/tags")
    assert resp.status_code == 200


# ── ML router insights ────────────────────────────────────────────────────────

def test_ml_router_insights_page_200(client):
    resp = client.get("/web/insights/ml-router")
    assert resp.status_code == 200


# ── Attempt detail ────────────────────────────────────────────────────────────

def test_attempt_detail_200(client, db_session, monkeypatch):
    import app.main as app_main
    app_main._RATE_LIMIT_BUCKETS.clear()
    job_id = _create_completed_job(client, db_session, monkeypatch)
    from sqlmodel import select as sm_select
    attempts = db_session.exec(
        sm_select(JobAttempt).where(JobAttempt.job_id == job_id)
    ).all()
    if attempts:
        resp = client.get(f"/web/attempts/{attempts[0].id}")
        assert resp.status_code == 200


def test_attempt_detail_nonexistent_404(client):
    resp = client.get("/web/attempts/999999")
    assert resp.status_code == 404
