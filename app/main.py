from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from app.config import get_audience_profile
from app.db import get_session, init_db
from app.graph import run_job_pipeline
from app.models import Document, DocumentClue, DocumentTag, Job, JobAttempt, Tag

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR.parent / "templates"
STATIC_DIR = APP_DIR.parent / "static"

SAMPLE_TEXT = (
    "Asort Design is developing a modular analysis workflow for biomedical content. "
    "The workflow routes content to audience-specific specialists who generate a one-line "
    "summary, key clues, decision bullets, and a mind map, followed by evaluation and revision "
    "to meet constraints."
)

AUDIENCES = ["auto", "commercial", "medical_affairs", "r_and_d", "cross_functional"]
MAX_DOC_CHARS = 20000

def _audience_display(audience: str) -> str:
    if not audience or audience == "auto":
        return "Auto"
    try:
        return get_audience_profile(audience).get("display_name", audience)
    except ValueError:
        return audience

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Asort Design", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/web", response_class=HTMLResponse)
def home(request: Request, audience: Optional[str] = None) -> HTMLResponse:
    history = []
    route_counts = {aud: 0 for aud in AUDIENCES}
    audience_display = {aud: _audience_display(aud) for aud in AUDIENCES}
    selected_audience = audience if audience in AUDIENCES else "auto"
    with get_session() as session:
        recent_docs = session.exec(
            select(Document).order_by(Document.created_at.desc()).limit(7)
        ).all()

        for doc in recent_docs:
            job = session.exec(
                select(Job)
                .where(Job.document_id == doc.id)
                .order_by(Job.created_at.desc())
            ).first()

            attempt = None
            if job:
                attempt = session.exec(
                    select(JobAttempt)
                    .where(JobAttempt.job_id == job.id)
                    .order_by(JobAttempt.attempt_no.desc())
                ).first()

            history.append(
                {
                    "job_id": job.id if job else None,
                    "document_id": doc.id,
                    "snippet": (doc.content or "")[:180],
                    "summary": attempt.generated_one_line_summary if attempt else "",
                    "audience": job.audience or job.selected_audience if job else "auto",
                    "status": job.status if job else "pending",
                }
            )

        jobs = session.exec(select(Job)).all()
        for job in jobs:
            audience = job.audience or job.selected_audience or "auto"
            route_counts[audience] = route_counts.get(audience, 0) + 1

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "audiences": AUDIENCES,
            "audience_display": audience_display,
            "selected_audience": selected_audience,
            "sample_text": SAMPLE_TEXT,
            "history": history,
            "route_counts": route_counts,
        },
    )


@app.get("/web/documents", response_class=HTMLResponse)
def documents_index(request: Request, audience: Optional[str] = None) -> HTMLResponse:
    audience_display = {aud: _audience_display(aud) for aud in AUDIENCES}
    selected_audience = audience if audience in AUDIENCES else "all"
    route_counts = {aud: 0 for aud in AUDIENCES}
    documents = []

    with get_session() as session:
        all_jobs = session.exec(select(Job)).all()
        for job in all_jobs:
            audience_code = job.audience or job.selected_audience or "auto"
            route_counts[audience_code] = route_counts.get(audience_code, 0) + 1

        recent_docs = session.exec(
            select(Document).order_by(Document.created_at.desc()).limit(50)
        ).all()

        for doc in recent_docs:
            job = session.exec(
                select(Job)
                .where(Job.document_id == doc.id)
                .order_by(Job.created_at.desc())
            ).first()
            if not job:
                continue

            audience_code = job.audience or job.selected_audience or "auto"
            if selected_audience != "all" and audience_code != selected_audience:
                continue

            attempt = session.exec(
                select(JobAttempt)
                .where(JobAttempt.job_id == job.id)
                .order_by(JobAttempt.attempt_no.desc())
            ).first()

            documents.append(
                {
                    "document_id": doc.id,
                    "job_id": job.id,
                    "audience": audience_code,
                    "status": job.status,
                    "snippet": (doc.content or "")[:200],
                    "summary": attempt.generated_one_line_summary if attempt else "",
                }
            )

    return templates.TemplateResponse(
        "documents.html",
        {
            "request": request,
            "audiences": AUDIENCES,
            "audience_display": audience_display,
            "selected_audience": selected_audience,
            "route_counts": route_counts,
            "documents": documents,
        },
    )


@app.post("/web/documents", response_class=HTMLResponse)
def create_document(
    request: Request,
    input_text: str = Form(""),
    input_url: str = Form(""),
    use_sample: Optional[str] = Form(None),
    audience: str = Form("auto"),
) -> HTMLResponse:
    content = ""
    source_type = ""

    if use_sample:
        content = SAMPLE_TEXT
        source_type = "sample"
    elif input_url.strip():
        content = fetch_url_text(input_url.strip())
        source_type = "url"
    elif input_text.strip():
        content = input_text.strip()
        source_type = "text"

    if not content:
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "audiences": AUDIENCES,
                "audience_display": {aud: _audience_display(aud) for aud in AUDIENCES},
                "selected_audience": "auto",
                "sample_text": SAMPLE_TEXT,
                "history": [],
                "route_counts": {aud: 0 for aud in AUDIENCES},
                "error": "Provide text, URL, or choose sample content.",
            },
            status_code=400,
        )

    if audience not in AUDIENCES:
        raise HTTPException(status_code=400, detail="Invalid audience selection.")

    max_words = None
    if audience != "auto":
        max_words = get_audience_profile(audience).get("default_max_words")

    if len(content) > MAX_DOC_CHARS:
        content = content[:MAX_DOC_CHARS]

    doc = Document(content=content, source_type=source_type)
    job = Job(
        document_id=0,
        selected_audience=audience,
        audience=None if audience == "auto" else audience,
        max_words=max_words,
    )

    with get_session() as session:
        session.add(doc)
        session.commit()
        session.refresh(doc)

        job.document_id = doc.id
        session.add(job)
        session.commit()
        session.refresh(job)

    return RedirectResponse(url=f"/web/jobs/{job.id}", status_code=303)


@app.post("/web/jobs/{job_id}/run", response_class=HTMLResponse)
def run_job(job_id: int) -> HTMLResponse:
    with get_session() as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        run_job_pipeline(session, job)
    return RedirectResponse(url=f"/web/jobs/{job_id}", status_code=303)


@app.get("/web/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int) -> HTMLResponse:
    with get_session() as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        doc = session.get(Document, job.document_id)

        attempts = session.exec(
            select(JobAttempt)
            .where(JobAttempt.job_id == job.id)
            .order_by(JobAttempt.attempt_no)
        ).all()

        tags = session.exec(
            select(Tag)
            .join(DocumentTag, Tag.id == DocumentTag.tag_id)
            .where(DocumentTag.document_id == job.document_id)
        ).all()

        clues = session.exec(
            select(DocumentClue).where(DocumentClue.document_id == job.document_id)
        ).all()

    audience_code = job.audience or job.selected_audience or "auto"
    audience_label = _audience_display(audience_code)

    attempt_views = []
    final_summary = None
    for attempt in attempts:
        evaluator_data = _safe_json(attempt.evaluator_json)
        fail_reasons = evaluator_data.get("fail_reasons", [])
        attempt_views.append(
            {
                "attempt_no": attempt.attempt_no,
                "passed": attempt.passed,
                "summary_preview": (attempt.generated_one_line_summary or "")[:200],
                "fail_reasons": fail_reasons,
            }
        )
        if attempt.passed:
            mind_map = (attempt.generated_mindmap or "").strip()
            final_summary = {
                "one_line_summary": attempt.generated_one_line_summary,
                "decision_bullets": _safe_json_list(attempt.generated_bullets_json),
                "mind_map": mind_map,
            }
            if final_summary["decision_bullets"] and not final_summary["mind_map"]:
                final_summary["mind_map"] = _fallback_mind_map_from_bullets(
                    final_summary["decision_bullets"]
                )

    return templates.TemplateResponse(
        "job_detail.html",
        {
            "request": request,
            "job": job,
            "audience_label": audience_label,
            "job_ref": f"JD-{job.id:06d}",
            "document": doc,
            "attempts": attempt_views,
            "tags": tags,
            "clues": clues,
            "final_summary": final_summary,
        },
    )


@app.get("/web/documents/{doc_id}", response_class=HTMLResponse)
def document_detail(request: Request, doc_id: int) -> HTMLResponse:
    with get_session() as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        job = session.exec(
            select(Job)
            .where(Job.document_id == doc.id)
            .order_by(Job.created_at.desc())
        ).first()

        final_summary = None
        if job:
            attempt = session.exec(
                select(JobAttempt)
                .where(JobAttempt.job_id == job.id)
                .where(JobAttempt.passed == True)  # noqa: E712
                .order_by(JobAttempt.attempt_no.desc())
            ).first()
            if attempt:
                mind_map = (attempt.generated_mindmap or "").strip()
                final_summary = {
                    "one_line_summary": attempt.generated_one_line_summary,
                    "decision_bullets": _safe_json_list(attempt.generated_bullets_json),
                    "mind_map": mind_map,
                }
                if final_summary["decision_bullets"] and not final_summary["mind_map"]:
                    final_summary["mind_map"] = _fallback_mind_map_from_bullets(
                        final_summary["decision_bullets"]
                    )

        tags = session.exec(
            select(Tag)
            .join(DocumentTag, Tag.id == DocumentTag.tag_id)
            .where(DocumentTag.document_id == doc.id)
        ).all()

        clues = session.exec(
            select(DocumentClue).where(DocumentClue.document_id == doc.id)
        ).all()

    return templates.TemplateResponse(
        "document_detail.html",
        {
            "request": request,
            "document": doc,
            "job": job,
            "final_summary": final_summary,
            "tags": tags,
            "clues": clues,
        },
    )


def fetch_url_text(url: str) -> str:
    try:
        response = httpx.get(url, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    return text[:MAX_DOC_CHARS]


def _safe_json(payload: str) -> dict:
    try:
        return json.loads(payload) if payload else {}
    except json.JSONDecodeError:
        return {}


def _safe_json_list(payload: str) -> list:
    data = _safe_json(payload)
    if isinstance(data, list):
        return data
    return []


def _fallback_mind_map_from_bullets(bullets: list[str]) -> str:
    lines = ["mindmap", "  root((Summary))"]
    for bullet in bullets[:5]:
        title = str(bullet).split(":", 1)[0].strip()
        if not title:
            title = str(bullet).strip()[:40]
        if title:
            lines.append(f"    {title}")
    return "\n".join(lines)
