from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from app.config import get_audience_profile
from app.db import get_session, init_db
from app.graph import run_job_pipeline
from app.models import Document, DocumentTag, Job, JobAttempt, QuizQuestion, Tag

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR.parent / "templates"

SAMPLE_TEXT = (
    "Asort Design is developing a modular analysis workflow for biomedical content. "
    "The workflow routes content to audience-specific specialists who generate summaries, "
    "tags, and quiz questions, followed by evaluation and revision to meet constraints."
)

AUDIENCES = ["auto", "commercial", "medical_affairs", "r_and_d", "cross_functional"]
MAX_DOC_CHARS = 20000

app = FastAPI(title="Asort Design")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/web", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "audiences": AUDIENCES,
            "sample_text": SAMPLE_TEXT,
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
                "sample_text": SAMPLE_TEXT,
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

        questions = session.exec(
            select(QuizQuestion).where(QuizQuestion.document_id == job.document_id)
        ).all()

    return templates.TemplateResponse(
        "job_detail.html",
        {
            "request": request,
            "job": job,
            "document": doc,
            "attempts": attempts,
            "tags": tags,
            "questions": questions,
        },
    )


@app.get("/web/documents/{doc_id}", response_class=HTMLResponse)
def document_detail(request: Request, doc_id: int) -> HTMLResponse:
    with get_session() as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        tags = session.exec(
            select(Tag)
            .join(DocumentTag, Tag.id == DocumentTag.tag_id)
            .where(DocumentTag.document_id == doc.id)
        ).all()

        questions = session.exec(
            select(QuizQuestion).where(QuizQuestion.document_id == doc.id)
        ).all()

    return templates.TemplateResponse(
        "document_detail.html",
        {
            "request": request,
            "document": doc,
            "tags": tags,
            "questions": questions,
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
