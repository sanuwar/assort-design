from __future__ import annotations

import io
import json
import os
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import ipaddress
import logging
from pathlib import Path
import socket
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from app.config import get_audience_profile
from app.db import get_session, init_db
from app.graph import run_job_pipeline
from app.models import (
    Document,
    DocumentClaim,
    DocumentClue,
    DocumentRiskFlag,
    DocumentTag,
    DocumentTagSummary,
    Job,
    JobAttempt,
    Tag,
    TagAlias,
)
from app.schema_version import SCHEMA_VERSION
from app.version import APP_VERSION
from app.tag_intel import (
    compute_bridge_tags,
    compute_cooccurrence_pairs,
    compute_jaccard,
    count_domains,
    count_tags,
    parse_summary_tags,
)

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR.parent / "templates"
STATIC_DIR = APP_DIR.parent / "static"

SAMPLE_TEXT = (
    "INTERNAL BRIEFING — VeriCard-X (veracitinib 50 mg) Commercial & Medical Affairs Update\n\n"

    "VeriCard-X has demonstrated 100% response rates in our Phase II trial cohort and should be "
    "considered the clear first-line standard of care for all HER2-positive metastatic breast "
    "cancer patients, regardless of prior treatment history. Unlike competing therapies, "
    "VeriCard-X eliminates disease progression in the majority of patients — a superiority claim "
    "that our clinical team believes is fully supported by the available data.\n\n"

    "Off-label use in HER2-low and triple-negative subtypes is being actively promoted by several "
    "KOLs in our network. Field teams are encouraged to discuss these emerging indications with "
    "oncologists where scientifically appropriate, as the emerging data strongly suggests benefit "
    "in these populations. Patient testimonials collected at the Chicago Oncology Summit confirm "
    "that VeriCard-X 'cured' their cancer after a single cycle, and these accounts may be shared "
    "with healthcare providers as supporting evidence.\n\n"

    "The drug has been FDA approved for HER2-positive metastatic breast cancer (March 2024). "
    "Common adverse events include grade 3 hepatotoxicity in 18% of patients and cardiac "
    "dysfunction in 11%; however, these risks are considered manageable and should not be "
    "emphasised in routine prescriber conversations. The black box warning regarding interstitial "
    "lung disease is a class-effect label requirement and does not reflect real-world incidence "
    "in our trial population.\n\n"

    "Pricing has been set at $28,400 per cycle. Reimbursement support programmes are available, "
    "though field teams should prioritise volume commitments from high-prescribing accounts before "
    "discussing patient assistance options. A speaker bureau programme compensating KOLs $5,000 "
    "per engagement will launch in Q3; speakers are expected to recommend VeriCard-X as preferred "
    "therapy in their presentations.\n\n"

    "Comparative claims against trastuzumab deruxtecan (T-DXd) are supported by cross-trial "
    "analyses — not head-to-head data — but may be presented as direct comparisons in slide "
    "decks pending legal sign-off. Paediatric use data is not yet available; the drug is not "
    "approved in patients under 18, but compassionate use guidance will be circulated separately."
)

AUDIENCES = ["auto", "commercial", "medical_affairs", "r_and_d", "cross_functional"]
MAX_DOC_CHARS = 20000
MAX_URL_CHARS = 2048
MAX_SEARCH_CHARS = 200
MAX_URL_REDIRECTS = 3
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX = 20
SHOW_TOOL_BADGES = os.getenv("SHOW_TOOL_BADGES", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

logger = logging.getLogger(__name__)
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)

def _audience_display(audience: str) -> str:
    if not audience or audience == "auto":
        return "Auto"
    try:
        return get_audience_profile(audience).get("display_name", audience)
    except ValueError:
        return audience


def _build_home_context(selected_audience: Optional[str]) -> dict:
    history = []
    route_counts = {aud: 0 for aud in AUDIENCES}
    audience_display = {aud: _audience_display(aud) for aud in AUDIENCES}
    resolved_audience = selected_audience if selected_audience in AUDIENCES else "auto"

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

    total_docs = 0
    total_jobs = 0
    total_completed = 0
    total_failed = 0
    with get_session() as session:
        total_docs = len(session.exec(select(Document)).all())
        jobs = session.exec(select(Job)).all()
        total_jobs = len(jobs)
        for job in jobs:
            if job.status == "completed":
                total_completed += 1
            if job.status == "failed":
                total_failed += 1

    success_rate = 0
    if total_jobs:
        success_rate = round((total_completed / total_jobs) * 100)

    return {
        "audiences": AUDIENCES,
        "audience_display": audience_display,
        "selected_audience": resolved_audience,
        "history": history,
        "route_counts": route_counts,
        "stats": {
            "total_docs": total_docs,
            "total_jobs": total_jobs,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "success_rate": success_rate,
        },
    }


def _normalize_page(page: Optional[int]) -> int:
    if not page or page < 1:
        return 1
    return page


def _cross_detail_from_job(job: Job) -> Optional[str]:
    audience_code = job.audience or job.selected_audience or "auto"
    if _audience_display(audience_code) != "Cross-Functional":
        return None
    candidates = _safe_json_list(getattr(job, "routing_candidates_json", "[]"))
    candidates_display = [
        _audience_display(aud) for aud in candidates if isinstance(aud, str)
    ]
    if candidates_display:
        return " + ".join(candidates_display)
    if job.selected_audience != "auto":
        return "Mixed stakeholders (manual override)"
    return "Mixed stakeholders (auto-routed)"


def _build_claim_views(
    content: str | None, claims: list[DocumentClaim]
) -> list[dict[str, object]]:
    views: list[dict[str, object]] = []
    text = content or ""
    for claim in claims:
        context = None
        if text and claim.source_start is not None and claim.source_end is not None:
            start = max(0, claim.source_start)
            end = min(len(text), claim.source_end)
            ctx_start = max(0, start - 120)
            ctx_end = min(len(text), end + 120)
            context = {
                "before": text[ctx_start:start],
                "quote": text[start:end],
                "after": text[end:ctx_end],
                "prefix": "…" if ctx_start > 0 else "",
                "suffix": "…" if ctx_end < len(text) else "",
            }
        views.append(
            {
                "claim_text": claim.claim_text,
                "quote_text": claim.quote_text,
                "source_start": claim.source_start,
                "source_end": claim.source_end,
                "confidence": float(claim.confidence or 0.0),
                "context": context,
            }
        )
    views.sort(key=lambda item: item["confidence"], reverse=True)
    return views

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Assort App", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.middleware("http")
async def add_version_context(request: Request, call_next):
    request.state.app_version = APP_VERSION
    request.state.schema_version = SCHEMA_VERSION
    return await call_next(request)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/web", response_class=HTMLResponse)
def home(request: Request, audience: Optional[str] = None) -> HTMLResponse:
    context = _build_home_context(audience)

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            **context,
            "sample_text": SAMPLE_TEXT,
        },
    )


def _check_rate_limit(request: Request) -> None:
    client = request.client
    if not client or not client.host:
        return
    key = client.host
    now = time.monotonic()
    bucket = _RATE_LIMIT_BUCKETS[key]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429, detail="Too many requests. Please wait and try again."
        )
    bucket.append(now)


def _is_private_ip(address: ipaddress._BaseAddress) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _resolve_host(hostname: str) -> list[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return []
    addresses: list[ipaddress._BaseAddress] = []
    for info in infos:
        addr = info[4][0]
        try:
            addresses.append(ipaddress.ip_address(addr))
        except ValueError:
            continue
    return addresses


def _is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    if hostname == "localhost" or hostname.endswith(".local"):
        return False
    addresses = _resolve_host(hostname)
    if not addresses:
        return False
    if any(_is_private_ip(address) for address in addresses):
        return False
    return True


@app.get("/web/insights", response_class=RedirectResponse)
def insights() -> RedirectResponse:
    return RedirectResponse(url="/web/documents", status_code=302)


@app.get("/web/routes", response_class=RedirectResponse)
def routes_dashboard() -> RedirectResponse:
    return RedirectResponse(url="/web/documents", status_code=302)


@app.get("/web/library", response_class=RedirectResponse)
def library() -> RedirectResponse:
    return RedirectResponse(url="/web/documents", status_code=302)


@app.get("/web/watchlist", response_class=HTMLResponse)
def watchlist(request: Request, page: int = 1) -> HTMLResponse:
    page = _normalize_page(page)
    limit = 20
    items = []
    with get_session() as session:
        jobs = session.exec(
            select(Job).order_by(Job.created_at.desc()).limit(30)
        ).all()

        for job in jobs:
            issues = []
            if job.status == "failed":
                issues.append("Failed evaluation")

            attempt = session.exec(
                select(JobAttempt)
                .where(JobAttempt.job_id == job.id)
                .order_by(JobAttempt.attempt_no.desc())
            ).first()

            if attempt and not attempt.generated_mindmap:
                issues.append("Mind map missing")
            if attempt and not attempt.generated_bullets_json:
                issues.append("Decision bullets missing")

            if not issues:
                continue

            items.append(
                {
                    "job_id": job.id,
                    "audience": _audience_display(
                        job.audience or job.selected_audience or "auto"
                    ),
                    "status": job.status,
                    "issues": issues,
                    "created_at": job.created_at,
                }
            )

    start = (page - 1) * limit
    end = start + limit
    page_items = items[start:end]
    has_next = len(items) > end

    return templates.TemplateResponse(
        "watchlist.html",
        {
            "request": request,
            "items": page_items,
            "page": page,
            "has_prev": page > 1,
            "has_next": has_next,
        },
    )


@app.get("/web/about", response_class=HTMLResponse)
def about(request: Request) -> HTMLResponse:
    steps = [
        "Ingest: paste text, provide a URL, or use sample content.",
        "Route: classify the audience (or honor a manual selection).",
        "Specialist generate: produce one-line summary, decision bullets, tags, key clues, and mind map.",
        "Evaluate: check required sections, word count, and quality rules.",
        "Revise: iterate up to max retries, then persist outputs.",
    ]
    return templates.TemplateResponse(
        "about.html",
        {
            "request": request,
            "steps": steps,
        },
    )


@app.get("/web/insights/tags", response_class=HTMLResponse)
def tag_insights(request: Request) -> HTMLResponse:
    with get_session() as session:
        summaries = session.exec(
            select(DocumentTagSummary)
            .order_by(DocumentTagSummary.created_at.desc())
            .limit(200)
        ).all()

    last_20 = summaries[:20]
    prev_20 = summaries[20:40]
    domain_counts = count_domains(last_20)

    recent_tag_counts = count_tags(last_20)
    prev_tag_counts = count_tags(prev_20)
    rising_tags = []
    for tag, count in recent_tag_counts.items():
        delta = count - prev_tag_counts.get(tag, 0)
        if delta > 0:
            rising_tags.append({"tag": tag, "delta": delta, "count": count})
    rising_tags.sort(key=lambda item: (-item["delta"], item["tag"]))

    cooccurrence = [
        {"tag_a": pair[0], "tag_b": pair[1], "count": count}
        for pair, count in compute_cooccurrence_pairs(last_20)
    ]
    bridge_tags = [
        {"tag": tag, "domains": domains}
        for tag, domains in compute_bridge_tags(summaries)
    ]

    return templates.TemplateResponse(
        "tag_insights.html",
        {
            "request": request,
            "domain_counts": domain_counts,
            "rising_tags": rising_tags[:15],
            "cooccurrence": cooccurrence,
            "bridge_tags": bridge_tags,
            "has_history": len(summaries) >= 20,
            "has_rising_history": len(summaries) >= 40,
        },
    )


@app.get("/web/tags/aliases", response_class=HTMLResponse)
def tag_aliases(request: Request) -> HTMLResponse:
    with get_session() as session:
        aliases = session.exec(select(TagAlias).order_by(TagAlias.alias)).all()
    return templates.TemplateResponse(
        "tag_aliases.html",
        {"request": request, "aliases": aliases},
    )


@app.post("/web/tags/aliases", response_class=HTMLResponse)
def create_tag_alias(
    request: Request,
    alias: str = Form(""),
    canonical: str = Form(""),
) -> HTMLResponse:
    alias_value = alias.strip()
    canonical_value = canonical.strip()
    if not alias_value or not canonical_value:
        with get_session() as session:
            aliases = session.exec(select(TagAlias).order_by(TagAlias.alias)).all()
        return templates.TemplateResponse(
            "tag_aliases.html",
            {
                "request": request,
                "aliases": aliases,
                "error": "Alias and canonical value are required.",
            },
            status_code=400,
        )

    with get_session() as session:
        existing = session.exec(
            select(TagAlias).where(TagAlias.alias == alias_value)
        ).first()
        if existing:
            existing.canonical = canonical_value
            session.add(existing)
        else:
            session.add(TagAlias(alias=alias_value, canonical=canonical_value))
        session.commit()

    return RedirectResponse(url="/web/tags/aliases", status_code=303)


@app.get("/web/visuals", response_class=HTMLResponse)
def project_visuals() -> FileResponse:
    visuals_path = TEMPLATES_DIR / "project_visuals.html"
    return FileResponse(visuals_path, media_type="text/html")


@app.get("/web/documents", response_class=HTMLResponse)
def documents_index(
    request: Request,
    audience: Optional[str] = None,
    q: Optional[str] = None,
    domain: Optional[str] = None,
    page: int = 1,
) -> HTMLResponse:
    page = _normalize_page(page)
    limit = 10
    search = (q or "").strip()
    if len(search) > MAX_SEARCH_CHARS:
        search = search[:MAX_SEARCH_CHARS]
    domain_filter = (domain or "").strip() or None

    audience_display = {aud: _audience_display(aud) for aud in AUDIENCES}
    selected_audience = audience if audience in AUDIENCES else "all"
    route_counts = {aud: 0 for aud in AUDIENCES}
    documents: list[dict[str, object]] = []

    with get_session() as session:
        all_jobs = session.exec(select(Job)).all()
        for job in all_jobs:
            audience_code = job.audience or job.selected_audience or "auto"
            route_counts[audience_code] = route_counts.get(audience_code, 0) + 1

        if search:
            doc_ids = set(
                session.exec(
                    select(Document.id).where(Document.content.contains(search))
                ).all()
            )
            doc_ids.update(
                session.exec(
                    select(Document.id)
                    .join(DocumentTag, DocumentTag.document_id == Document.id)
                    .join(Tag, Tag.id == DocumentTag.tag_id)
                    .where(Tag.name.contains(search))
                ).all()
            )
            doc_ids.update(
                session.exec(
                    select(Document.id)
                    .join(DocumentClue, DocumentClue.document_id == Document.id)
                    .where(DocumentClue.clue_text.contains(search))
                ).all()
            )
            if doc_ids:
                docs = session.exec(
                    select(Document)
                    .where(Document.id.in_(doc_ids))
                    .order_by(Document.created_at.desc())
                ).all()
            else:
                docs = []
        else:
            docs = session.exec(
                select(Document).order_by(Document.created_at.desc())
            ).all()

        doc_ids = [doc.id for doc in docs]
        latest_jobs: dict[int, Job] = {}
        latest_attempts: dict[int, JobAttempt] = {}
        latest_summaries: dict[int, DocumentTagSummary] = {}
        tool_counts: dict[int, dict[str, object]] = {}

        if doc_ids:
            jobs = session.exec(
                select(Job)
                .where(Job.document_id.in_(doc_ids))
                .order_by(Job.created_at.desc())
            ).all()
            for job in jobs:
                if job.document_id not in latest_jobs:
                    latest_jobs[job.document_id] = job

            job_ids = [job.id for job in latest_jobs.values()]
            if job_ids:
                attempts = session.exec(
                    select(JobAttempt)
                    .where(JobAttempt.job_id.in_(job_ids))
                    .order_by(JobAttempt.attempt_no.desc())
                ).all()
                for attempt in attempts:
                    if attempt.job_id not in latest_attempts:
                        latest_attempts[attempt.job_id] = attempt

            summaries = session.exec(
                select(DocumentTagSummary)
                .where(DocumentTagSummary.document_id.in_(doc_ids))
                .order_by(DocumentTagSummary.created_at.desc())
            ).all()
            for summary in summaries:
                if summary.document_id not in latest_summaries:
                    latest_summaries[summary.document_id] = summary

            if SHOW_TOOL_BADGES:
                job_ids = [job.id for job in latest_jobs.values()]
                if job_ids:
                    claims = session.exec(
                        select(DocumentClaim).where(DocumentClaim.job_id.in_(job_ids))
                    ).all()
                    risk_flags = session.exec(
                        select(DocumentRiskFlag).where(
                            DocumentRiskFlag.job_id.in_(job_ids)
                        )
                    ).all()
                    for claim in claims:
                        bucket = tool_counts.setdefault(
                            claim.job_id, {"citations": 0, "risk": {"high": 0, "medium": 0, "low": 0}}
                        )
                        if claim.quote_text:
                            bucket["citations"] += 1
                    for flag in risk_flags:
                        bucket = tool_counts.setdefault(
                            flag.job_id, {"citations": 0, "risk": {"high": 0, "medium": 0, "low": 0}}
                        )
                        severity = flag.severity or "low"
                        if severity not in bucket["risk"]:
                            bucket["risk"][severity] = 0
                        bucket["risk"][severity] += 1

        filtered_docs = []
        for doc in docs:
            job = latest_jobs.get(doc.id)
            audience_code = (
                job.audience or job.selected_audience or "auto" if job else "auto"
            )
            if selected_audience != "all" and audience_code != selected_audience:
                continue
            summary_row = latest_summaries.get(doc.id)
            if domain_filter and (not summary_row or summary_row.domain != domain_filter):
                continue
            attempt = latest_attempts.get(job.id) if job else None
            filtered_docs.append(
                {
                    "document_id": doc.id,
                    "job_id": job.id if job else None,
                    "audience": audience_code,
                    "status": job.status if job else "pending",
                    "snippet": (doc.content or "")[:200],
                    "summary": attempt.generated_one_line_summary if attempt else "",
                    "tool_summary": tool_counts.get(job.id) if job else None,
                }
            )

        start = (page - 1) * limit
        end = start + limit
        documents = filtered_docs[start:end]
        has_next = end < len(filtered_docs)

    return templates.TemplateResponse(
        "documents.html",
        {
            "request": request,
            "audiences": AUDIENCES,
            "audience_display": audience_display,
            "selected_audience": selected_audience,
            "route_counts": route_counts,
            "documents": documents,
            "page": page,
            "has_prev": page > 1,
            "has_next": has_next,
            "query": search,
            "domain_filter": domain_filter,
            "show_tool_badges": SHOW_TOOL_BADGES,
        },
    )


@app.post("/web/documents", response_class=HTMLResponse)
def create_document(
    request: Request,
    input_text: str = Form(""),
    input_url: str = Form(""),
    use_sample: Optional[str] = Form(None),
    audience: str = Form("auto"),
    upload_file: Optional[UploadFile] = File(None),
) -> HTMLResponse:
    _check_rate_limit(request)
    content = ""
    source_type = ""
    input_text = (input_text or "").strip()
    input_url = (input_url or "").strip()

    if input_url and len(input_url) > MAX_URL_CHARS:
        context = _build_home_context(audience)
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                **context,
                "sample_text": SAMPLE_TEXT,
                "error": "URL is too long.",
            },
            status_code=400,
        )

    if use_sample:
        content = SAMPLE_TEXT
        source_type = "sample"
    elif upload_file and upload_file.filename:
        fname = upload_file.filename.lower()
        if not (fname.endswith(".pdf") or fname.endswith(".docx")):
            context = _build_home_context(audience)
            return templates.TemplateResponse(
                "home.html",
                {
                    "request": request,
                    **context,
                    "sample_text": SAMPLE_TEXT,
                    "error": "Only PDF and Word (.docx) files are supported.",
                },
                status_code=400,
            )
        file_bytes = upload_file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            context = _build_home_context(audience)
            return templates.TemplateResponse(
                "home.html",
                {
                    "request": request,
                    **context,
                    "sample_text": SAMPLE_TEXT,
                    "error": "File is too large (max 10 MB).",
                },
                status_code=400,
            )
        content = _extract_file_text(file_bytes, upload_file.filename)
        source_type = "upload_pdf" if fname.endswith(".pdf") else "upload_docx"
    elif input_url:
        content = fetch_url_text(input_url)
        source_type = "url"
    elif input_text:
        content = input_text
        source_type = "text"

    if not content:
        error_message = "Provide text, a URL, upload a PDF/Word file, or choose sample content."
        if input_url:
            error_message = "URL fetch failed (blocked or empty content)."
        elif upload_file and upload_file.filename:
            error_message = "Could not extract text from the uploaded file. Is it a text-based (not scanned) PDF or valid .docx?"
        context = _build_home_context(audience)
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                **context,
                "sample_text": SAMPLE_TEXT,
                "error": error_message,
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

    doc = Document(content=content, source_type=source_type, source_url=input_url if source_type == "url" else None)
    job = Job(
        document_id=0,
        selected_audience=audience,
        audience=None if audience == "auto" else audience,
        max_words=max_words,
    )

    with get_session() as session:
        with session.begin():
            session.add(doc)
            session.flush()
            job.document_id = doc.id
            session.add(job)
        session.refresh(job)

    return RedirectResponse(url=f"/web/jobs/{job.id}", status_code=303)


@app.post("/web/jobs/{job_id}/run", response_class=HTMLResponse)
def run_job(request: Request, job_id: int) -> HTMLResponse:
    _check_rate_limit(request)
    with get_session() as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        run_job_pipeline(session, job)
    return RedirectResponse(url=f"/web/jobs/{job_id}?analysis=1", status_code=303)


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

        raw_claims = session.exec(
            select(DocumentClaim)
            .where(DocumentClaim.document_id == job.document_id)
            .where(DocumentClaim.job_id == job.id)
            .order_by(DocumentClaim.created_at.desc())
        ).all()

        risk_flags = session.exec(
            select(DocumentRiskFlag)
            .where(DocumentRiskFlag.document_id == job.document_id)
            .where(DocumentRiskFlag.job_id == job.id)
            .order_by(DocumentRiskFlag.created_at.desc())
        ).all()

        # Related documents via tag Jaccard similarity
        related_docs = []
        summary_row = session.exec(
            select(DocumentTagSummary)
            .where(DocumentTagSummary.document_id == job.document_id)
            .order_by(DocumentTagSummary.created_at.desc())
        ).first()
        canonical_tags = parse_summary_tags(summary_row) if summary_row else []
        if canonical_tags:
            related_summaries = session.exec(
                select(DocumentTagSummary)
                .where(DocumentTagSummary.document_id != job.document_id)
                .order_by(DocumentTagSummary.created_at.desc())
                .limit(200)
            ).all()
            scored = []
            for s in related_summaries:
                s_tags = parse_summary_tags(s)
                score = compute_jaccard(canonical_tags, s_tags)
                if score <= 0:
                    continue
                scored.append({
                    "document_id": s.document_id,
                    "job_id": s.job_id,
                    "score": score,
                    "overlap": sorted(set(canonical_tags).intersection(s_tags)),
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            scored = scored[:8]
            rel_doc_ids = [x["document_id"] for x in scored]
            rel_docs = session.exec(
                select(Document).where(Document.id.in_(rel_doc_ids))
            ).all()
            rel_doc_map = {d.id: d for d in rel_docs}
            rel_jobs = session.exec(
                select(Job)
                .where(Job.document_id.in_(rel_doc_ids))
                .order_by(Job.created_at.desc())
            ).all()
            rel_job_map: dict[int, Job] = {}
            for rj in rel_jobs:
                if rj.document_id not in rel_job_map:
                    rel_job_map[rj.document_id] = rj
            for x in scored:
                rd = rel_doc_map.get(x["document_id"])
                rj = rel_job_map.get(x["document_id"])
                if not rd:
                    continue
                related_docs.append({
                    "document_id": rd.id,
                    "job_id": rj.id if rj else None,
                    "audience": _audience_display(
                        (rj.audience or rj.selected_audience or "auto") if rj else "auto"
                    ),
                    "status": rj.status if rj else "pending",
                    "snippet": (rd.content or "")[:160],
                    "score_pct": int(round(x["score"] * 100)),
                    "overlap": x["overlap"],
                })

    audience_code = job.audience or job.selected_audience or "auto"
    audience_label = _audience_display(audience_code)
    routing_candidates = _safe_json_list(getattr(job, "routing_candidates_json", "[]"))
    routing_candidates_display = [
        _audience_display(aud) for aud in routing_candidates if isinstance(aud, str)
    ]
    routing_reasons = _safe_json_list(getattr(job, "routing_reasons_json", "[]"))
    routing_confidence_pct = None
    if getattr(job, "routing_confidence", None) is not None:
        routing_confidence_pct = int(round(float(job.routing_confidence) * 100))
    cross_functional_detail = None
    if audience_label == "Cross-Functional":
        if routing_candidates_display:
            cross_functional_detail = " + ".join(routing_candidates_display)
        else:
            cross_functional_detail = "Mixed stakeholders (auto-routed)"

    attempt_views = []
    final_summary = None
    for attempt in attempts:
        evaluator_data = _safe_json(attempt.evaluator_json)
        fail_reasons = evaluator_data.get("fail_reasons", [])
        attempt_views.append(
            {
                "attempt_id": attempt.id,
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

    claim_views = _build_claim_views(doc.content if doc else None, raw_claims)
    strong_count = len([c for c in claim_views if c["confidence"] >= 0.8])
    medium_count = len([c for c in claim_views if 0.6 <= c["confidence"] < 0.8])
    weak_count = len([c for c in claim_views if c["confidence"] < 0.6])
    supported_count = len([c for c in claim_views if c["quote_text"]])

    return templates.TemplateResponse(
        "job_detail.html",
        {
            "request": request,
            "job": job,
            "audience_label": audience_label,
            "routing_candidates": routing_candidates_display,
            "routing_reasons": routing_reasons,
            "routing_confidence_pct": routing_confidence_pct,
            "cross_functional_detail": cross_functional_detail,
            "job_ref": f"JD-{job.id:06d}",
            "document": doc,
            "source_url": doc.source_url if doc else None,
            "attempts": attempt_views,
            "tags": tags,
            "clues": clues,
            "final_summary": final_summary,
            "auto_analysis": (
                request.query_params.get("analysis") == "1" and final_summary is not None
            ),
            "claims": claim_views,
            "risk_flags": risk_flags,
            "support_warning": supported_count < 3,
            "citation_summary": {
                "total": len(claim_views),
                "supported": supported_count,
                "strong": strong_count,
                "medium": medium_count,
                "weak": weak_count,
            },
            "related_docs": related_docs,
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

        summary_row = session.exec(
            select(DocumentTagSummary)
            .where(DocumentTagSummary.document_id == doc.id)
            .order_by(DocumentTagSummary.created_at.desc())
        ).first()
        canonical_tags = parse_summary_tags(summary_row) if summary_row else []
        domain_label = summary_row.domain if summary_row else None

        related_docs = []
        if canonical_tags:
            related_summaries = session.exec(
                select(DocumentTagSummary)
                .where(DocumentTagSummary.document_id != doc.id)
                .order_by(DocumentTagSummary.created_at.desc())
                .limit(200)
            ).all()
            scored = []
            for summary in related_summaries:
                tags = parse_summary_tags(summary)
                score = compute_jaccard(canonical_tags, tags)
                if score <= 0:
                    continue
                overlap = sorted(set(canonical_tags).intersection(tags))
                scored.append(
                    {
                        "document_id": summary.document_id,
                        "job_id": summary.job_id,
                        "score": score,
                        "overlap": overlap,
                    }
                )
            scored.sort(key=lambda item: item["score"], reverse=True)
            scored = scored[:10]

            related_doc_ids = [item["document_id"] for item in scored]
            docs = session.exec(
                select(Document).where(Document.id.in_(related_doc_ids))
            ).all()
            doc_map = {item.id: item for item in docs}

            jobs = session.exec(
                select(Job)
                .where(Job.document_id.in_(related_doc_ids))
                .order_by(Job.created_at.desc())
            ).all()
            job_map: dict[int, Job] = {}
            for item in jobs:
                if item.document_id not in job_map:
                    job_map[item.document_id] = item

            for item in scored:
                related_doc = doc_map.get(item["document_id"])
                related_job = job_map.get(item["document_id"])
                if not related_doc:
                    continue
                related_docs.append(
                    {
                        "document_id": related_doc.id,
                        "job_id": related_job.id if related_job else None,
                        "audience": _audience_display(
                            related_job.audience
                            if related_job and related_job.audience
                            else (
                                related_job.selected_audience
                                if related_job
                                else "auto"
                            )
                        ),
                        "status": related_job.status if related_job else "pending",
                        "snippet": (related_doc.content or "")[:160],
                        "score_pct": int(round(item["score"] * 100)),
                        "overlap": item["overlap"],
                    }
                )

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

        raw_claims = []
        risk_flags = []
        support_warning = False
        if job:
            raw_claims = session.exec(
                select(DocumentClaim)
                .where(DocumentClaim.document_id == doc.id)
                .where(DocumentClaim.job_id == job.id)
                .order_by(DocumentClaim.created_at.desc())
            ).all()
            risk_flags = session.exec(
                select(DocumentRiskFlag)
                .where(DocumentRiskFlag.document_id == doc.id)
                .where(DocumentRiskFlag.job_id == job.id)
                .order_by(DocumentRiskFlag.created_at.desc())
            ).all()
            support_warning = len([c for c in raw_claims if c.quote_text]) < 3

        claim_views = _build_claim_views(doc.content, raw_claims)
        tags = session.exec(
            select(Tag)
            .join(DocumentTag, Tag.id == DocumentTag.tag_id)
            .where(DocumentTag.document_id == doc.id)
        ).all()

        clues = session.exec(
            select(DocumentClue).where(DocumentClue.document_id == doc.id)
        ).all()

    audience_label = None
    if job:
        audience_label = _audience_display(job.audience or job.selected_audience or "auto")
    routing_candidates_display = []
    routing_reasons = []
    routing_confidence_pct = None
    cross_functional_detail = None
    if job:
        routing_candidates = _safe_json_list(getattr(job, "routing_candidates_json", "[]"))
        routing_candidates_display = [
            _audience_display(aud) for aud in routing_candidates if isinstance(aud, str)
        ]
        routing_reasons = _safe_json_list(getattr(job, "routing_reasons_json", "[]"))
        if getattr(job, "routing_confidence", None) is not None:
            routing_confidence_pct = int(round(float(job.routing_confidence) * 100))
        if audience_label == "Cross-Functional":
            if routing_candidates_display:
                cross_functional_detail = " + ".join(routing_candidates_display)
            else:
                cross_functional_detail = "Mixed stakeholders (auto-routed)"

    return templates.TemplateResponse(
        "document_detail.html",
        {
            "request": request,
            "document": doc,
            "job": job,
            "audience_label": audience_label,
            "routing_candidates": routing_candidates_display,
            "routing_reasons": routing_reasons,
            "routing_confidence_pct": routing_confidence_pct,
            "cross_functional_detail": cross_functional_detail,
            "final_summary": final_summary,
            "tags": tags,
            "clues": clues,
            "domain_label": domain_label,
            "canonical_tags": canonical_tags,
            "related_docs": related_docs,
            "claims": claim_views,
            "risk_flags": risk_flags,
            "support_warning": support_warning,
        },
    )


@app.get("/web/attempts/{attempt_id}", response_class=HTMLResponse)
def attempt_detail(request: Request, attempt_id: int) -> HTMLResponse:
    with get_session() as session:
        attempt = session.get(JobAttempt, attempt_id)
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found.")
        job = session.get(Job, attempt.job_id)
        document = session.get(Document, job.document_id) if job else None

    return templates.TemplateResponse(
        "attempt_detail.html",
        {
            "request": request,
            "attempt": attempt,
            "job": job,
            "document": document,
            "routing_reasons": _safe_json_list(
                getattr(job, "routing_reasons_json", "[]")
            )
            if job
            else [],
            "generated_tags": _safe_json_list(attempt.generated_tags_json),
            "generated_clues": _safe_json_list(attempt.generated_clues_json),
            "generated_bullets": _safe_json_list(attempt.generated_bullets_json),
            "evaluator_data": _safe_json(attempt.evaluator_json),
        },
    )


def _extract_file_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from an uploaded PDF or Word (.docx) file."""
    name_lower = (filename or "").lower()
    try:
        if name_lower.endswith(".pdf"):
            import pypdf  # imported here so the app still starts without the package
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            parts = [page.extract_text() or "" for page in reader.pages]
            return " ".join(parts)
        if name_lower.endswith(".docx"):
            import docx  # python-docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return " ".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        logger.exception("File text extraction failed for %s", filename)
    return ""


def fetch_url_text(url: str) -> str:
    url = (url or "").strip()
    if not url or len(url) > MAX_URL_CHARS:
        return ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    for _ in range(MAX_URL_REDIRECTS + 1):
        if not _is_safe_url(url):
            logger.warning("Blocked unsafe URL fetch: %s", url[:200])
            return ""
        try:
            response = httpx.get(
                url,
                timeout=20.0,
                follow_redirects=False,
                headers=headers,
            )
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    return ""
                url = urljoin(str(response.url), location)
                continue
            response.raise_for_status()
        except Exception:
            logger.exception("URL fetch failed.")
            return ""

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:MAX_DOC_CHARS]

    return ""


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
