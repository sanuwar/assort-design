"""Tests for app.models — _utcnow, field defaults, and DB-level constraints."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from app.models import (
    Document,
    DocumentClue,
    DocumentTag,
    Job,
    JobAttempt,
    Tag,
    TagAlias,
    _utcnow,
)


# ── _utcnow ───────────────────────────────────────────────────────────────────

def test_utcnow_returns_aware_datetime():
    dt = _utcnow()
    assert dt.tzinfo is not None


def test_utcnow_is_utc():
    dt = _utcnow()
    # UTC offset must be zero
    assert dt.utcoffset().total_seconds() == 0


def test_utcnow_advances_with_time():
    import time
    t1 = _utcnow()
    time.sleep(0.01)
    t2 = _utcnow()
    assert t2 > t1


# ── Document model ────────────────────────────────────────────────────────────

def test_document_created_at_auto_populated(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="hello", source_type="text")
        session.add(doc)
        session.commit()
        session.refresh(doc)
        assert doc.created_at is not None
        assert isinstance(doc.created_at, datetime)


def test_document_source_url_is_optional(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="hello", source_type="text", source_url=None)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        assert doc.source_url is None


def test_document_id_autoincrement(test_engine):
    with Session(test_engine) as session:
        d1 = Document(content="a", source_type="text")
        d2 = Document(content="b", source_type="text")
        session.add(d1)
        session.add(d2)
        session.commit()
        session.refresh(d1)
        session.refresh(d2)
        assert d1.id != d2.id


# ── Job model ─────────────────────────────────────────────────────────────────

def test_job_default_status_is_pending(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="content", source_type="text")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        job = Job(document_id=doc.id, selected_audience="auto")
        session.add(job)
        session.commit()
        session.refresh(job)
        assert job.status == "pending"


def test_job_default_max_retries(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="content", source_type="text")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        job = Job(document_id=doc.id, selected_audience="commercial")
        session.add(job)
        session.commit()
        session.refresh(job)
        assert job.max_retries == 2


def test_job_attempt_count_starts_at_zero(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="content", source_type="text")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        job = Job(document_id=doc.id, selected_audience="auto")
        session.add(job)
        session.commit()
        session.refresh(job)
        assert job.attempt_count == 0


def test_job_routing_json_defaults(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="content", source_type="text")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        job = Job(document_id=doc.id, selected_audience="auto")
        session.add(job)
        session.commit()
        session.refresh(job)
        assert job.routing_candidates_json == "[]"
        assert job.routing_reasons_json == "[]"


# ── Tag model ─────────────────────────────────────────────────────────────────

def test_tag_name_persisted(test_engine):
    with Session(test_engine) as session:
        tag = Tag(name="fibrosis")
        session.add(tag)
        session.commit()
        session.refresh(tag)
        assert tag.name == "fibrosis"


def test_tag_name_unique(test_engine):
    from sqlalchemy.exc import IntegrityError
    with Session(test_engine) as session:
        session.add(Tag(name="unique-tag-x"))
        session.commit()
    with Session(test_engine) as session:
        session.add(Tag(name="unique-tag-x"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


# ── TagAlias model ────────────────────────────────────────────────────────────

def test_tag_alias_persisted(test_engine):
    with Session(test_engine) as session:
        alias = TagAlias(alias="nash", canonical="mash")
        session.add(alias)
        session.commit()
        session.refresh(alias)
        assert alias.alias == "nash"
        assert alias.canonical == "mash"


def test_tag_alias_created_at_auto(test_engine):
    with Session(test_engine) as session:
        alias = TagAlias(alias="rct2", canonical="clinical trial")
        session.add(alias)
        session.commit()
        session.refresh(alias)
        assert alias.created_at is not None


# ── DocumentTag link model ────────────────────────────────────────────────────

def test_document_tag_link_persisted(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="doc", source_type="text")
        tag = Tag(name="link-tag")
        session.add(doc)
        session.add(tag)
        session.commit()
        session.refresh(doc)
        session.refresh(tag)

        link = DocumentTag(document_id=doc.id, tag_id=tag.id)
        session.add(link)
        session.commit()

        found = session.exec(
            select(DocumentTag).where(DocumentTag.document_id == doc.id)
        ).first()
        assert found is not None
        assert found.tag_id == tag.id


# ── DocumentClue model ────────────────────────────────────────────────────────

def test_document_clue_persisted(test_engine):
    with Session(test_engine) as session:
        doc = Document(content="doc", source_type="text")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        clue = DocumentClue(document_id=doc.id, clue_text="Relevant signal here.")
        session.add(clue)
        session.commit()
        session.refresh(clue)
        assert clue.clue_text == "Relevant signal here."
        assert clue.created_at is not None
