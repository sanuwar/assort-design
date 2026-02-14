from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class DocumentTag(SQLModel, table=True):
    document_id: int = Field(
        foreign_key="document.id", primary_key=True, index=True
    )
    tag_id: int = Field(foreign_key="tag.id", primary_key=True, index=True)


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    source_type: str
    source_url: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    jobs: List["Job"] = Relationship(back_populates="document")
    tags: List["Tag"] = Relationship(back_populates="documents", link_model=DocumentTag)
    clues: List["DocumentClue"] = Relationship(back_populates="document")


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    selected_audience: str
    audience: Optional[str] = None
    routing_confidence: Optional[float] = None
    routing_candidates_json: str = "[]"
    routing_reasons_json: str = "[]"
    status: str = "pending"
    attempt_count: int = 0
    max_words: Optional[int] = None
    max_retries: int = 2
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Optional[Document] = Relationship(back_populates="jobs")
    attempts: List["JobAttempt"] = Relationship(back_populates="job")


class JobAttempt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    attempt_no: int
    audience: str
    agent_used: str
    generated_one_line_summary: str
    generated_tags_json: str
    generated_clues_json: str
    generated_bullets_json: str
    generated_mindmap: str
    evaluator_json: str
    passed: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)

    job: Optional[Job] = Relationship(back_populates="attempts")


class Tag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

    documents: List[Document] = Relationship(back_populates="tags", link_model=DocumentTag)


class DocumentClue(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    clue_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Optional[Document] = Relationship(back_populates="clues")


class TagAlias(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    alias: str = Field(index=True, unique=True)
    canonical: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentTagSummary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    job_id: Optional[int] = Field(default=None, foreign_key="job.id", index=True)
    domain: str = Field(index=True)
    canonical_tags_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentClaim(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    job_id: Optional[int] = Field(default=None, foreign_key="job.id", index=True)
    claim_text: str
    quote_text: str = ""
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentRiskFlag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    job_id: Optional[int] = Field(default=None, foreign_key="job.id", index=True)
    severity: str
    category: str
    text_span: str
    suggested_fix: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
