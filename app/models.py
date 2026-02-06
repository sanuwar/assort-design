from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class DocumentTag(SQLModel, table=True):
    document_id: int = Field(foreign_key="document.id", primary_key=True)
    tag_id: int = Field(foreign_key="tag.id", primary_key=True)


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    source_type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    jobs: List["Job"] = Relationship(back_populates="document")
    tags: List["Tag"] = Relationship(back_populates="documents", link_model=DocumentTag)
    quiz_questions: List["QuizQuestion"] = Relationship(back_populates="document")


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    selected_audience: str
    audience: Optional[str] = None
    status: str = "pending"
    attempt_count: int = 0
    max_words: Optional[int] = None
    max_retries: int = 2
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Optional[Document] = Relationship(back_populates="jobs")
    attempts: List["JobAttempt"] = Relationship(back_populates="job")


class JobAttempt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id")
    attempt_no: int
    audience: str
    agent_used: str
    generated_summary: str
    generated_tags_json: str
    generated_quiz_json: str
    evaluator_json: str
    passed: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)

    job: Optional[Job] = Relationship(back_populates="attempts")


class Tag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

    documents: List[Document] = Relationship(back_populates="tags", link_model=DocumentTag)


class QuizQuestion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    question_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Optional[Document] = Relationship(back_populates="quiz_questions")
