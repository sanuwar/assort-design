from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlmodel import SQLModel, Session, create_engine

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "asort_design.db"
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)
    _ensure_job_columns()
    _ensure_indexes()


def get_session() -> Session:
    return Session(engine)


def _ensure_job_columns() -> None:
    with engine.begin() as conn:
        result = conn.execute(text("PRAGMA table_info(job)")).fetchall()
        columns = {row[1] for row in result}

        if "routing_confidence" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN routing_confidence FLOAT"))
        if "routing_candidates_json" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN routing_candidates_json TEXT"))
        if "routing_reasons_json" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN routing_reasons_json TEXT"))


def _ensure_indexes() -> None:
    with engine.begin() as conn:
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_job_document_id ON job (document_id)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_jobattempt_job_id "
                "ON jobattempt (job_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documentclue_document_id "
                "ON documentclue (document_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documenttag_document_id "
                "ON documenttag (document_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documenttag_tag_id "
                "ON documenttag (tag_id)"
            )
        )
