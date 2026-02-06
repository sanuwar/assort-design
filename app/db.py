from __future__ import annotations

from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "asort_design.db"
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
