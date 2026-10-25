"""SQLite engine and session setup."""

from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'speechgrade.db').as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """Yield a database session for FastAPI dependencies."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    # create_all does not add columns to an existing table; preserve older local DBs.
    if "word_timestamps" not in {column["name"] for column in inspect(engine).get_columns("features")}:
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE features ADD COLUMN word_timestamps JSON NOT NULL DEFAULT '[]'")
            )
