"""SQLAlchemy models for SpeechGrade's evaluation history."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Speech(Base):
    __tablename__ = "speeches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    speaker_label: Mapped[str | None] = mapped_column(String, nullable=True)
    audio_path: Mapped[str] = mapped_column(String, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    speech_type: Mapped[str] = mapped_column(String, nullable=False)

    features: Mapped[list["Feature"]] = relationship(back_populates="speech", cascade="all, delete-orphan")
    rubric_scores: Mapped[list["RubricScore"]] = relationship(back_populates="speech", cascade="all, delete-orphan")
    flaws: Mapped[list["Flaw"]] = relationship(back_populates="speech", cascade="all, delete-orphan")
    eval_runs: Mapped[list["EvalRun"]] = relationship(back_populates="speech", cascade="all, delete-orphan")


class Feature(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    speech_id: Mapped[int] = mapped_column(ForeignKey("speeches.id"), nullable=False)
    wpm: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    filler_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pause_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    filler_timestamps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    pause_timestamps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    word_timestamps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    speech: Mapped[Speech] = relationship(back_populates="features")


class RubricScore(Base):
    __tablename__ = "rubric_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    speech_id: Mapped[int] = mapped_column(ForeignKey("speeches.id"), nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rubric_version: Mapped[str] = mapped_column(String, nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    speech: Mapped[Speech] = relationship(back_populates="rubric_scores")


class Flaw(Base):
    __tablename__ = "flaws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    speech_id: Mapped[int] = mapped_column(ForeignKey("speeches.id"), nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)
    flaw_type: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False)

    speech: Mapped[Speech] = relationship(back_populates="flaws")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    speech_id: Mapped[int] = mapped_column(ForeignKey("speeches.id"), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    raw_llm_response: Mapped[str] = mapped_column(Text, nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    speech: Mapped[Speech] = relationship(back_populates="eval_runs")
