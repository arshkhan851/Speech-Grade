"""FastAPI entry point for the SpeechGrade scaffold."""

from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .db import PROJECT_ROOT, get_db, init_db
from .features import extract_features
from .ingestion import transcribe_audio
from .models import Feature, Flaw, Speech
from .rubric import score_speech
from .schemas import SpeechUploadResponse

UPLOAD_DIR = PROJECT_ROOT / "data"
ALLOWED_SPEECH_TYPES = {"ideal", "flawed", "user_submission"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="SpeechGrade", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/speeches", response_model=SpeechUploadResponse, status_code=201)
async def upload_speech(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    speaker_label: str | None = Form(default=None),
    speech_type: str = Form(default="user_submission"),
    db: Session = Depends(get_db),
) -> SpeechUploadResponse:
    """Store an uploaded audio file and create its initial speech record."""
    if speech_type not in ALLOWED_SPEECH_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported speech_type.")

    original_name = Path(file.filename or "speech.audio").name
    suffix = Path(original_name).suffix.lower()[:12]
    stored_name = f"{uuid4().hex}{suffix}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / stored_name

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)

        speech = Speech(
            title=(title or Path(original_name).stem or "Untitled speech"),
            speaker_label=speaker_label,
            audio_path=destination.relative_to(PROJECT_ROOT).as_posix(),
            duration_seconds=0.0,
            transcript="",
            speech_type=speech_type,
        )
        db.add(speech)
        db.commit()
        db.refresh(speech)
    except Exception:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    # The upload and speech row are committed first so a model/download/inference
    # failure does not discard the user's file or its database record.
    try:
        transcription = transcribe_audio(str(PROJECT_ROOT / speech.audio_path))
        extracted = extract_features(transcription["words"])
        speech.transcript = transcription["text"]
        speech.duration_seconds = max(
            (float(word["end"]) for word in transcription["words"]), default=0.0
        )
        db.add(
            Feature(
                speech_id=speech.id,
                wpm=extracted["wpm"],
                filler_count=extracted["filler_count"],
                pause_count=extracted["pause_count"],
                filler_timestamps=extracted["filler_timestamps"],
                pause_timestamps=extracted["pause_timestamps"],
                word_timestamps=transcription["words"],
            )
        )
        db.add(speech)
        db.commit()
        message = "Speech uploaded and transcribed."
    except Exception as exc:
        db.rollback()
        message = f"Speech uploaded, but transcription failed: {exc}"

    response = SpeechUploadResponse(
        id=speech.id,
        title=speech.title,
        audio_path=speech.audio_path,
        speech_type=speech.speech_type,
        message=message,
    )
    if message == "Speech uploaded and transcribed.":
        try:
            scored = score_speech(speech.id, db)
            response.message = "Speech uploaded, transcribed, and scored."
            response.scores = scored["scores"]
            response.flaw_count = len(scored["flaws"])
        except Exception as exc:
            db.rollback()
            response.message = f"Speech uploaded and transcribed, but scoring failed: {exc}"
            response.scores = None
            response.flaw_count = None
    return response
