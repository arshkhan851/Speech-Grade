"""API response schemas."""

from pydantic import BaseModel


class SpeechUploadResponse(BaseModel):
    id: int
    title: str
    audio_path: str
    speech_type: str
    message: str = "Speech uploaded and transcribed."
    scores: dict[str, float] | None = None
    flaw_count: int | None = None
