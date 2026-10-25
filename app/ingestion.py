"""Local word-level transcription with faster-whisper."""

from pathlib import Path
from threading import Lock
from typing import Any

from faster_whisper import WhisperModel

# Keep this small for development; use "small" for improved accuracy when needed.
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

_model: WhisperModel | None = None
_model_lock = Lock()


def _get_model() -> WhisperModel:
    """Load the model once on first use and reuse it for subsequent uploads."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe_audio(audio_path: str) -> dict[str, Any]:
    """Transcribe an audio file and return text plus word timestamps in seconds."""
    path = Path(audio_path)
    try:
        model = _get_model()
        segments, _info = model.transcribe(str(path), word_timestamps=True)
        text_parts: list[str] = []
        words: list[dict[str, float | str]] = []

        for segment in segments:
            segment_text = segment.text.strip()
            if segment_text:
                text_parts.append(segment_text)
            for word in segment.words or []:
                words.append(
                    {
                        "word": word.word.strip(),
                        "start": float(word.start),
                        "end": float(word.end),
                    }
                )

        return {"text": " ".join(text_parts), "words": words}
    except Exception as exc:
        raise RuntimeError(f"Transcription failed for audio file '{path}': {exc}") from exc
