"""Objective timing and disfluency features from word-level timestamps."""

import re
from typing import Any

FILLER_WORDS = {"um", "uh", "like", "so", "actually", "basically", "literally"}
PAUSE_THRESHOLD_SECONDS = 1.5


def _normalized_word(word: str) -> str:
    """Remove surrounding punctuation while keeping internal apostrophes."""
    return re.sub(r"(^[^\w']+|[^\w']+$)", "", word.strip().lower(), flags=re.UNICODE)


def extract_features(words: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute WPM, filler occurrences, and pauses from timed words."""
    if not words:
        return {
            "wpm": 0.0,
            "filler_count": 0,
            "filler_timestamps": [],
            "pause_count": 0,
            "pause_timestamps": [],
        }

    duration = float(words[-1]["end"]) - float(words[0]["start"])
    wpm = len(words) / (duration / 60.0) if duration > 0 else 0.0

    filler_timestamps: list[dict[str, Any]] = []
    normalized = [_normalized_word(str(item.get("word", ""))) for item in words]
    i = 0
    while i < len(words):
        # Treat the two-token phrase as one filler occurrence and one time span.
        if normalized[i] == "you" and i + 1 < len(words) and normalized[i + 1] == "know":
            filler_timestamps.append(
                {
                    "word": "you know",
                    "start_sec": float(words[i]["start"]),
                    "end_sec": float(words[i + 1]["end"]),
                }
            )
            i += 2
            continue
        if normalized[i] in FILLER_WORDS:
            filler_timestamps.append(
                {
                    "word": normalized[i],
                    "start_sec": float(words[i]["start"]),
                    "end_sec": float(words[i]["end"]),
                }
            )
        i += 1

    pause_timestamps: list[dict[str, float]] = []
    for previous, following in zip(words, words[1:]):
        start_sec = float(previous["end"])
        end_sec = float(following["start"])
        pause_duration = end_sec - start_sec
        if pause_duration > PAUSE_THRESHOLD_SECONDS:
            pause_timestamps.append(
                {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": pause_duration,
                }
            )

    return {
        "wpm": float(wpm),
        "filler_count": len(filler_timestamps),
        "filler_timestamps": filler_timestamps,
        "pause_count": len(pause_timestamps),
        "pause_timestamps": pause_timestamps,
    }
