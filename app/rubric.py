"""LLM rubric prompt construction, validation, and persistence."""

import json
import logging
import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .llm_client import call_llm
from .models import EvalRun, Feature, Flaw, RubricScore, Speech

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "v1"
CATEGORIES = ("delivery", "structure", "content", "engagement")

SYSTEM_PROMPT = """You are a speech evaluation assistant. You will receive a speech transcript with word-level timestamps, plus pre-computed delivery metrics
(words-per-minute, filler word occurrences with timestamps, pause locations with timestamps).

Score the speech on exactly these 4 categories, each 0-10:
- delivery: pacing, filler word frequency, pause usage, clarity
- structure: intro/body/conclusion clarity, logical flow between points
- content: strength and clarity of the argument or message, use of evidence/examples
- engagement: variation in tone/phrasing, confidence markers, audience connection

For each category, identify 1-3 SPECIFIC flaws. Every flaw MUST include a real
timestamp from the transcript or the provided feature data -- never invent a
timestamp, and never give a flaw without one.

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:

{
  \"scores\": {
    \"delivery\": <0-10>,
    \"structure\": <0-10>,
    \"content\": <0-10>,
    \"engagement\": <0-10>
  },
  \"flaws\": [
    {
      \"category\": \"<one of the 4 categories>\",
      \"timestamp_sec\": <float>,
      \"flaw_type\": \"<short_snake_case_label>\",
      \"note\": \"<specific, actionable feedback, one sentence>\"
    }
  ],
  \"overall_summary\": \"<2-3 sentence summary of the speech's main strengths and
weaknesses>\"
}

If the transcript is too short or unclear to evaluate a category fairly, still return
a score (default to 5) and explain why in that category's flaw note."""


def _user_prompt(speech: Speech, feature: Feature) -> str:
    return """Evaluate this speech using the rubric.

Transcript:
{transcript}

Word-level timestamps (seconds):
{words}

Pre-computed delivery metrics:
- Duration (seconds): {duration:.2f}
- Words per minute: {wpm:.2f}
- Filler occurrences (timestamps in seconds): {fillers}
- Pause locations (timestamps in seconds): {pauses}

Use timestamps only from the transcript or provided feature data. If the transcript is too short or unclear to evaluate fairly, explain the limitation.""".format(
        transcript=speech.transcript,
        words=json.dumps(feature.word_timestamps, ensure_ascii=False),
        duration=speech.duration_seconds,
        wpm=feature.wpm,
        fillers=json.dumps(feature.filler_timestamps, ensure_ascii=False),
        pauses=json.dumps(feature.pause_timestamps, ensure_ascii=False),
    )


def _validate_result(result: Any, duration_seconds: float) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("LLM JSON must be an object.")
    scores = result.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(CATEGORIES):
        raise ValueError("LLM JSON must contain scores for exactly the four rubric categories.")
    for category, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
            raise ValueError(f"Invalid score for {category}: expected a number from 0 to 10.")

    flaws = result.get("flaws")
    if not isinstance(flaws, list):
        raise ValueError("LLM JSON must contain a flaws array.")
    counts = {category: 0 for category in CATEGORIES}
    for flaw in flaws:
        if not isinstance(flaw, dict) or flaw.get("category") not in CATEGORIES:
            raise ValueError("Each flaw must name one of the four rubric categories.")
        timestamp = flaw.get("timestamp_sec")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Each flaw must have a non-negative numeric timestamp_sec.")
        if timestamp > duration_seconds:
            raise ValueError("A flaw timestamp falls outside the speech duration.")
        counts[flaw["category"]] += 1
        if not isinstance(flaw.get("flaw_type"), str) or not flaw["flaw_type"].strip():
            raise ValueError("Each flaw must have a flaw_type.")
        if not isinstance(flaw.get("note"), str) or not flaw["note"].strip():
            raise ValueError("Each flaw must have a note.")
    if any(count < 1 or count > 3 for count in counts.values()):
        raise ValueError("Each category must have between one and three specific flaws.")
    if not isinstance(result.get("overall_summary"), str):
        raise ValueError("LLM JSON must contain an overall_summary string.")
    return result


def score_speech(speech_id: int, db_session: Session) -> dict[str, Any]:
    """Score a persisted speech and save the response, four scores, and flaws."""
    speech = db_session.get(Speech, speech_id)
    if speech is None:
        raise LookupError(f"Speech {speech_id} was not found.")
    feature = db_session.scalar(
        select(Feature).where(Feature.speech_id == speech_id).order_by(Feature.id.desc())
    )
    if feature is None:
        raise LookupError(f"Features for speech {speech_id} were not found.")

    raw_response = call_llm(SYSTEM_PROMPT, _user_prompt(speech, feature))
    # Preserve the model output even if parsing or validation below fails.
    db_session.add(EvalRun(speech_id=speech_id, prompt_version=RUBRIC_VERSION, raw_llm_response=raw_response))
    db_session.commit()

    try:
        parsed = _validate_result(json.loads(raw_response), speech.duration_seconds)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.exception("Could not parse/validate LLM response for speech %s. Raw response: %s", speech_id, raw_response)
        raise ValueError(f"LLM returned invalid rubric JSON: {exc}") from exc

    for category in CATEGORIES:
        db_session.add(
            RubricScore(
                speech_id=speech_id,
                category=category,
                score=float(parsed["scores"][category]),
                rubric_version=RUBRIC_VERSION,
            )
        )
    for flaw in parsed["flaws"]:
        db_session.add(
            Flaw(
                speech_id=speech_id,
                category=flaw["category"],
                timestamp_sec=float(flaw["timestamp_sec"]),
                flaw_type=flaw["flaw_type"],
                note=flaw["note"],
            )
        )
    db_session.commit()
    return parsed
