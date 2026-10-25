"""Groq client with a local Ollama fallback."""

import json
import logging
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
OLLAMA_MODEL = "nemotron-3-nano:4b"
OLLAMA_BASE_URL = "http://localhost:11434"
GROQ_TIMEOUT_SECONDS = 45.0
OLLAMA_TIMEOUT_SECONDS = 180.0

logger = logging.getLogger(__name__)

OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                category: {"type": "number", "minimum": 0, "maximum": 10}
                for category in ("delivery", "structure", "content", "engagement")
            },
            "required": ["delivery", "structure", "content", "engagement"],
            "additionalProperties": False,
        },
        "flaws": {
            "type": "array",
            "minItems": 4,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["delivery", "structure", "content", "engagement"],
                    },
                    "timestamp_sec": {"type": "number", "minimum": 0},
                    "flaw_type": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["category", "timestamp_sec", "flaw_type", "note"],
                "additionalProperties": False,
            },
        },
        "overall_summary": {"type": "string"},
    },
    "required": ["scores", "flaws", "overall_summary"],
    "additionalProperties": False,
}


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Call Groq, retrying once after request failures."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS, max_retries=0)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Groq returned an empty response.")
            return content
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.5)

    raise RuntimeError(f"Groq request failed after 2 attempts: {last_error}") from last_error


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call the local Ollama chat endpoint and return its raw message content."""
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": OLLAMA_RESPONSE_SCHEMA,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Local Ollama request failed: {exc}") from exc

    content = body.get("message", {}).get("content") if isinstance(body, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Local Ollama returned an empty or malformed response.")
    return content


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Try Groq first, then fall back to the configured local Ollama model."""
    try:
        return _call_groq(system_prompt, user_prompt)
    except Exception as groq_error:
        logger.warning("Groq call failed, falling back to local Ollama: %s", groq_error)
        try:
            return _call_ollama(system_prompt, user_prompt)
        except Exception as ollama_error:
            raise RuntimeError(
                f"Groq failed and local Ollama fallback failed: {ollama_error}"
            ) from ollama_error
