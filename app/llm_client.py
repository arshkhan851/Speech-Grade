"""Small, swappable wrapper around the Groq chat completions API."""

import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "llama-3.1-8b-instant"
REQUEST_TIMEOUT_SECONDS = 45.0


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Return the model's raw response, retrying once after any request failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file or export it in the environment."
        )

    client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
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
