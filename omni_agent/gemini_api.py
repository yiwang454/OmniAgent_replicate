"""Gemini legacy HTTP client matching react-agent-avqa-optimize."""

from __future__ import annotations

import base64
import mimetypes
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from omni_agent.config import (
    GEMINI_BASE_URL,
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    GEMINI_RETRY_DELAY_S,
    GEMINI_TIMEOUT,
    YOUR_API_KEY_GEMINI,
)


def _inline_media_part(media_path: str, *, fps: float | None = None) -> dict[str, Any]:
    path = Path(media_path)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    part: dict[str, Any] = {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
        }
    }
    if fps is not None:
        part["videoMetadata"] = {"fps": fps}
    return part


def _response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini response has no candidates: {payload}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and not part.get("thought")
    ).strip()
    if not text:
        finish_reason = candidates[0].get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned no answer text (finishReason={finish_reason})")
    return text


def call_gemini_with_media(
    media_path: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    fps: float | None = None,
) -> str:
    """Send inline audio/video to the configured Gemini 2.5 Flash endpoint."""
    if not YOUR_API_KEY_GEMINI:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to OmniAgent/.env or export it "
            "before starting OmniAgent."
        )
    if not GEMINI_BASE_URL:
        raise RuntimeError("GEMINI_BASE_URL is not set.")

    model_path = urllib.parse.quote(GEMINI_MODEL, safe="")
    url = f"{GEMINI_BASE_URL}/v1beta/models/{model_path}:generateContent"
    headers = {
        "Authorization": f"Bearer {YOUR_API_KEY_GEMINI}",
        "Content-Type": "application/json",
    }
    data: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    _inline_media_part(media_path, fps=fps),
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.6,
            "topP": 0.95,
            "topK": 20,
            "maxOutputTokens": GEMINI_MAX_TOKENS,
        },
    }
    if system_prompt:
        data["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    last_error: Exception | None = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=GEMINI_TIMEOUT,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Gemini API error {response.status_code}: {response.text[:2000]}"
                )
            return _response_text(response.json())
        except Exception as exc:
            last_error = exc
            if attempt < GEMINI_MAX_RETRIES:
                time.sleep(GEMINI_RETRY_DELAY_S * (2 ** (attempt - 1)))

    raise RuntimeError(f"Gemini API call failed: {last_error}") from last_error
