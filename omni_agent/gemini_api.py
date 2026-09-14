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
    GEMINI_RETRY_MAX_DELAY_S,
    GEMINI_TIMEOUT,
    YOUR_API_KEY_GEMINI,
)
from omni_agent.tracing import active_trace_recorder


PART_ONEOF_ERROR = "required oneof field 'data' must have one initialized field"


class GeminiAPIError(RuntimeError):
    """HTTP error returned by the Gemini-compatible gateway."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Gemini API error {status_code}: {body[:2000]}")


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


def _request_payload(
    media_part: dict[str, Any],
    prompt: str,
    *,
    system_prompt: str | None,
    text_first: bool,
) -> dict[str, Any]:
    text_part = {"text": prompt}
    parts = [text_part, media_part] if text_first else [media_part, text_part]
    data: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.6,
            "topP": 0.95,
            "topK": 20,
            "maxOutputTokens": GEMINI_MAX_TOKENS,
        },
    }
    if system_prompt:
        data["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    return data


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, GeminiAPIError):
        return (
            error.status_code == 429
            or error.status_code >= 500
            or (error.status_code == 400 and PART_ONEOF_ERROR in error.body)
        )
    return isinstance(error, requests.RequestException)


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

    recorder = active_trace_recorder()
    trace_call = None
    if recorder is not None:
        trace_call = recorder.start_perception_call(
            media_path=media_path,
            prompt=prompt,
            system_prompt=system_prompt,
            model=GEMINI_MODEL,
            fps=fps,
        )

    model_path = urllib.parse.quote(GEMINI_MODEL, safe="")
    url = f"{GEMINI_BASE_URL}/v1beta/models/{model_path}:generateContent"
    headers = {
        "Authorization": f"Bearer {YOUR_API_KEY_GEMINI}",
        "Content-Type": "application/json",
    }
    media_part = _inline_media_part(media_path, fps=fps)
    text_first = False

    last_error: Exception | None = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        data = _request_payload(
            media_part,
            prompt,
            system_prompt=system_prompt,
            text_first=text_first,
        )
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=GEMINI_TIMEOUT,
            )
            if not response.ok:
                raise GeminiAPIError(response.status_code, response.text)
            answer = _response_text(response.json())
            if trace_call is not None:
                trace_call["response"] = answer
                trace_call["attempts"] = attempt
            return answer
        except Exception as exc:
            last_error = exc
            if isinstance(exc, GeminiAPIError) and PART_ONEOF_ERROR in exc.body:
                # Some instances behind the compatible gateway intermittently
                # discard a text Part when it follows inlineData. Both orders
                # are valid Gemini REST, so retry using text first.
                text_first = True
            if attempt >= GEMINI_MAX_RETRIES or not _is_retryable(exc):
                break
            delay = min(
                GEMINI_RETRY_DELAY_S * (2 ** (attempt - 1)),
                GEMINI_RETRY_MAX_DELAY_S,
            )
            print(
                f"[warn] Gemini attempt {attempt}/{GEMINI_MAX_RETRIES} failed: "
                f"{exc}. Retrying in {delay:g}s.",
                flush=True,
            )
            time.sleep(delay)

    if trace_call is not None:
        trace_call["error"] = f"{type(last_error).__name__}: {last_error}"
        trace_call["attempts"] = attempt
    raise RuntimeError(f"Gemini API call failed: {last_error}") from last_error
