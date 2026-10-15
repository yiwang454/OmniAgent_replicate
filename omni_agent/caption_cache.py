"""Question-scoped audio_global_caption cache integration.

The cache schema comes from the sibling react-agent-avqa-optimize repository's
OmniAgent cache builder. Cached generation usage is retained as provenance and
is deliberately not counted as live-run usage.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from omni_agent.tracing import active_trace_recorder


AUDIO_GLOBAL_CAPTION_PROMPT = (
    "Provide a high-level summary of the audio. "
    "Focus on the main topics, key events, and the overall atmosphere, "
)


@dataclass(frozen=True)
class CaptionCacheContext:
    cache_dir: Path
    question_id: str


@dataclass(frozen=True)
class CachedAudioCaption:
    response: str
    path: Path
    metadata: dict[str, Any]
    lookup_seconds: float


_ACTIVE_CACHE: contextvars.ContextVar[CaptionCacheContext | None] = (
    contextvars.ContextVar("omniagent_caption_cache", default=None)
)


def _safe_entry_path(cache_dir: Path, question_id: str) -> Path:
    normalized = str(question_id or "").strip()
    if not normalized or Path(normalized).name != normalized:
        raise ValueError(f"Unsafe caption-cache question ID: {question_id!r}")
    return cache_dir / f"{normalized}.json"


def _source_latency(payload: Mapping[str, Any]) -> float | None:
    direct = payload.get("source_latency_seconds")
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return float(direct)
    source_path = Path(str(payload.get("source_results_file") or ""))
    if not source_path.is_file() or source_path.suffix.lower() != ".json":
        return None
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = source.get("total_wall_seconds") if isinstance(source, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def load_cached_audio_caption(
    cache_dir: Path, question_id: str
) -> CachedAudioCaption:
    started = time.perf_counter()
    root = cache_dir.expanduser().resolve()
    path = _safe_entry_path(root, question_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Caption cache entry is not an object: {path}")
    if str(payload.get("question_id") or "") != question_id:
        raise ValueError(f"Caption cache question ID mismatch: {path}")
    if str(payload.get("caption_prompt") or "") != AUDIO_GLOBAL_CAPTION_PROMPT:
        raise ValueError(f"Caption cache prompt mismatch: {path}")
    response = str(payload.get("response") or "").strip()
    if not response:
        raise ValueError(f"Caption cache response is empty: {path}")

    metadata = {
        "caption_cache_hit": True,
        "caption_cache_path": str(path),
        "caption_cache_source_results": payload.get("source_results_file"),
        "caption_cache_source_rank": payload.get("source_rank"),
        "caption_cache_source_kind": payload.get("source_kind"),
        "caption_cache_source_backend": payload.get("source_backend"),
        "caption_cache_source_model": payload.get("source_model"),
        "caption_cache_source_token_usage": payload.get("source_token_usage"),
        "caption_cache_source_latency_seconds": _source_latency(payload),
    }
    return CachedAudioCaption(
        response=response,
        path=path,
        metadata=metadata,
        lookup_seconds=time.perf_counter() - started,
    )


@contextlib.contextmanager
def activate_caption_cache(
    cache_dir: Path | None, question_id: str | None
) -> Iterator[None]:
    context = None
    if cache_dir is not None:
        if not question_id:
            raise ValueError("Caption cache requires a question ID")
        context = CaptionCacheContext(cache_dir.resolve(), str(question_id))
    token = _ACTIVE_CACHE.set(context)
    try:
        yield
    finally:
        _ACTIVE_CACHE.reset(token)


def active_cached_audio_caption(video_path: str) -> CachedAudioCaption | None:
    context = _ACTIVE_CACHE.get()
    if context is None:
        return None
    cached = load_cached_audio_caption(context.cache_dir, context.question_id)
    recorder = active_trace_recorder()
    if recorder is not None:
        recorder.record_caption_cache_call(
            media_path=video_path,
            prompt=AUDIO_GLOBAL_CAPTION_PROMPT,
            response=cached.response,
            lookup_seconds=cached.lookup_seconds,
            metadata=cached.metadata,
        )
    return cached
