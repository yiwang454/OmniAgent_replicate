"""Structured prompt/response tracing for OmniAgent rollouts."""

from __future__ import annotations

import contextlib
import contextvars
import threading
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


def json_safe(value: Any) -> Any:
    """Convert LangChain/SDK values into JSON-serializable Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return json_safe(method())
            except Exception:
                pass
    return str(value)


def _message_dict(message: Any) -> dict[str, Any]:
    """Keep the complete, readable parts of one LangChain message."""
    serialized = json_safe(message)
    if isinstance(serialized, dict):
        serialized["role"] = serialized.get(
            "type", getattr(message, "type", type(message).__name__)
        )
        return serialized
    return {
        "role": getattr(message, "type", type(message).__name__),
        "content": json_safe(getattr(message, "content", serialized)),
    }


def _number(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def _planner_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    generations = response.get("generations") or []
    first = generations[0] if generations else {}
    if isinstance(first, dict):
        usage = first.get("usage_metadata")
        if isinstance(usage, dict):
            return usage
        metadata = first.get("response_metadata") or {}
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
            if isinstance(usage, dict):
                return usage
    llm_output = response.get("llm_output") or {}
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(usage, dict):
            return usage
    return None


def summarize_token_usage(recorder: "RolloutTraceRecorder") -> dict[str, Any]:
    """Aggregate exact successful planner and Gemini usage for one sample."""
    planner = {
        "calls": len(recorder.planner_calls),
        "calls_with_usage": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    for call in recorder.planner_calls:
        usage = call.get("usage_metadata")
        if not isinstance(usage, dict):
            continue
        planner["calls_with_usage"] += 1
        planner["input_tokens"] += _number(usage, "input_tokens", "prompt_tokens")
        planner["output_tokens"] += _number(
            usage, "output_tokens", "completion_tokens"
        )
        planner["total_tokens"] += _number(usage, "total_tokens")
        input_details = usage.get("input_token_details") or usage.get(
            "prompt_tokens_details"
        ) or {}
        output_details = usage.get("output_token_details") or usage.get(
            "completion_tokens_details"
        ) or {}
        if isinstance(input_details, dict):
            planner["cached_input_tokens"] += _number(
                input_details, "cache_read", "cached_tokens"
            )
        if isinstance(output_details, dict):
            planner["reasoning_output_tokens"] += _number(
                output_details, "reasoning", "reasoning_tokens"
            )

    perception = {
        "calls": len(recorder.perception_calls),
        "calls_with_usage": 0,
        "prompt_tokens": 0,
        "cached_content_tokens": 0,
        "candidate_tokens": 0,
        "thoughts_tokens": 0,
        "tool_use_prompt_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_by_modality": {},
        "candidate_tokens_by_modality": {},
    }
    for call in recorder.perception_calls:
        usage = call.get("usage_metadata")
        if not isinstance(usage, dict):
            continue
        perception["calls_with_usage"] += 1
        perception["prompt_tokens"] += _number(usage, "promptTokenCount")
        perception["cached_content_tokens"] += _number(
            usage, "cachedContentTokenCount"
        )
        perception["candidate_tokens"] += _number(usage, "candidatesTokenCount")
        perception["thoughts_tokens"] += _number(usage, "thoughtsTokenCount")
        perception["tool_use_prompt_tokens"] += _number(
            usage, "toolUsePromptTokenCount"
        )
        perception["total_tokens"] += _number(usage, "totalTokenCount")
        for detail_key, summary_key in (
            ("promptTokensDetails", "prompt_tokens_by_modality"),
            ("candidatesTokensDetails", "candidate_tokens_by_modality"),
        ):
            details = usage.get(detail_key) or []
            if not isinstance(details, list):
                continue
            target = perception[summary_key]
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                modality = str(detail.get("modality") or "UNKNOWN")
                target[modality] = target.get(modality, 0) + _number(
                    detail, "tokenCount"
                )
    return {"planner": planner, "perception": perception}


class RolloutTraceRecorder(BaseCallbackHandler):
    """Capture every central planner call and raw Gemini perception call."""

    def __init__(self) -> None:
        self.planner_calls: list[dict[str, Any]] = []
        self.perception_calls: list[dict[str, Any]] = []
        self.caption_cache_calls: list[dict[str, Any]] = []
        self._planner_by_run_id: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self._lock = threading.Lock()

    def _next_sequence(self) -> int:
        with self._lock:
            self._sequence += 1
            return self._sequence

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = kwargs.get("metadata") or {}
        invocation_params = kwargs.get("invocation_params") or {}
        call = {
            "sequence": self._next_sequence(),
            "model": metadata.get("ls_model_name")
            or invocation_params.get("model")
            or invocation_params.get("model_name")
            or (serialized or {}).get("kwargs", {}).get("model_name")
            or (serialized or {}).get("name"),
            "messages": [
                [_message_dict(message) for message in prompt_messages]
                for prompt_messages in messages
            ],
            "response_text": None,
            "response": None,
            "error": None,
        }
        self.planner_calls.append(call)
        self._planner_by_run_id[str(run_id)] = call

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        call = self._planner_by_run_id.get(str(run_id))
        if call is None:
            return

        generations: list[dict[str, Any]] = []
        response_texts: list[str] = []
        for generation_group in getattr(response, "generations", []) or []:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                if message is not None:
                    generation_data = _message_dict(message)
                    text = generation_data.get("content")
                else:
                    text = getattr(generation, "text", "")
                    generation_data = {"content": json_safe(text)}
                generations.append(generation_data)
                if isinstance(text, str):
                    response_texts.append(text)

        call["response_text"] = "\n".join(response_texts)
        call["response"] = {
            "generations": generations,
            "llm_output": json_safe(getattr(response, "llm_output", None)),
        }
        call["usage_metadata"] = _planner_usage(call["response"])

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        call = self._planner_by_run_id.get(str(run_id))
        if call is not None:
            call["error"] = f"{type(error).__name__}: {error}"

    def start_perception_call(
        self,
        *,
        media_path: str,
        prompt: str,
        system_prompt: str | None,
        model: str,
        fps: float | None,
    ) -> dict[str, Any]:
        call = {
            "sequence": self._next_sequence(),
            "model": model,
            "media_path": media_path,
            "fps": fps,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "response": None,
            "usage_metadata": None,
            "error": None,
        }
        self.perception_calls.append(call)
        return call

    def record_caption_cache_call(
        self,
        *,
        media_path: str,
        prompt: str,
        response: str,
        lookup_seconds: float,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Record a cache observation alongside live perception calls."""
        call = {
            "sequence": self._next_sequence(),
            "perception_backend": "caption_cache",
            "media_path": media_path,
            "prompt": prompt,
            "response": response,
            "caption_cache_lookup_seconds": round(lookup_seconds, 6),
            **json_safe(metadata),
        }
        self.caption_cache_calls.append(call)
        return call


_ACTIVE_RECORDER: contextvars.ContextVar[RolloutTraceRecorder | None] = (
    contextvars.ContextVar("omniagent_rollout_trace_recorder", default=None)
)


@contextlib.contextmanager
def activate_trace_recorder(recorder: RolloutTraceRecorder) -> Iterator[None]:
    """Make ``recorder`` available to raw perception API helpers."""
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_RECORDER.reset(token)


def active_trace_recorder() -> RolloutTraceRecorder | None:
    return _ACTIVE_RECORDER.get()
