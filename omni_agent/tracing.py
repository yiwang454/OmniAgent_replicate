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


class RolloutTraceRecorder(BaseCallbackHandler):
    """Capture every central planner call and raw Gemini perception call."""

    def __init__(self) -> None:
        self.planner_calls: list[dict[str, Any]] = []
        self.perception_calls: list[dict[str, Any]] = []
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
            "error": None,
        }
        self.perception_calls.append(call)
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
