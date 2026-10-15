"""Per-question evaluation, usage, cost, cache, and failure summaries."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-F])\s*</answer>", re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(r"^\s*([A-F])(?:[\).\]:\s]|$)", re.IGNORECASE)

# Public Standard/Paid Tier rates used by the existing repository report.
USD_PER_MILLION = {
    "o3_uncached_input": 2.00,
    "o3_cached_input": 0.50,
    "o3_output_including_reasoning": 8.00,
    "gemini_2_5_flash_text_image_video_input": 0.30,
    "gemini_2_5_flash_audio_input": 1.00,
    "gemini_2_5_flash_output_including_thinking": 2.50,
}


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().upper()
    if match := ANSWER_TAG_RE.search(text):
        return match.group(1).upper()
    if len(text) == 1 and text in "ABCDEF":
        return text
    if match := ANSWER_PREFIX_RE.search(text):
        return match.group(1).upper()
    return ""


def _number(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def planner_cost(usage: Mapping[str, Any]) -> float:
    total_input = _number(usage, "input_tokens")
    cached_input = min(total_input, _number(usage, "cached_input_tokens"))
    uncached_input = total_input - cached_input
    output = _number(usage, "output_tokens")
    return (
        uncached_input * USD_PER_MILLION["o3_uncached_input"]
        + cached_input * USD_PER_MILLION["o3_cached_input"]
        + output * USD_PER_MILLION["o3_output_including_reasoning"]
    ) / 1_000_000


def gemini_cost(usage: Mapping[str, Any] | None) -> float | None:
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("promptTokensDetails") or usage.get("prompt_tokens_by_modality") or {}
    modality: dict[str, int] = {}
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes, bytearray)):
        for item in details:
            if isinstance(item, Mapping):
                key = str(item.get("modality") or "UNKNOWN").upper()
                modality[key] = modality.get(key, 0) + _number(item, "tokenCount")
    elif isinstance(details, Mapping):
        modality = {str(key).upper(): int(value) for key, value in details.items()}

    prompt_total = _number(usage, "promptTokenCount") or _number(usage, "prompt_tokens")
    audio = modality.get("AUDIO", 0)
    non_audio_known = sum(value for key, value in modality.items() if key != "AUDIO")
    unclassified = max(0, prompt_total - audio - non_audio_known)
    text_image_video = non_audio_known + unclassified
    candidates = _number(usage, "candidatesTokenCount") or _number(usage, "candidate_tokens")
    thoughts = _number(usage, "thoughtsTokenCount") or _number(usage, "thoughts_tokens")
    return (
        audio * USD_PER_MILLION["gemini_2_5_flash_audio_input"]
        + text_image_video
        * USD_PER_MILLION["gemini_2_5_flash_text_image_video_input"]
        + (candidates + thoughts)
        * USD_PER_MILLION["gemini_2_5_flash_output_including_thinking"]
    ) / 1_000_000


def _cache_calls(turn_trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        turn
        for turn in turn_trace
        if turn.get("caption_cache_hit") is True
    ]


def build_question_metrics(question_data: Mapping[str, Any]) -> dict[str, Any]:
    trace_value = question_data.get("turn_trace") or []
    trace = [turn for turn in trace_value if isinstance(turn, Mapping)]
    tool_turns = [turn for turn in trace if turn.get("planner_action") == "tool"]
    by_tool = Counter(str(turn.get("tool_name") or "<unknown>") for turn in tool_turns)
    response = str(question_data.get("response") or "")
    gold = normalize_answer(question_data.get("answer"))
    predicted = normalize_answer(response)

    failures: list[dict[str, Any]] = []
    if response.startswith("[ERROR]"):
        failures.append({"scope": "run", "error": response})
    for turn in trace:
        if turn.get("tool_error"):
            failures.append(
                {
                    "scope": "tool",
                    "turn_id": turn.get("turn_id"),
                    "tool_name": turn.get("tool_name"),
                    "error": turn.get("tool_error"),
                }
            )
        for call in turn.get("perception_calls") or []:
            if isinstance(call, Mapping) and call.get("error"):
                failures.append(
                    {
                        "scope": "perception",
                        "turn_id": turn.get("turn_id"),
                        "tool_name": turn.get("tool_name"),
                        "error": call.get("error"),
                    }
                )

    token_usage = question_data.get("token_usage") or {}
    planner_usage = token_usage.get("planner") or {}
    perception_usage = token_usage.get("perception") or {}
    planner_usd = planner_cost(planner_usage)
    perception_usd = gemini_cost(perception_usage) or 0.0

    cache_calls = _cache_calls(trace)
    unique_cache_calls: dict[str, Mapping[str, Any]] = {}
    for call in cache_calls:
        key = str(call.get("caption_cache_path") or f"turn:{call.get('turn_id')}")
        unique_cache_calls.setdefault(key, call)
    cache_costs: list[float] = []
    cache_latencies: list[float] = []
    unknown_cost = 0
    unknown_latency = 0
    for call in unique_cache_calls.values():
        cost = gemini_cost(call.get("caption_cache_source_token_usage"))
        if cost is None:
            unknown_cost += 1
        else:
            cache_costs.append(cost)
        latency = call.get("caption_cache_source_latency_seconds")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            cache_latencies.append(float(latency))
        else:
            unknown_latency += 1

    live_cost = planner_usd + perception_usd
    original_cache_cost = sum(cache_costs)
    warm_lookup = sum(
        float(call.get("caption_cache_lookup_seconds") or 0.0) for call in cache_calls
    )
    latency = question_data.get("latency") or {}
    warm_wall = latency.get("wall_clock_seconds")
    source_latency = sum(cache_latencies)
    has_cache_entry = bool(unique_cache_calls)
    cold_complete = has_cache_entry and unknown_latency == 0
    cold_end_to_end = (
        float(warm_wall) + source_latency
        if cold_complete and isinstance(warm_wall, (int, float))
        else None
    )

    return {
        "evaluation": {
            "gold_answer": gold,
            "predicted_answer": predicted,
            "parseable": bool(predicted),
            "correct": bool(gold and predicted and gold == predicted),
        },
        "tool_calls": {"total": len(tool_turns), "by_tool": dict(sorted(by_tool.items()))},
        "tokens": token_usage,
        "estimated_cost_usd": {
            "rates_per_million_tokens": USD_PER_MILLION,
            "planner_live": round(planner_usd, 8),
            "perception_live": round(perception_usd, 8),
            "live_marginal_total": round(live_cost, 8),
            "caption_cache_original_generation_known": round(original_cache_cost, 8),
            "caption_cache_original_generation_unknown_entries": unknown_cost,
            "cold_cache_total": (
                round(live_cost + original_cache_cost, 8)
                if has_cache_entry and unknown_cost == 0
                else None
            ),
            "cold_cache_known_component": round(live_cost + original_cache_cost, 8),
        },
        "cache_latency": {
            "cache_hits": len(cache_calls),
            "unique_cache_entries": len(unique_cache_calls),
            "warm_cache_lookup_seconds": round(warm_lookup, 6),
            "warm_run_wall_clock_seconds": warm_wall,
            "cold_cache_source_generation_seconds_known": round(source_latency, 3),
            "cold_cache_source_latency_unknown_entries": unknown_latency,
            "cold_cache_end_to_end_estimate_seconds": (
                round(cold_end_to_end, 3) if cold_end_to_end is not None else None
            ),
        },
        "failure": {
            "status": "failed" if failures else "success",
            "details": failures,
        },
    }
