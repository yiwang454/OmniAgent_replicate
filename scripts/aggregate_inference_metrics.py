#!/usr/bin/env python3
"""Validate and aggregate per-question latency and model token usage."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omni_agent.run_metrics import build_question_metrics


DURATION_KEYS = (
    "wall_clock_seconds",
    "retry_adjusted_seconds",
    "successful_model_call_seconds",
)
CALL_COUNT_KEYS = (
    "planner_successful_calls",
    "perception_successful_calls",
)
PLANNER_TOKEN_KEYS = (
    "calls",
    "calls_with_usage",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
PERCEPTION_TOKEN_KEYS = (
    "calls",
    "calls_with_usage",
    "prompt_tokens",
    "cached_content_tokens",
    "candidate_tokens",
    "thoughts_tokens",
    "tool_use_prompt_tokens",
    "total_tokens",
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summary(values: list[float | int], *, total_rows: int) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "null_count": total_rows,
            "sum": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(values),
        "null_count": total_rows - len(values),
        "sum": round(sum(numeric), 3),
        "mean": round(statistics.fmean(numeric), 3),
        "median": round(statistics.median(numeric), 3),
        "p90": round(_percentile(numeric, 0.90), 3),
        "p95": round(_percentile(numeric, 0.95), 3),
        "min": round(min(numeric), 3),
        "max": round(max(numeric), 3),
    }


def _nonnegative_number(value: Any, *, field: str, sample_id: str) -> float | int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid {field} for {sample_id}: {value!r}")
    return value


def aggregate_metrics(input_jsonl: Path, *, expected_count: int = 0) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if expected_count < 0:
        raise ValueError("expected_count must be nonnegative")
    if expected_count and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")

    sample_ids = [str(row.get("video_id") or "") for row in rows]
    if not all(sample_ids):
        raise ValueError("At least one output row has an empty video_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Output contains duplicate video_id values")

    latency_values: dict[str, list[float | int]] = {
        key: [] for key in (*DURATION_KEYS, *CALL_COUNT_KEYS)
    }
    token_values: dict[str, dict[str, list[float | int]]] = {
        "planner": {key: [] for key in PLANNER_TOKEN_KEYS},
        "perception": {key: [] for key in PERCEPTION_TOKEN_KEYS},
    }
    modality_totals: dict[str, dict[str, int]] = {
        "prompt_tokens_by_modality": {},
        "candidate_tokens_by_modality": {},
    }
    evaluation = {
        "with_gold": 0,
        "parseable": 0,
        "correct": 0,
    }
    tool_calls_total: list[int] = []
    tool_calls_by_name: Counter[str] = Counter()
    cost_totals: dict[str, float] = Counter()
    cost_null_counts: Counter[str] = Counter()
    cache_latency_values: dict[str, list[float | int]] = {
        "warm_cache_lookup_seconds": [],
        "warm_run_wall_clock_seconds": [],
        "cold_cache_source_generation_seconds_known": [],
        "cold_cache_end_to_end_estimate_seconds": [],
        "cache_hits": [],
    }
    cache_unknown_cost_entries = 0
    cache_unknown_latency_entries = 0
    failure_records: list[dict[str, Any]] = []

    for sample_id, row in zip(sample_ids, rows):
        question_data = row.get("question_data")
        if not isinstance(question_data, dict):
            raise ValueError(f"Missing question_data for {sample_id}")

        latency = question_data.get("latency")
        if not isinstance(latency, dict):
            raise ValueError(f"Missing latency for {sample_id}")
        for key in DURATION_KEYS:
            value = latency.get(key)
            if value is None:
                continue
            latency_values[key].append(
                _nonnegative_number(value, field=key, sample_id=sample_id)
            )
        for key in CALL_COUNT_KEYS:
            value = _nonnegative_number(
                latency.get(key), field=key, sample_id=sample_id
            )
            if not isinstance(value, int):
                raise ValueError(f"Invalid integer {key} for {sample_id}: {value!r}")
            latency_values[key].append(value)

        usage = question_data.get("token_usage")
        if not isinstance(usage, dict):
            raise ValueError(f"Missing token_usage for {sample_id}")
        for component, keys in (
            ("planner", PLANNER_TOKEN_KEYS),
            ("perception", PERCEPTION_TOKEN_KEYS),
        ):
            component_usage = usage.get(component)
            if not isinstance(component_usage, dict):
                raise ValueError(f"Missing {component} token usage for {sample_id}")
            for key in keys:
                value = _nonnegative_number(
                    component_usage.get(key),
                    field=f"{component}.{key}",
                    sample_id=sample_id,
                )
                if not isinstance(value, int):
                    raise ValueError(
                        f"Invalid integer {component}.{key} for {sample_id}: {value!r}"
                    )
                token_values[component][key].append(value)

        perception_usage = usage["perception"]
        for key in modality_totals:
            values = perception_usage.get(key) or {}
            if not isinstance(values, dict):
                raise ValueError(f"Invalid perception.{key} for {sample_id}")
            for modality, value in values.items():
                numeric = _nonnegative_number(
                    value,
                    field=f"perception.{key}.{modality}",
                    sample_id=sample_id,
                )
                if not isinstance(numeric, int):
                    raise ValueError(
                        f"Invalid integer perception.{key}.{modality} for {sample_id}"
                    )
                modality_totals[key][str(modality)] = (
                    modality_totals[key].get(str(modality), 0) + numeric
                )

        metrics = question_data.get("metrics")
        if not isinstance(metrics, dict):
            metrics = build_question_metrics(question_data)
        item_evaluation = metrics.get("evaluation") or {}
        evaluation["with_gold"] += int(bool(item_evaluation.get("gold_answer")))
        evaluation["parseable"] += int(bool(item_evaluation.get("parseable")))
        evaluation["correct"] += int(bool(item_evaluation.get("correct")))

        item_tool_calls = metrics.get("tool_calls") or {}
        total_tool_calls = _nonnegative_number(
            item_tool_calls.get("total"), field="metrics.tool_calls.total", sample_id=sample_id
        )
        if not isinstance(total_tool_calls, int):
            raise ValueError(f"Invalid tool call count for {sample_id}")
        tool_calls_total.append(total_tool_calls)
        for name, value in (item_tool_calls.get("by_tool") or {}).items():
            numeric = _nonnegative_number(
                value, field=f"metrics.tool_calls.by_tool.{name}", sample_id=sample_id
            )
            tool_calls_by_name[str(name)] += int(numeric)

        item_cost = metrics.get("estimated_cost_usd") or {}
        for key in (
            "planner_live",
            "perception_live",
            "live_marginal_total",
            "caption_cache_original_generation_known",
            "cold_cache_total",
            "cold_cache_known_component",
        ):
            value = item_cost.get(key)
            if value is None:
                cost_null_counts[key] += 1
                continue
            cost_totals[key] += float(
                _nonnegative_number(value, field=f"metrics.cost.{key}", sample_id=sample_id)
            )
        cache_unknown_cost_entries += int(
            _nonnegative_number(
                item_cost.get("caption_cache_original_generation_unknown_entries", 0),
                field="metrics.cost.caption_cache_original_generation_unknown_entries",
                sample_id=sample_id,
            )
        )

        item_cache = metrics.get("cache_latency") or {}
        for key in cache_latency_values:
            value = item_cache.get(key)
            if value is None:
                continue
            cache_latency_values[key].append(
                _nonnegative_number(
                    value, field=f"metrics.cache_latency.{key}", sample_id=sample_id
                )
            )
        cache_unknown_latency_entries += int(
            _nonnegative_number(
                item_cache.get("cold_cache_source_latency_unknown_entries", 0),
                field="metrics.cache_latency.cold_cache_source_latency_unknown_entries",
                sample_id=sample_id,
            )
        )

        failure = metrics.get("failure") or {}
        if failure.get("status") != "success":
            failure_records.append(
                {"question_id": sample_id, "details": failure.get("details") or []}
            )

    total_rows = len(rows)
    accuracy_denominator = evaluation["with_gold"]
    return {
        "input_jsonl": str(input_jsonl.resolve()),
        "row_count": total_rows,
        "unique_id_count": len(set(sample_ids)),
        "latency_seconds": {
            key: _summary(latency_values[key], total_rows=total_rows)
            for key in DURATION_KEYS
        },
        "successful_call_counts": {
            key: _summary(latency_values[key], total_rows=total_rows)
            for key in CALL_COUNT_KEYS
        },
        "token_usage": {
            component: {
                key: _summary(values, total_rows=total_rows)
                for key, values in component_values.items()
            }
            for component, component_values in token_values.items()
        },
        "perception_token_modality_totals": modality_totals,
        "accuracy": {
            **evaluation,
            "percent": (
                round(100.0 * evaluation["correct"] / accuracy_denominator, 4)
                if accuracy_denominator
                else None
            ),
        },
        "tool_calls": {
            "total": sum(tool_calls_total),
            "per_question": _summary(tool_calls_total, total_rows=total_rows),
            "by_tool": dict(sorted(tool_calls_by_name.items())),
        },
        "estimated_cost_usd": {
            "totals": {key: round(value, 8) for key, value in sorted(cost_totals.items())},
            "null_question_counts": dict(sorted(cost_null_counts.items())),
            "caption_cache_original_generation_unknown_entries": cache_unknown_cost_entries,
        },
        "cache_latency": {
            key: _summary(values, total_rows=total_rows)
            for key, values in cache_latency_values.items()
        }
        | {
            "cold_cache_source_latency_unknown_entries": cache_unknown_latency_entries
        },
        "failures": {
            "count": len(failure_records),
            "records": failure_records,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--expected-count", type=int, default=0)
    args = parser.parse_args()
    output_json = args.output_json or args.input_jsonl.with_name(
        "inference_metrics_summary.json"
    )
    try:
        summary = aggregate_metrics(
            args.input_jsonl, expected_count=args.expected_count
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Inference metrics aggregation failed: {exc}") from exc
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote inference metrics summary: {output_json}")


if __name__ == "__main__":
    main()
