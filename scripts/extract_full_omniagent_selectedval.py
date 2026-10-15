#!/usr/bin/env python3
"""Extract a manifest-defined subset from legacy full OmniAgent runs.

The legacy traces contain exact o3 planner usage, but no Gemini usageMetadata.
Gemini tokens and cost are therefore estimated with the same duration/character
method used for the Full OmniAgent row in react-agent-avqa-optimize/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ANSWER_TAG = re.compile(r"<answer>\s*([A-F])\s*</answer>", re.I)
ANSWER_PREFIX = re.compile(r"^\s*([A-F])(?:[\).\]:\s]|$)", re.I)
AUDIO_TOOLS = {
    "audio_global_caption",
    "audio_qa",
    "audio_ASR",
    "Audio_EventList",
    "Audio_EventLocation",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def answer(value: Any) -> str:
    text = str(value or "").strip().upper()
    if match := ANSWER_TAG.search(text):
        return match.group(1)
    if len(text) == 1 and text in "ABCDEF":
        return text
    if match := ANSWER_PREFIX.search(text):
        return match.group(1)
    return ""


def prediction(question: dict[str, Any]) -> str:
    for turn in reversed(question.get("turn_trace") or []):
        if turn.get("planner_action") == "final":
            if result := answer(turn.get("final_answer")):
                return result
    return answer(question.get("response"))


def planner_usage(turn: dict[str, Any]) -> dict[str, int] | None:
    try:
        usage = turn["planner_response"]["generations"][0]["usage_metadata"]
    except (KeyError, IndexError, TypeError):
        return None
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    return {
        "input": int(usage.get("input_tokens") or 0),
        "cached_input": int(input_details.get("cache_read") or 0),
        "output": int(usage.get("output_tokens") or 0),
        "reasoning": int(output_details.get("reasoning") or 0),
    }


def estimate_row(row: dict[str, Any], duration: float) -> dict[str, Any]:
    q = row["question_data"]
    planner = Counter()
    gemini = Counter()
    tool_calls = Counter()
    missing_planner_usage = 0
    failures: list[str] = []

    for turn in q.get("turn_trace") or []:
        action = str(turn.get("planner_action") or "")
        tool_name = str(turn.get("tool_name") or "")
        if action == "tool":
            tool_calls[tool_name or "<unknown>"] += 1
        if action == "error":
            failures.append("planner_action=error")
        if turn.get("tool_error"):
            failures.append(str(turn["tool_error"]))

        usage = planner_usage(turn)
        if usage is None:
            missing_planner_usage += 1
        else:
            planner.update(usage)

        args = turn.get("tool_args") or {}
        for call in turn.get("perception_calls") or []:
            if call.get("error"):
                failures.append(str(call["error"]))
                continue
            gemini["calls"] += 1
            text = str(call.get("system_prompt") or "") + str(
                call.get("prompt") or ""
            )
            gemini["text_input"] += math.ceil(len(text) / 4)
            gemini["visible_output"] += math.ceil(
                len(str(call.get("response") or "")) / 4
            )
            if tool_name in AUDIO_TOOLS:
                gemini["audio_input"] += round(32 * duration)
            elif tool_name == "video_global_qa":
                fps = float(call.get("fps") or 2)
                gemini["video_input"] += round(263 * fps * duration)
                # Historical global-video files contain their audio track.
                gemini["audio_input"] += round(32 * duration)
            elif tool_name == "video_clip_qa":
                time_range = args.get("time_range") or [0, duration]
                start, end = map(float, time_range[:2])
                clip_duration = max(0.2, end - start)
                fps = float(call.get("fps") or 5)
                gemini["video_input"] += round(263 * fps * clip_duration)

    pred = prediction(q)
    gold = answer(q.get("answer"))
    planner_cost = (
        (planner["input"] - planner["cached_input"]) * 2.0
        + planner["cached_input"] * 0.5
        + planner["output"] * 8.0
    ) / 1_000_000
    gemini_cost = (
        gemini["text_input"] * 0.30
        + gemini["video_input"] * 0.30
        + gemini["audio_input"] * 1.00
        + gemini["visible_output"] * 2.50
    ) / 1_000_000
    return {
        "question_id": str(q.get("question_id") or row.get("video_id")),
        "gold_answer": gold,
        "predicted_answer": pred,
        "parseable": bool(pred),
        "correct": bool(gold and pred == gold and not failures),
        "planner_tokens_exact": dict(planner),
        "planner_turns_missing_usage": missing_planner_usage,
        "gemini_tokens_estimated": dict(gemini),
        "combined_tokens": {
            "input_estimated": planner["input"]
            + gemini["text_input"]
            + gemini["video_input"]
            + gemini["audio_input"],
            "output_reasoning_lower_bound": planner["reasoning"],
            "output_non_reasoning_estimated": planner["output"]
            - planner["reasoning"]
            + gemini["visible_output"],
        },
        "tool_calls": sum(tool_calls.values()),
        "tool_calls_by_name": dict(tool_calls),
        "estimated_cost_usd": {
            "planner_exact_from_recorded_usage": planner_cost,
            "gemini_estimated": gemini_cost,
            "total_estimated": planner_cost + gemini_cost,
        },
        "failures": list(dict.fromkeys(failures)),
    }


def add_counters(total: Counter[str], values: dict[str, Any]) -> None:
    total.update({key: value for key, value in values.items() if value is not None})


def summarize(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(metrics)
    planner = Counter()
    gemini = Counter()
    combined = Counter()
    tools = Counter()
    costs = Counter()
    missing_planner_usage = 0
    for item in metrics:
        add_counters(planner, item["planner_tokens_exact"])
        add_counters(gemini, item["gemini_tokens_estimated"])
        add_counters(combined, item["combined_tokens"])
        tools.update(item["tool_calls_by_name"])
        add_counters(costs, item["estimated_cost_usd"])
        missing_planner_usage += item["planner_turns_missing_usage"]
    correct = sum(item["correct"] for item in metrics)
    failures = [item for item in metrics if item["failures"]]
    no_prediction = [item["question_id"] for item in metrics if not item["parseable"]]
    return {
        "questions": n,
        "accuracy": {
            "correct": correct,
            "total": n,
            "percent": 100 * correct / n,
            "no_parseable_prediction": len(no_prediction),
        },
        "tokens_total": {
            "planner_exact": dict(planner),
            "planner_turns_missing_usage": missing_planner_usage,
            "gemini_estimated": dict(gemini),
            "combined": dict(combined),
        },
        "tokens_mean_per_question": {
            "input_estimated": combined["input_estimated"] / n,
            "output_reasoning_lower_bound": combined[
                "output_reasoning_lower_bound"
            ]
            / n,
            "output_non_reasoning_estimated": combined[
                "output_non_reasoning_estimated"
            ]
            / n,
            "planner_input_exact": planner["input"] / n,
            "planner_output_exact": planner["output"] / n,
        },
        "tool_calls": {
            "total": sum(tools.values()),
            "mean_per_question": sum(tools.values()) / n,
            "by_name": dict(tools),
        },
        "estimated_cost_usd": {
            "total": dict(costs),
            "mean_per_question": {
                key: value / n for key, value in costs.items()
            },
        },
        "latency": {
            "available": False,
            "reason": "Legacy output and trace do not persist per-question timing.",
        },
        "failures": {
            "count": len(failures),
            "records": [
                {"question_id": item["question_id"], "reasons": item["failures"]}
                for item in failures
            ],
            "no_prediction_ids": no_prediction,
        },
        "methodology": {
            "planner": "Exact recorded o3 usage; cached input priced separately.",
            "gemini": (
                "Estimated because legacy perception_calls omit usageMetadata: "
                "text=ceil(chars/4), video=263*fps*seconds, audio=32*seconds."
            ),
            "prices_usd_per_million": {
                "o3_input": 2.0,
                "o3_cached_input": 0.5,
                "o3_output_including_reasoning": 8.0,
                "gemini_text_or_video_input": 0.30,
                "gemini_audio_input": 1.0,
                "gemini_output": 2.50,
            },
            "gemini_hidden_thinking": "Unavailable and excluded.",
        },
    }


def process(source: Path, subset: Path, manifest: Path, output_dir: Path) -> None:
    source_rows = load_jsonl(source)
    subset_rows = load_jsonl(subset)
    manifest_rows = load_jsonl(manifest)
    subset_ids = [str(row["id"]) for row in subset_rows]
    if len(subset_ids) != len(set(subset_ids)):
        raise ValueError("Subset manifest contains duplicate IDs")
    source_by_id = {str(row["video_id"]): row for row in source_rows}
    duration_by_id = {str(row["id"]): float(row["duration"]) for row in manifest_rows}
    missing = [sample_id for sample_id in subset_ids if sample_id not in source_by_id]
    missing_duration = [sample_id for sample_id in subset_ids if sample_id not in duration_by_id]
    if missing or missing_duration:
        raise ValueError(f"Missing results={missing}; missing durations={missing_duration}")

    selected = [source_by_id[sample_id] for sample_id in subset_ids]
    metrics = [
        estimate_row(row, duration_by_id[sample_id])
        for sample_id, row in zip(subset_ids, selected)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "output_test.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    (output_dir / "per_question_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics),
        encoding="utf-8",
    )
    summary = summarize(metrics)
    summary["provenance"] = {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_rows": len(source_rows),
        "subset_manifest": str(subset.resolve()),
        "subset_manifest_sha256": sha256(subset),
        "dataset_manifest": str(manifest.resolve()),
        "selected_rows": len(selected),
        "selection": "Exact video_id == subset id; emitted in subset-manifest order.",
        "output_sha256": sha256(output),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(args.source, args.subset, args.manifest, args.output_dir)
