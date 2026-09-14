#!/usr/bin/env python3
"""Evaluate OmniAgent DailyOmni output while separating unusable responses.

The record normalization and answer extraction follow the conventions used by
react-agent-avqa-optimize/scripts/visualize_dspy_output.py. Explicit run errors
and literal empty responses are excluded from the requested population. A
non-empty response without a parseable answer is reported separately and is
counted as wrong in the strict metric.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"\b([A-F])\b", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-F])\s*</answer>", re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(r"^\s*([A-F])(?:[\).\]:\s]|$)", re.IGNORECASE)
EMPTY_MARKERS = {"", "<empty>", "none", "null", "{}", "[]"}
TASK_CATEGORIES = (
    "AV Event Alignment",
    "Inference",
    "Event Sequence",
    "Reasoning",
    "Comparative",
    "Context understanding",
)


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if match := ANSWER_TAG_RE.search(text):
        return match.group(1).upper()
    if len(text) == 1 and text in "ABCDEF":
        return text
    if match := ANSWER_PREFIX_RE.search(text):
        return match.group(1).upper()
    if match := ANSWER_RE.search(text):
        return match.group(1).upper()
    return ""


def question_data(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("question_data")
    return value if isinstance(value, Mapping) else record


def record_id(record: Mapping[str, Any], fallback: str = "") -> str:
    qd = question_data(record)
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(
        first_non_empty(
            qd.get("question_id"),
            record.get("video_id"),
            metadata.get("video_id"),
            fallback,
        )
        or ""
    )


def extract_prediction(record: Mapping[str, Any]) -> str:
    qd = question_data(record)
    if answer := normalize_answer(qd.get("pred_answer")):
        return answer

    trace = qd.get("turn_trace")
    if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes, bytearray)):
        for turn in reversed(trace):
            if not isinstance(turn, Mapping):
                continue
            if str(turn.get("planner_action") or "").strip().lower() == "final":
                if answer := normalize_answer(turn.get("final_answer")):
                    return answer

    return normalize_answer(
        first_non_empty(qd.get("response"), qd.get("reasoning_summary"))
    )


def error_reason(record: Mapping[str, Any]) -> str:
    qd = question_data(record)
    reasons: list[str] = []
    if qd.get("error"):
        reasons.append(str(qd["error"]))
    response = str(qd.get("response") or "").strip()
    if response.startswith("[ERROR]"):
        reasons.append(response)

    trace = qd.get("turn_trace")
    if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes, bytearray)):
        for turn in trace:
            if not isinstance(turn, Mapping):
                continue
            if str(turn.get("planner_action") or "").strip().lower() == "error":
                reasons.append("planner_action=error")
            for key in ("tool_error", "planner_error"):
                if turn.get(key):
                    reasons.append(str(turn[key]))
            calls = turn.get("perception_calls")
            if isinstance(calls, Sequence) and not isinstance(
                calls, (str, bytes, bytearray)
            ):
                for call in calls:
                    if isinstance(call, Mapping) and call.get("error"):
                        reasons.append(str(call["error"]))
    return " | ".join(dict.fromkeys(reasons))


def is_empty_response(record: Mapping[str, Any]) -> bool:
    response = question_data(record).get("response")
    if response is None:
        return True
    if isinstance(response, str):
        return response.strip().lower() in EMPTY_MARKERS
    if isinstance(response, (Mapping, Sequence)) and not isinstance(
        response, (str, bytes, bytearray)
    ):
        return len(response) == 0
    return False


def normalize_record(item: Mapping[str, Any], fallback_id: str = "") -> dict[str, Any]:
    record = dict(item)
    if isinstance(record.get("question_data"), Mapping):
        return record
    return {
        "video_id": fallback_id or record.get("question_id"),
        "metadata": {"video_id": fallback_id or record.get("question_id")},
        "question_data": record,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, Mapping):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(normalize_record(item))
    return records


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return [normalize_record(payload, path.stem)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [normalize_record(item) for item in payload if isinstance(item, Mapping)]
    raise ValueError(f"Expected a JSON object or list in {path}")


def resolve_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        aggregate = path / "output_test.jsonl"
        if aggregate.is_file():
            return aggregate
        raise FileNotFoundError(f"No output_test.jsonl in {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_records(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    resolved = resolve_input(path)
    records = _load_json(resolved) if resolved.suffix == ".json" else _load_jsonl(resolved)
    ids = [record_id(record) for record in records]
    missing = sum(not value for value in ids)
    duplicates = len(ids) - len(set(ids))
    if missing or duplicates:
        raise ValueError(f"Invalid record IDs: missing={missing}, duplicates={duplicates}")
    return resolved, records


def classify_error(reason: str) -> str:
    lowered = reason.lower()
    if "required oneof field 'data'" in lowered:
        return "gemini_part_oneof_400"
    if "gemini api error 429" in lowered:
        return "gemini_429"
    if "gemini api error" in lowered:
        return "gemini_other_http"
    if "filenotfounderror" in lowered:
        return "missing_file"
    if "start_time" in lowered or "end_time" in lowered:
        return "invalid_clip_range"
    if "toolexception" in lowered:
        return "other_tool_error"
    return "other_error"


def evaluate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    errors: list[dict[str, str]] = []
    empty_ids: list[str] = []
    no_prediction_ids: list[str] = []
    scored = 0
    correct = 0
    eligible_with_gold = 0
    task_metrics: dict[str, dict[str, int]] = {
        task: {"eligible": 0, "answered": 0, "correct": 0}
        for task in TASK_CATEGORIES
    }

    for record in records:
        sample_id = record_id(record)
        qd = question_data(record)
        reason = error_reason(record)
        if reason:
            errors.append(
                {"question_id": sample_id, "category": classify_error(reason), "reason": reason}
            )
            continue
        if is_empty_response(record):
            empty_ids.append(sample_id)
            continue

        task_type = str(qd.get("task_type") or "<unknown>")
        metrics = task_metrics.setdefault(
            task_type, {"eligible": 0, "answered": 0, "correct": 0}
        )
        metrics["eligible"] += 1
        gold = normalize_answer(first_non_empty(qd.get("answer"), qd.get("gold_answer")))
        pred = extract_prediction(record)
        if gold:
            eligible_with_gold += 1
        if not pred:
            no_prediction_ids.append(sample_id)
            continue
        if not gold:
            continue
        scored += 1
        metrics["answered"] += 1
        if pred == gold:
            correct += 1
            metrics["correct"] += 1

    error_categories = Counter(item["category"] for item in errors)
    eligible = total - len(errors) - len(empty_ids)
    return {
        "total_records": total,
        "excluded_errors": len(errors),
        "excluded_empty_responses": len(empty_ids),
        "eligible_non_error_non_empty": eligible,
        "eligible_with_gold": eligible_with_gold,
        "no_parseable_prediction": len(no_prediction_ids),
        "successfully_answered": scored,
        "correct": correct,
        "accuracy_successfully_answered_percent": 100 * correct / scored if scored else 0.0,
        "strict_accuracy_after_error_empty_exclusion_percent": (
            100 * correct / eligible_with_gold if eligible_with_gold else 0.0
        ),
        "answered_coverage_of_all_records_percent": 100 * scored / total if total else 0.0,
        "answered_coverage_of_eligible_percent": 100 * scored / eligible if eligible else 0.0,
        "error_categories": dict(sorted(error_categories.items())),
        "task_metrics": task_metrics,
        "error_records": errors,
        "empty_response_ids": empty_ids,
        "no_prediction_ids": no_prediction_ids,
    }


def print_report(source: Path, metrics: Mapping[str, Any]) -> None:
    print(f"Source: {source}")
    print(f"Total records: {metrics['total_records']}")
    print(f"Excluded errors: {metrics['excluded_errors']}")
    print(f"Excluded empty responses: {metrics['excluded_empty_responses']}")
    print(f"Eligible non-error/non-empty: {metrics['eligible_non_error_non_empty']}")
    print(f"No parseable prediction: {metrics['no_parseable_prediction']}")
    print(f"Successfully answered: {metrics['successfully_answered']}")
    print(f"Correct: {metrics['correct']}")
    print(
        "Accuracy among successfully answered: "
        f"{metrics['accuracy_successfully_answered_percent']:.2f}%"
    )
    print(
        "Strict accuracy after excluding only errors/empty responses: "
        f"{metrics['strict_accuracy_after_error_empty_exclusion_percent']:.2f}%"
    )
    print(
        "Answered coverage: "
        f"{metrics['successfully_answered']}/{metrics['total_records']} "
        f"({metrics['answered_coverage_of_all_records_percent']:.2f}% of all; "
        f"{metrics['answered_coverage_of_eligible_percent']:.2f}% of eligible)"
    )
    print("Error categories:")
    for category, count in metrics["error_categories"].items():
        print(f"  {category}: {count}")
    print("Task accuracy among successfully answered:")
    for task, values in metrics["task_metrics"].items():
        answered = values["answered"]
        if not values["eligible"]:
            continue
        accuracy = 100 * values["correct"] / answered if answered else 0.0
        print(
            f"  {task}: {values['correct']}/{answered} = {accuracy:.2f}% "
            f"(eligible={values['eligible']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="OmniAgent output_test.jsonl, JSON file, or its containing directory.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable metrics/details JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source, records = load_records(args.input)
    metrics = evaluate(records)
    print_report(source, metrics)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": str(source), **metrics}
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report: {args.json_output}")


if __name__ == "__main__":
    main()
