#!/usr/bin/env python3
# ruff: noqa: D101,D102,D103,T201
"""Audit tool usage and original-question reuse in ReAct/OmniAgent traces."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)

# Calls without one of these semantic question fields remain in tool-usage
# totals, but are excluded from the question-match denominator.
QUESTION_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "ask_perception": ("perceptual_question",),
    "audio_qa": ("question",),
    "video_clip_qa": ("question",),
    "video_global_qa": ("question",),
    "Audio_EventLocation": ("query",),
}


@dataclass(frozen=True)
class PerceptionCall:
    """A question-bearing tool call.

    The historical class name and fields are retained so the copied GEPA-step
    analyzer can continue importing this module.
    """

    question_id: str
    turn_id: int
    original_question: str
    perception_question: str
    observation_answer: str
    exact_match: bool
    first_edge_match: bool
    last_edge_match: bool
    option_answer_match: bool = False
    tool_name: str = "ask_perception"
    question_argument: str = "perceptual_question"

    @property
    def verbatim_match(self) -> bool:
        return bool(self.original_question) and (
            self.original_question in self.perception_question
        )

    @property
    def casefold_verbatim_match(self) -> bool:
        return bool(self.original_question) and (
            self.original_question.casefold() in self.perception_question.casefold()
        )

    @property
    def edge_match(self) -> bool:
        return self.first_edge_match or self.last_edge_match

    @property
    def detected(self) -> bool:
        return self.edge_match or self.option_answer_match


def normalized_words(text: Any) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(str(text or ""))]


def contains_words(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def extract_explicit_option(observation: Any) -> tuple[str, str]:
    """Compatibility helper used by the copied GEPA-step analyzer."""
    if isinstance(observation, dict):
        answer = str(observation.get("answer") or "").strip()
    else:
        answer = str(observation or "").strip()
    match = re.match(
        r"^\s*[\[(]?([A-D])[\])]?(?:\s|[.：:\-]|$)", answer, re.IGNORECASE
    )
    return (match.group(1).upper() if match else ""), answer


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for row_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{row_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{row_number}")
            yield record


def question_data(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("question_data")
    return nested if isinstance(nested, dict) else record


def summarize_tool_usage(
    question_ids: list[str], traces: Iterable[tuple[str, Any]]
) -> dict[str, Any]:
    call_counts: Counter[str] = Counter()
    sample_ids_by_tool: dict[str, set[str]] = defaultdict(set)
    samples_with_any_tool: set[str] = set()
    budget_exempt_calls = 0
    for question_id, trace in traces:
        for turn in trace or []:
            if not isinstance(turn, dict):
                continue
            tool_name = str(turn.get("tool_name") or "").strip()
            if not tool_name:
                continue
            call_counts[tool_name] += 1
            sample_ids_by_tool[tool_name].add(question_id)
            samples_with_any_tool.add(question_id)
            budget_exempt_calls += bool(turn.get("budget_exempt"))
    return {
        "total_tool_calls": sum(call_counts.values()),
        "samples_with_any_tool": len(samples_with_any_tool),
        "samples_without_any_tool": len(question_ids) - len(samples_with_any_tool),
        "budget_exempt_calls": budget_exempt_calls,
        "calls_by_tool": dict(call_counts.most_common()),
        "samples_by_tool": {
            name: len(sample_ids_by_tool[name]) for name in call_counts
        },
    }


def extract_question_argument(tool_name: str, tool_args: Any) -> tuple[str, str]:
    if not isinstance(tool_args, dict):
        return "", ""
    for argument in QUESTION_ARGUMENTS.get(tool_name, ()):
        value = str(tool_args.get(argument) or "").strip()
        if value:
            return argument, value
    return "", ""


def analyze(
    path: Path, edge_word_count: int
) -> tuple[list[str], list[PerceptionCall], dict[str, Any]]:
    question_ids: list[str] = []
    calls: list[PerceptionCall] = []
    traces: list[tuple[str, Any]] = []
    for record in iter_records(path):
        data = question_data(record)
        question_id = str(data.get("question_id") or "").strip()
        if not question_id:
            raise ValueError("Every result row must have question_id")
        question_ids.append(question_id)
        trace = data.get("turn_trace") or []
        traces.append((question_id, trace))
        original_question = str(data.get("question") or "").strip()
        original_words = normalized_words(original_question)
        enough_words = len(original_words) >= edge_word_count
        for turn_index, turn in enumerate(trace, 1):
            if not isinstance(turn, dict):
                continue
            tool_name = str(turn.get("tool_name") or "").strip()
            argument, asked_question = extract_question_argument(
                tool_name, turn.get("tool_args")
            )
            if not asked_question:
                continue
            asked_words = normalized_words(asked_question)
            observation = turn.get("tool_observation")
            observation_answer = (
                str(observation.get("answer") or "").strip()
                if isinstance(observation, dict)
                else str(observation or "").strip()
            )
            calls.append(
                PerceptionCall(
                    question_id=question_id,
                    turn_id=int(turn.get("turn_id") or turn_index),
                    original_question=original_question,
                    perception_question=asked_question,
                    observation_answer=observation_answer,
                    exact_match=contains_words(original_words, asked_words),
                    first_edge_match=enough_words
                    and contains_words(
                        original_words[:edge_word_count], asked_words
                    ),
                    last_edge_match=enough_words
                    and contains_words(
                        original_words[-edge_word_count:], asked_words
                    ),
                    tool_name=tool_name,
                    question_argument=argument,
                )
            )
    duplicates = [
        item for item, count in Counter(question_ids).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {duplicates[:5]}")
    return question_ids, calls, summarize_tool_usage(question_ids, traces)


def metric(count: int, denominator: int) -> dict[str, int | float]:
    return {
        "count": count,
        "denominator": denominator,
        "percent": 100.0 * count / denominator if denominator else 0.0,
    }


MATCHERS = {
    "verbatim_original_question": lambda call: call.verbatim_match,
    "casefold_verbatim_original_question": lambda call: call.casefold_verbatim_match,
    "normalized_full_original_question": lambda call: call.exact_match,
    "exact_original_question": lambda call: call.exact_match,
    "first_edge": lambda call: call.first_edge_match,
    "last_edge": lambda call: call.last_edge_match,
    "rough_first_or_last_edge": lambda call: call.edge_match,
    "first_or_last_edge": lambda call: call.edge_match,
    "explicit_abcd_observation": lambda call: call.option_answer_match,
    "combined_edge_or_abcd": lambda call: call.detected,
}


def build_summary(
    question_ids: list[str], calls: list[PerceptionCall]
) -> dict[str, Any]:
    calls_by_question: dict[str, list[PerceptionCall]] = {
        question_id: [] for question_id in question_ids
    }
    for call in calls:
        calls_by_question[call.question_id].append(call)
    question_call_ids = {call.question_id for call in calls}
    metrics: dict[str, Any] = {}
    for name, predicate in MATCHERS.items():
        call_count = sum(predicate(call) for call in calls)
        sample_count = sum(
            any(predicate(call) for call in sample_calls)
            for sample_calls in calls_by_question.values()
        )
        metrics[name] = {
            "question_bearing_calls": metric(call_count, len(calls)),
            "all_samples": metric(sample_count, len(question_ids)),
            "samples_with_question_bearing_tool": metric(
                sample_count, len(question_call_ids)
            ),
        }

    per_tool: dict[str, Any] = {}
    for tool_name in sorted({call.tool_name for call in calls}):
        tool_calls = [call for call in calls if call.tool_name == tool_name]
        tool_sample_ids = {call.question_id for call in tool_calls}
        per_tool[tool_name] = {
            "question_bearing_calls": len(tool_calls),
            "samples": len(tool_sample_ids),
            "matches": {
                name: {
                    "calls": metric(
                        sum(predicate(call) for call in tool_calls), len(tool_calls)
                    ),
                    "samples": metric(
                        len(
                            {
                                call.question_id
                                for call in tool_calls
                                if predicate(call)
                            }
                        ),
                        len(tool_sample_ids),
                    ),
                }
                for name, predicate in MATCHERS.items()
            },
        }
    return {
        "total_samples": len(question_ids),
        "samples_with_question_bearing_tool": len(question_call_ids),
        "samples_with_perception": len(question_call_ids),
        "samples_without_question_bearing_tool": (
            len(question_ids) - len(question_call_ids)
        ),
        "samples_without_perception": len(question_ids) - len(question_call_ids),
        "total_question_bearing_calls": len(calls),
        "total_perception_calls": len(calls),
        "samples_with_multiple_question_bearing_calls": sum(
            len(sample_calls) > 1 for sample_calls in calls_by_question.values()
        ),
        "samples_with_multiple_perception_calls": sum(
            len(sample_calls) > 1 for sample_calls in calls_by_question.values()
        ),
        "metrics": metrics,
        "by_tool": per_tool,
    }


def percent_text(item: dict[str, Any]) -> str:
    return f"{item['percent']:.2f}% ({item['count']}/{item['denominator']})"


def render_markdown(
    source: Path, edge_word_count: int, summary: dict[str, Any]
) -> str:
    tool_usage = summary["tool_usage"]
    tool_rows = []
    for tool_name, call_count in tool_usage["calls_by_tool"].items():
        sample_count = tool_usage["samples_by_tool"].get(tool_name, 0)
        tool_rows.append(
            f"| `{tool_name}` | {call_count} | "
            f"{100 * call_count / tool_usage['total_tool_calls']:.2f}% | "
            f"{sample_count} | "
            f"{100 * sample_count / summary['total_samples']:.2f}% |"
        )

    labels = {
        "verbatim_original_question": "Verbatim (case-sensitive)",
        "casefold_verbatim_original_question": "Verbatim (case-insensitive)",
        "normalized_full_original_question": "Full normalized question",
        "first_edge": f"First {edge_word_count} normalized words",
        "last_edge": f"Last {edge_word_count} normalized words",
        "rough_first_or_last_edge": (
            f"Rough: first or last {edge_word_count} words"
        ),
    }
    match_rows = []
    for key, label in labels.items():
        value = summary["metrics"][key]
        match_rows.append(
            f"| {label} | "
            f"{percent_text(value['question_bearing_calls'])} | "
            f"{percent_text(value['all_samples'])} | "
            f"{percent_text(value['samples_with_question_bearing_tool'])} |"
        )

    per_tool_rows = []
    for tool_name, data in summary["by_tool"].items():
        matches = data["matches"]
        per_tool_rows.append(
            f"| `{tool_name}` | {data['question_bearing_calls']} | "
            f"{data['samples']} | "
            f"{percent_text(matches['verbatim_original_question']['calls'])} | "
            f"{percent_text(matches['normalized_full_original_question']['calls'])} | "
            f"{percent_text(matches['rough_first_or_last_edge']['calls'])} |"
        )

    return "\n".join(
        [
            "# OmniAgent tool usage and original-question reuse",
            "",
            f"Source: `{source}`",
            "",
            "## Scope",
            "",
            f"- Samples: {summary['total_samples']}",
            f"- Total tool calls: {tool_usage['total_tool_calls']}",
            (
                "- Question-bearing tool calls: "
                f"{summary['total_question_bearing_calls']}"
            ),
            (
                "- Samples with a question-bearing tool: "
                f"{summary['samples_with_question_bearing_tool']}"
            ),
            "",
            "The copied ReAct analyzer was not directly valid for this run: it "
            "only selected `ask_perception.perceptual_question`. This report "
            "recognizes OmniAgent's `question` and `query` arguments using an "
            "explicit per-tool schema. Calls without a semantic question "
            "argument remain in tool-usage totals but are excluded from the "
            "question-match denominator.",
            "",
            "## Tool usage",
            "",
            (
                "| Tool | Calls | Share of all calls | Samples using tool | "
                "Sample coverage |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
            *tool_rows,
            "",
            "## Original-question reuse",
            "",
            (
                "| Match definition | Question-bearing calls | "
                "All samples with >=1 match | Samples with question tool |"
            ),
            "| --- | ---: | ---: | ---: |",
            *match_rows,
            "",
            (
                "`Rough` follows the copied analyzer's edge heuristic: after "
                "lowercasing and removing punctuation, either the first or last "
                f"{edge_word_count} words of the original question must occur "
                "contiguously in the tool question. It includes full normalized "
                "matches and is a lexical heuristic, not a semantic-similarity "
                "judgment."
            ),
            "",
            "## Match frequency by tool",
            "",
            (
                "| Tool | Question calls | Samples | Verbatim | "
                "Full normalized | Rough edge |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *per_tool_rows,
            "",
            (
                "Detailed artifacts: `summary.json`, "
                "`tool_question_calls.csv`, and "
                "`rough_only_review_sample.json`."
            ),
            "",
        ]
    )


def write_calls_csv(path: Path, calls: list[PerceptionCall]) -> None:
    fieldnames = [
        "question_id",
        "turn_id",
        "tool_name",
        "question_argument",
        "verbatim_match",
        "casefold_verbatim_match",
        "exact_match",
        "first_edge_match",
        "last_edge_match",
        "edge_match",
        "original_question",
        "tool_question",
        "observation_answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for call in calls:
            writer.writerow(
                {
                    "question_id": call.question_id,
                    "turn_id": call.turn_id,
                    "tool_name": call.tool_name,
                    "question_argument": call.question_argument,
                    "verbatim_match": call.verbatim_match,
                    "casefold_verbatim_match": call.casefold_verbatim_match,
                    "exact_match": call.exact_match,
                    "first_edge_match": call.first_edge_match,
                    "last_edge_match": call.last_edge_match,
                    "edge_match": call.edge_match,
                    "original_question": call.original_question,
                    "tool_question": call.perception_question,
                    "observation_answer": call.observation_answer,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-word-count", type=int, default=3)
    parser.add_argument("--review-sample-size", type=int, default=20)
    parser.add_argument("--review-seed", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.edge_word_count < 1 or args.review_sample_size < 0:
        raise SystemExit(
            "--edge-word-count must be >= 1 and "
            "--review-sample-size must be >= 0"
        )
    source = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    question_ids, calls, tool_usage = analyze(source, args.edge_word_count)
    summary = build_summary(question_ids, calls)
    rough_only = [
        call for call in calls if call.edge_match and not call.exact_match
    ]
    summary.update(
        {
            "source": str(source),
            "edge_word_count": args.edge_word_count,
            "review_seed": args.review_seed,
            "review_sample_size": min(
                args.review_sample_size, len(rough_only)
            ),
            "tool_usage": tool_usage,
        }
    )
    rng = random.Random(args.review_seed)
    review = rng.sample(
        rough_only, min(args.review_sample_size, len(rough_only))
    )
    report = render_markdown(source, args.edge_word_count, summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "rough_only_review_sample.json").write_text(
        json.dumps(
            [asdict(call) for call in review],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_calls_csv(output_dir / "tool_question_calls.csv", calls)
    print(report)
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
