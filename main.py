"""Run OmniAgent on one example or the complete DailyOmni benchmark."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omni_agent.agent_builder import build_agent
from omni_agent.tracing import (
    RolloutTraceRecorder,
    activate_trace_recorder,
    json_safe,
)


DEFAULT_INPUT_JSONL = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_BENCHMARK_WORKERS = 4
OUTPUT_FILENAME = "output_test.jsonl"
ANSWER_INSTRUCTION = (
    "Please select the most correct answer (A/B/C/D) and output your choice "
    "wrapped in <answer> tags, e.g., <answer>A</answer>."
)
EXAMPLE_VIDEO_PATH = Path(__file__).resolve().parent / "example/d6b4OmUFt7I_video.mp4"
EXAMPLE_QUESTION = (
    "Which visual sequences correspond to the audio mentions of 'go on the "
    "equipment' versus 'familiar with the machines'?"
)
EXAMPLE_CHOICES = [
    "A. First shows an excavator lifting dirt, second shows a dump truck driving through mud",
    "B. First shows binder review in office, second shows tractor parked on site",
    "C. First shows safety vest demonstration, second shows mountain landscape",
    "D. First shows a bulldozer moving earth, second shows a water tank in background",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows and report malformed line numbers."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object")
            rows.append(row)
    return rows


def cut_id(cut: dict[str, Any]) -> str:
    value = str(cut.get("id") or "")
    if not value:
        raise ValueError("DailyOmni cut has an empty id")
    return value


def supervision_data(cut: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    supervisions = cut.get("supervisions") or []
    if not supervisions or not isinstance(supervisions[0], dict):
        raise ValueError(f"Cut {cut.get('id')} has no supervision")
    supervision = supervisions[0]
    custom = supervision.get("custom") or {}
    return supervision, custom


def build_question_prompt(question: str, options: list[Any]) -> str:
    choices = "\n".join(str(option) for option in options)
    return f"{question}\n{choices}\n{ANSWER_INSTRUCTION}"


def build_input(cut: dict[str, Any]) -> dict[str, str]:
    """Map one DailyOmni cut to the existing OmniAgent input contract."""
    supervision, custom = supervision_data(cut)
    recording = cut.get("recording") or {}
    sources = recording.get("sources") or []
    source_video = sources[0].get("source") if sources and isinstance(sources[0], dict) else None

    question = supervision.get("text") or custom.get("Question") or custom.get("question")
    options = custom.get("options") or custom.get("Choice") or []
    video_path = custom.get("video_path") or source_video
    if not question or not options or not video_path:
        raise ValueError(
            f"Missing DailyOmni fields for cut {cut.get('id')}: "
            f"question={bool(question)}, options={bool(options)}, video_path={bool(video_path)}"
        )
    return {
        "video_path": str(video_path),
        "question": build_question_prompt(str(question), list(options)),
    }


def build_result_row(cut: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Build the process_cut_task-style record used by the sibling repository."""
    supervision, custom = supervision_data(cut)
    sample_id = cut_id(cut)
    return {
        "video_id": sample_id,
        "metadata": {
            "video_id": sample_id,
            "duration": custom.get("duration"),
            "domain": custom.get("domain"),
            "sub_category": custom.get("sub_category"),
        },
        "question_data": {
            "question_id": supervision.get("id"),
            "task_type": custom.get("task_type") or custom.get("Type"),
            "question": supervision.get("text") or custom.get("Question"),
            "options": custom.get("options") or custom.get("Choice"),
            "answer": custom.get("answer") or custom.get("Answer"),
            "response": response_text,
            "fps": None,
        },
    }


def _planner_call_for_step(
    calls: list[dict[str, Any]], index: int
) -> dict[str, Any] | None:
    return calls[index] if index < len(calls) else None


def _perception_calls_between(
    perception_calls: list[dict[str, Any]],
    planner_call: dict[str, Any] | None,
    next_planner_call: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if planner_call is None:
        return []
    lower = planner_call["sequence"]
    upper = next_planner_call["sequence"] if next_planner_call else float("inf")
    return [call for call in perception_calls if lower < call["sequence"] < upper]


def build_turn_trace(
    result: dict[str, Any], recorder: RolloutTraceRecorder
) -> list[dict[str, Any]]:
    """Combine LangChain actions with exact planner/perception API calls."""
    trace: list[dict[str, Any]] = []
    intermediate_steps = result.get("intermediate_steps") or []
    planner_calls = recorder.planner_calls

    for index, step in enumerate(intermediate_steps):
        action, observation = step
        planner_call = _planner_call_for_step(planner_calls, index)
        next_planner_call = _planner_call_for_step(planner_calls, index + 1)
        trace.append(
            {
                "turn_id": index + 1,
                "planner_action": "tool",
                "tool_name": getattr(action, "tool", None),
                "tool_args": json_safe(getattr(action, "tool_input", None)),
                "tool_observation": json_safe(observation),
                "tool_error": None,
                "final_answer": None,
                "planner_raw": getattr(action, "log", None),
                "planner_prompt": (planner_call or {}).get("messages"),
                "planner_response": (planner_call or {}).get("response"),
                "planner_response_text": (planner_call or {}).get("response_text"),
                "perception_calls": json_safe(
                    _perception_calls_between(
                        recorder.perception_calls, planner_call, next_planner_call
                    )
                ),
            }
        )

    final_call = _planner_call_for_step(planner_calls, len(intermediate_steps))
    trace.append(
        {
            "turn_id": len(trace) + 1,
            "planner_action": "final",
            "tool_name": None,
            "tool_args": None,
            "tool_observation": None,
            "tool_error": None,
            "final_answer": str(result.get("output", "")),
            "planner_raw": None,
            "planner_prompt": (final_call or {}).get("messages"),
            "planner_response": (final_call or {}).get("response"),
            "planner_response_text": (final_call or {}).get("response_text"),
            "perception_calls": json_safe(
                _perception_calls_between(recorder.perception_calls, final_call, None)
            ),
        }
    )
    return trace


def invoke_agent(
    video_path: str,
    question: str,
    *,
    max_iterations: int,
    print_steps: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one isolated rollout and return its result and structured trace."""
    os.makedirs("Cache", exist_ok=True)
    # A fresh executor also gives every benchmark question fresh conversation memory.
    agent = build_agent(max_iterations=max_iterations, verbose=print_steps)
    recorder = RolloutTraceRecorder()
    with activate_trace_recorder(recorder):
        result = agent.invoke(
            {"video_path": video_path, "question": question},
            config={"callbacks": [recorder]},
        )
    return result, build_turn_trace(result, recorder)


def write_question_json(output_dir: Path, row: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{row['video_id']}.json"
    temporary_path = output_dir / f".{row['video_id']}.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(row["question_data"]), handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)
    return path


def write_output_jsonl(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write the required aggregate artifact using its fixed filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUTPUT_FILENAME
    temporary_path = output_dir / f".{OUTPUT_FILENAME}.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    temporary_path.replace(path)
    return path


def append_output_row(output_path: Path, row: dict[str, Any]) -> None:
    """Persist one completed row without repeatedly rewriting the full benchmark."""
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def is_retryable_history(row: dict[str, Any]) -> bool:
    """Return whether a saved row should be rerun during resume.

    Max-iteration responses are intentionally treated as completed outcomes to
    preserve the original experiment policy. Only explicit run errors and empty
    responses are retried automatically.
    """
    question_data = row.get("question_data") or {}
    response = str(question_data.get("response") or "").strip()
    return not response or response.startswith("[ERROR]")


def load_resume_rows(
    output_dir: Path,
    cuts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Recover the latest result for each sample from aggregate/per-question files."""
    recovered: dict[str, dict[str, Any]] = {}
    aggregate = output_dir / OUTPUT_FILENAME
    if aggregate.is_file() and aggregate.stat().st_size:
        for row in read_jsonl(aggregate):
            sample_id = str(row.get("video_id") or "")
            if sample_id:
                recovered[sample_id] = row

    # Per-question JSON is the durable trajectory source and takes precedence.
    for cut in cuts:
        sample_id = cut_id(cut)
        question_path = output_dir / f"{sample_id}.json"
        if not question_path.is_file():
            continue
        try:
            question_data = json.loads(question_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid saved trajectory JSON in {question_path}: {exc}") from exc
        if not isinstance(question_data, dict):
            raise ValueError(f"Saved trajectory is not a JSON object: {question_path}")
        row = build_result_row(cut, str(question_data.get("response") or ""))
        row["question_data"] = question_data
        recovered[sample_id] = row
    return recovered


def run_benchmark_sample(
    cut: dict[str, Any],
    *,
    max_iterations: int,
    print_steps: bool,
) -> tuple[dict[str, Any], str | None]:
    """Run one sample independently so benchmark samples can execute concurrently."""
    try:
        payload = build_input(cut)
        result, turn_trace = invoke_agent(
            payload["video_path"],
            payload["question"],
            max_iterations=max_iterations,
            print_steps=print_steps,
        )
        row = build_result_row(cut, str(result.get("output", "")))
        row["question_data"]["turn_trace"] = turn_trace
        return row, None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        row = build_result_row(cut, f"[ERROR] {error}")
        row["question_data"]["turn_trace"] = [
            {
                "turn_id": 1,
                "planner_action": "error",
                "tool_name": None,
                "tool_args": None,
                "tool_observation": None,
                "tool_error": error,
                "final_answer": None,
                "planner_prompt": None,
                "planner_response": None,
                "planner_response_text": None,
            }
        ]
        return row, error


def run_benchmark(args: argparse.Namespace) -> None:
    all_cuts = read_jsonl(args.input_jsonl)
    cuts = all_cuts
    if args.sample_id:
        requested = set(args.sample_id)
        cuts = [cut for cut in cuts if cut_id(cut) in requested]
        found = {cut_id(cut) for cut in cuts}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Sample IDs not found: {', '.join(missing)}")
    if args.limit is not None:
        cuts = cuts[: args.limit]

    print(f"Loaded {len(cuts)} DailyOmni sample(s) from {args.input_jsonl}")
    print(f"Per-question rollouts: {args.output_dir}")
    print(f"Aggregate artifact: {args.output_dir / OUTPUT_FILENAME}")

    resume = getattr(args, "resume", True)
    recovered = load_resume_rows(args.output_dir, all_cuts) if resume else {}
    completed_ids = {
        sample_id
        for sample_id, row in recovered.items()
        if not is_retryable_history(row)
    }
    cuts_to_run = [cut for cut in cuts if cut_id(cut) not in completed_ids]

    # Rebuild first so an interrupted prior aggregate can be recovered entirely
    # from the durable per-question trajectories. Error rows are omitted until
    # they are rerun, preventing stale failures from occupying completed slots.
    preserved_rows = [
        recovered[cut_id(cut)]
        for cut in all_cuts
        if cut_id(cut) in completed_ids
    ]
    output_path = write_output_jsonl(args.output_dir, preserved_rows)
    if resume:
        retry_count = sum(
            sample_id not in completed_ids for sample_id in recovered
        )
        print(
            f"Resume recovered {len(recovered)} trajectory file(s): "
            f"skip={len(completed_ids)}, retry_error_or_empty={retry_count}, "
            f"selected_to_run={len(cuts_to_run)}"
        )

    workers = getattr(args, "workers", DEFAULT_BENCHMARK_WORKERS)
    print(f"Concurrent question samples: {workers}")
    completed = 0
    future_metadata = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="omniagent") as executor:
        for index, cut in enumerate(cuts_to_run, start=1):
            sample_id = cut_id(cut)
            print(f"[{index}/{len(cuts_to_run)}] {sample_id}")
            future = executor.submit(
                run_benchmark_sample,
                cut,
                max_iterations=args.max_iterations,
                print_steps=args.print_steps,
            )
            future_metadata[future] = (index, sample_id)

        for future in as_completed(future_metadata):
            index, sample_id = future_metadata[future]
            row, error = future.result()
            if error is not None:
                print(f"[{index}/{len(cuts_to_run)}] ERROR {sample_id}: {error}")

            # Keep writes in the coordinator: per-question files are atomic and
            # the aggregate stays valid while worker rollouts finish out of order.
            write_question_json(args.output_dir, row)
            append_output_row(output_path, row)
            completed += 1

    print(
        f"Wrote {completed} new row(s); aggregate now has "
        f"{len(preserved_rows) + completed} row(s): {output_path}"
    )


def run_single(args: argparse.Namespace) -> None:
    if args.example:
        video_path = str(EXAMPLE_VIDEO_PATH)
        question = build_question_prompt(EXAMPLE_QUESTION, EXAMPLE_CHOICES)
    else:
        video_path = args.video_path
        question = args.question

    result, _ = invoke_agent(
        video_path,
        question,
        max_iterations=args.max_iterations,
        print_steps=args.print_steps,
    )
    print("\n=== Final Answer ===")
    print(result["output"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniAgent on DailyOmni (default), the bundled example, or one custom input."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help=f"DailyOmni cut JSONL (default: {DEFAULT_INPUT_JSONL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Rollout directory. The aggregate is always written as output_test.jsonl here.",
    )
    parser.add_argument("--example", action="store_true", help="Run the bundled example only.")
    parser.add_argument("--video_path", type=str, default=None, help="Custom single-video path.")
    parser.add_argument("--question", type=str, default=None, help="Custom single-video question.")
    parser.add_argument(
        "--print-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print LangChain's step-by-step rollout (use --no-print-steps to disable).",
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_BENCHMARK_WORKERS,
        help=(
            "Number of DailyOmni question samples to run concurrently "
            f"(default: {DEFAULT_BENCHMARK_WORKERS})."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from per-question JSON trajectories in --output-dir "
            "(default). Use --no-resume to intentionally rerun everything."
        ),
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Run only this DailyOmni cut ID; may be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected samples (useful for smoke tests).",
    )
    args = parser.parse_args()

    if args.example and (args.video_path or args.question):
        parser.error("--example cannot be combined with --video_path or --question")
    if bool(args.video_path) != bool(args.question):
        parser.error("--video_path and --question must be provided together")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if args.example or args.video_path:
        run_single(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
