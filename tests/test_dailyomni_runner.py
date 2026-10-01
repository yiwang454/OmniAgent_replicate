from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from omni_agent.tracing import RolloutTraceRecorder


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("omniagent_main", REPO_ROOT / "main.py")
assert SPEC is not None and SPEC.loader is not None
main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main)


def sample_cut() -> dict:
    return {
        "id": "video-1",
        "supervisions": [
            {
                "id": "video-1",
                "text": "What happens?",
                "custom": {
                    "options": ["A. One", "B. Two", "C. Three", "D. Four"],
                    "answer": "B",
                    "task_type": "Event Sequence",
                    "video_path": "/data/video.mp4",
                },
            }
        ],
    }


class DailyOmniRunnerTests(unittest.TestCase):
    def test_build_input_appends_options_and_answer_contract(self):
        payload = main.build_input(sample_cut())
        self.assertEqual(payload["video_path"], "/data/video.mp4")
        self.assertIn("What happens?\nA. One\nB. Two\nC. Three\nD. Four", payload["question"])
        self.assertTrue(payload["question"].endswith("<answer>A</answer>."))

    def test_benchmark_writes_sibling_schema_and_fixed_aggregate_name(self):
        cut = sample_cut()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_jsonl = root / "input.jsonl"
            input_jsonl.write_text(json.dumps(cut) + "\n", encoding="utf-8")
            output_dir = root / "rollouts"
            args = argparse.Namespace(
                input_jsonl=input_jsonl,
                output_dir=output_dir,
                sample_id=None,
                limit=None,
                max_iterations=30,
                print_steps=False,
                resume=True,
            )
            fake_trace = [{"turn_id": 1, "planner_prompt": [], "planner_response": {}}]
            with patch.object(
                main,
                "invoke_agent",
                return_value=({"output": "<answer>B</answer>"}, fake_trace),
            ):
                main.run_benchmark(args)

            aggregate = output_dir / "output_test.jsonl"
            self.assertTrue(aggregate.is_file())
            row = json.loads(aggregate.read_text(encoding="utf-8").strip())
            self.assertEqual(set(row), {"video_id", "metadata", "question_data"})
            self.assertEqual(row["video_id"], "video-1")
            self.assertEqual(row["question_data"]["response"], "<answer>B</answer>")
            self.assertEqual(row["question_data"]["turn_trace"], fake_trace)

            per_question = json.loads(
                (output_dir / "video-1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(per_question, row["question_data"])

    def test_resume_rebuilds_empty_aggregate_and_skips_successful_trajectory(self):
        cut = sample_cut()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_jsonl = root / "input.jsonl"
            input_jsonl.write_text(json.dumps(cut) + "\n", encoding="utf-8")
            output_dir = root / "rollouts"
            output_dir.mkdir()
            (output_dir / "output_test.jsonl").write_text("", encoding="utf-8")
            saved = main.build_result_row(cut, "<answer>B</answer>")["question_data"]
            saved["turn_trace"] = [{"turn_id": 1, "planner_action": "final"}]
            (output_dir / "video-1.json").write_text(
                json.dumps(saved), encoding="utf-8"
            )
            args = argparse.Namespace(
                input_jsonl=input_jsonl,
                output_dir=output_dir,
                sample_id=None,
                limit=None,
                max_iterations=30,
                print_steps=False,
                resume=True,
            )

            with patch.object(main, "invoke_agent") as invoke:
                main.run_benchmark(args)

            invoke.assert_not_called()
            aggregate = main.read_jsonl(output_dir / "output_test.jsonl")
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(
                aggregate[0]["question_data"]["response"], "<answer>B</answer>"
            )

    def test_resume_reruns_error_and_replaces_its_history(self):
        cut = sample_cut()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_jsonl = root / "input.jsonl"
            input_jsonl.write_text(json.dumps(cut) + "\n", encoding="utf-8")
            output_dir = root / "rollouts"
            output_dir.mkdir()
            saved = main.build_result_row(cut, "[ERROR] RuntimeError: failed")[
                "question_data"
            ]
            (output_dir / "video-1.json").write_text(
                json.dumps(saved), encoding="utf-8"
            )
            args = argparse.Namespace(
                input_jsonl=input_jsonl,
                output_dir=output_dir,
                sample_id=None,
                limit=None,
                max_iterations=30,
                print_steps=False,
                resume=True,
            )
            with patch.object(
                main,
                "invoke_agent",
                return_value=(
                    {"output": "<answer>B</answer>"},
                    [{"turn_id": 1, "planner_action": "final"}],
                ),
            ) as invoke:
                main.run_benchmark(args)

            invoke.assert_called_once()
            aggregate = main.read_jsonl(output_dir / "output_test.jsonl")
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(
                aggregate[0]["question_data"]["response"], "<answer>B</answer>"
            )
            saved_again = json.loads(
                (output_dir / "video-1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_again["response"], "<answer>B</answer>")

    def test_benchmark_runs_question_samples_concurrently(self):
        cuts = [sample_cut(), sample_cut()]
        cuts[1]["id"] = "video-2"
        cuts[1]["supervisions"][0]["id"] = "video-2"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_jsonl = root / "input.jsonl"
            input_jsonl.write_text(
                "".join(json.dumps(cut) + "\n" for cut in cuts), encoding="utf-8"
            )
            args = argparse.Namespace(
                input_jsonl=input_jsonl,
                output_dir=root / "rollouts",
                sample_id=None,
                limit=None,
                max_iterations=30,
                print_steps=False,
                resume=True,
                workers=2,
            )
            lock = threading.Lock()
            both_started = threading.Event()
            state = {"active": 0, "max_active": 0}

            def concurrent_invoke(*_args, **_kwargs):
                with lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    if state["active"] == 2:
                        both_started.set()
                both_started.wait(timeout=2)
                with lock:
                    state["active"] -= 1
                return {"output": "<answer>B</answer>"}, []

            with patch.object(main, "invoke_agent", side_effect=concurrent_invoke):
                main.run_benchmark(args)

            self.assertEqual(state["max_active"], 2)

    def test_max_iteration_history_is_preserved_as_original_outcome(self):
        row = {"question_data": {"response": "Agent stopped due to max iterations."}}
        self.assertFalse(main.is_retryable_history(row))

    def test_turn_trace_records_exact_planner_and_perception_io(self):
        recorder = RolloutTraceRecorder()
        first_run_id = uuid4()
        second_run_id = uuid4()
        recorder.on_chat_model_start(
            {"name": "brain"},
            [[HumanMessage(content="planner prompt 1")]],
            run_id=first_run_id,
        )
        recorder.on_llm_end(
            LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "video_global_qa",
                                        "args": {"question": "look"},
                                        "id": "call-1",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                ]
            ),
            run_id=first_run_id,
        )
        perception_call = recorder.start_perception_call(
            media_path="video.mp4",
            prompt="perception prompt",
            system_prompt="perception system",
            model="gemini-2.5-flash",
            fps=2,
        )
        perception_call["response"] = "perception response"
        recorder.on_chat_model_start(
            {"name": "brain"},
            [[HumanMessage(content="planner prompt 2")]],
            run_id=second_run_id,
        )
        recorder.on_llm_end(
            LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="<answer>B</answer>"))]]
            ),
            run_id=second_run_id,
        )

        action = SimpleNamespace(
            tool="video_global_qa",
            tool_input={"question": "look"},
            log="raw tool call",
        )
        trace = main.build_turn_trace(
            {
                "output": "<answer>B</answer>",
                "intermediate_steps": [(action, {"answer": "perception response"})],
            },
            recorder,
        )

        self.assertEqual(trace[0]["planner_prompt"][0][0]["content"], "planner prompt 1")
        self.assertEqual(
            trace[0]["planner_response"]["generations"][0]["tool_calls"][0]["name"],
            "video_global_qa",
        )
        self.assertEqual(trace[0]["perception_calls"][0]["prompt"], "perception prompt")
        self.assertEqual(
            trace[0]["perception_calls"][0]["response"], "perception response"
        )
        self.assertEqual(trace[1]["planner_response_text"], "<answer>B</answer>")


if __name__ == "__main__":
    unittest.main()
