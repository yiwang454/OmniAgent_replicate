from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_inference_metrics.py"
SPEC = importlib.util.spec_from_file_location("aggregate_inference_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def row(sample_id: str, wall: float, adjusted: float | None) -> dict:
    return {
        "video_id": sample_id,
        "question_data": {
            "latency": {
                "wall_clock_seconds": wall,
                "retry_adjusted_seconds": adjusted,
                "successful_model_call_seconds": wall - 1,
                "planner_successful_calls": 2,
                "perception_successful_calls": 1,
            },
            "token_usage": {
                "planner": {
                    "calls": 2,
                    "calls_with_usage": 2,
                    "input_tokens": 100,
                    "cached_input_tokens": 25,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 10,
                    "total_tokens": 120,
                },
                "perception": {
                    "calls": 1,
                    "calls_with_usage": 1,
                    "prompt_tokens": 50,
                    "cached_content_tokens": 0,
                    "candidate_tokens": 5,
                    "thoughts_tokens": 3,
                    "tool_use_prompt_tokens": 0,
                    "total_tokens": 58,
                    "prompt_tokens_by_modality": {"VIDEO": 40, "TEXT": 10},
                    "candidate_tokens_by_modality": {"TEXT": 5},
                },
            },
        },
    }


class AggregateInferenceMetricsTests(unittest.TestCase):
    def test_aggregates_latency_and_both_model_usages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output_test.jsonl"
            path.write_text(
                "".join(
                    json.dumps(item) + "\n"
                    for item in (row("a", 2.0, 1.0), row("b", 4.0, None))
                ),
                encoding="utf-8",
            )
            summary = metrics.aggregate_metrics(path, expected_count=2)

        self.assertEqual(summary["latency_seconds"]["wall_clock_seconds"]["mean"], 3.0)
        self.assertEqual(
            summary["latency_seconds"]["retry_adjusted_seconds"]["null_count"], 1
        )
        self.assertEqual(summary["token_usage"]["planner"]["input_tokens"]["sum"], 200.0)
        self.assertEqual(summary["token_usage"]["perception"]["total_tokens"]["sum"], 116.0)
        self.assertEqual(
            summary["perception_token_modality_totals"]["prompt_tokens_by_modality"],
            {"VIDEO": 80, "TEXT": 20},
        )


if __name__ == "__main__":
    unittest.main()
