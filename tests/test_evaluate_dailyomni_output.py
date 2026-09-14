from __future__ import annotations

import unittest

from scripts.evaluate_dailyomni_output import evaluate, extract_prediction


def _record(sample_id: str, response: str, answer: str = "A", **turn_fields):
    turn = {
        "planner_action": "final",
        "final_answer": response,
        "tool_error": None,
        **turn_fields,
    }
    return {
        "video_id": sample_id,
        "question_data": {
            "question_id": sample_id,
            "task_type": "Inference",
            "answer": answer,
            "response": response,
            "turn_trace": [turn],
        },
    }


class DailyOmniEvaluationTests(unittest.TestCase):
    def test_extracts_tagged_final_answer_from_trace(self):
        record = _record("q1", "summary\n<answer>C</answer>", answer="C")
        record["question_data"]["response"] = "summary only"
        self.assertEqual(extract_prediction(record), "C")

    def test_excludes_errors_and_empty_but_reports_no_prediction(self):
        records = [
            _record("correct", "<answer>A</answer>"),
            _record("wrong", "<answer>B</answer>"),
            _record("error", "[ERROR] ToolException: failed", planner_action="error"),
            _record("empty", ""),
            _record("no-answer", "Agent stopped due to max iterations."),
        ]
        metrics = evaluate(records)
        self.assertEqual(metrics["total_records"], 5)
        self.assertEqual(metrics["excluded_errors"], 1)
        self.assertEqual(metrics["excluded_empty_responses"], 1)
        self.assertEqual(metrics["eligible_non_error_non_empty"], 3)
        self.assertEqual(metrics["no_parseable_prediction"], 1)
        self.assertEqual(metrics["successfully_answered"], 2)
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["accuracy_successfully_answered_percent"], 50.0)
        self.assertAlmostEqual(
            metrics["strict_accuracy_after_error_empty_exclusion_percent"],
            100 / 3,
        )


if __name__ == "__main__":
    unittest.main()
