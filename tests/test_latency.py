from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from omni_agent import latency


class LatencyTests(unittest.TestCase):
    def test_retry_adjusted_excludes_failed_attempt_and_backoff(self):
        with patch.object(
            latency.time, "perf_counter", side_effect=[0.0, 2.0, 5.0, 9.0, 10.0]
        ):
            tracker = latency.LatencyTracker()
            tracker.start_attempt("planner")
            tracker.finish_attempt(success=False)
            tracker.start_attempt("planner")
            tracker.finish_attempt(success=True)
            summary = tracker.summary(complete=True)

        self.assertEqual(
            summary,
            {
                "wall_clock_seconds": 10.0,
                "retry_adjusted_seconds": 5.0,
                "successful_model_call_seconds": 4.0,
                "planner_successful_calls": 1,
                "perception_successful_calls": 0,
            },
        )

    def test_incomplete_has_null_adjusted_time(self):
        with patch.object(latency.time, "perf_counter", side_effect=[4.0, 7.0, 8.0]):
            tracker = latency.LatencyTracker()
            tracker.start_attempt("perception")
            tracker.finish_attempt(success=False)
            summary = tracker.summary(complete=False)
        self.assertEqual(summary["wall_clock_seconds"], 4.0)
        self.assertIsNone(summary["retry_adjusted_seconds"])

    def test_context_is_isolated_between_workers(self):
        def run(component: str):
            with latency.sample_latency() as tracker:
                with latency.model_component(component):
                    latency.start_model_attempt()
                    latency.finish_model_attempt(success=True)
            return tracker.summary(complete=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            planner, perception = executor.map(run, ("planner", "perception"))
        self.assertEqual(planner["planner_successful_calls"], 1)
        self.assertEqual(planner["perception_successful_calls"], 0)
        self.assertEqual(perception["planner_successful_calls"], 0)
        self.assertEqual(perception["perception_successful_calls"], 1)


if __name__ == "__main__":
    unittest.main()
