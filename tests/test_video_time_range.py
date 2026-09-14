from __future__ import annotations

import unittest

from langchain_classic.agents import BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import tool

from omni_agent.budget_aware_executor import BudgetAwareAgentExecutor
from omni_agent.tool.Video.video_qa import video_clip_qa
from omni_agent.tool.Video.units import legalize_time_range
from omni_agent.tool_outcomes import invalid_tool_input


class _TwoStepAgent(BaseSingleActionAgent):
    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermediate_steps, **kwargs):
        if len(intermediate_steps) < 2:
            return AgentAction(
                tool="budget_probe",
                tool_input={"index": len(intermediate_steps)},
                log="",
            )
        return AgentFinish(return_values={"output": "done"}, log="")

    async def aplan(self, intermediate_steps, **kwargs):
        return self.plan(intermediate_steps, **kwargs)


@tool
def budget_probe(index: int):
    """Return one invalid observation followed by one valid observation."""
    if index == 0:
        return invalid_tool_input("invalid input")
    return {"answer": "valid"}


class VideoTimeRangeTests(unittest.TestCase):
    def test_end_not_after_start_returns_retryable_observation(self):
        result = video_clip_qa.invoke(
            {
                "video_path": "/path/does/not/need/to/exist.mp4",
                "question": "test",
                "time_range": (10, 10),
            }
        )

        self.assertIn("end <= start", result["error"])
        self.assertTrue(result["retryable"])
        self.assertTrue(result["_omniagent_budget_exempt"])

    def test_invalid_observation_does_not_consume_iteration_budget(self):
        executor = BudgetAwareAgentExecutor(
            agent=_TwoStepAgent(),
            tools=[budget_probe],
            max_iterations=2,
            return_intermediate_steps=True,
        )

        result = executor.invoke({"input": "test"})

        self.assertEqual(result["output"], "done")
        self.assertEqual(len(result["intermediate_steps"]), 2)

    def test_in_bounds_range_is_unchanged(self):
        self.assertEqual(legalize_time_range(5, 10, 30), (5.0, 10.0))

    def test_partially_out_of_bounds_ranges_are_clamped(self):
        self.assertEqual(legalize_time_range(-5, 5, 30), (0.0, 5.0))
        self.assertEqual(legalize_time_range(25, 35, 30), (25.0, 30.0))

    def test_range_wholly_after_video_is_shifted_left(self):
        self.assertEqual(legalize_time_range(40, 50, 30), (20.0, 30.0))

    def test_range_wholly_before_video_is_shifted_right(self):
        self.assertEqual(legalize_time_range(-20, -10, 30), (0.0, 10.0))

    def test_outside_range_longer_than_video_uses_full_video(self):
        self.assertEqual(legalize_time_range(40, 80, 30), (0.0, 30.0))
        self.assertEqual(legalize_time_range(-80, -40, 30), (0.0, 30.0))

    def test_non_positive_requested_duration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than start"):
            legalize_time_range(10, 10, 30)
        with self.assertRaisesRegex(ValueError, "greater than start"):
            legalize_time_range(10, 5, 30)

    def test_non_positive_video_duration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Video duration must be positive"):
            legalize_time_range(0, 5, 0)


if __name__ == "__main__":
    unittest.main()
