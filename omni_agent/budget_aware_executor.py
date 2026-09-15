"""Agent executor whose retryable input-validation failures are budget-free."""

from __future__ import annotations

import time
from typing import Any

from langchain_classic.agents import AgentExecutor
from langchain_core.agents import AgentFinish
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.messages import SystemMessage
from langchain_core.utils.input import get_color_mapping

from omni_agent.tool_outcomes import is_budget_exempt_observation


FORCED_FINAL_INSTRUCTION = """
The tool-call limit has been reached. This is the final answer round: do not
call any tool. Use the observations already available in the scratchpad and
answer the user's multiple-choice question now. Select exactly one of the
provided choices and return it wrapped in <answer> tags (for example,
<answer>A</answer>).
""".strip()


class BudgetAwareAgentExecutor(AgentExecutor):
    """Do not count marked invalid-input observations toward max_iterations."""

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green", "red"],
        )
        intermediate_steps = []
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()

        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                tool_return = self._get_tool_return(next_step_output[0])
                if tool_return is not None:
                    return self._return(
                        tool_return,
                        intermediate_steps,
                        run_manager=run_manager,
                    )

            budget_exempt = bool(next_step_output) and all(
                is_budget_exempt_observation(observation)
                for _, observation in next_step_output
            )
            if not budget_exempt:
                iterations += 1
            time_elapsed = time.time() - start_time

        # The normal AgentExecutor `force` policy emits a fixed stopped string,
        # which loses otherwise useful evidence collected by the final tool
        # turn.  Spend one planner-only pass at a *turn* limit instead.  The
        # injected system instruction prevents another tool call, so this does
        # not expand the perception/tool budget.
        if self.max_iterations is not None and iterations >= self.max_iterations:
            forced_inputs = dict(inputs)
            forced_inputs["forced_final_instruction"] = [
                SystemMessage(content=FORCED_FINAL_INSTRUCTION)
            ]
            forced_output = self._action_agent.plan(
                self._prepare_intermediate_steps(intermediate_steps),
                callbacks=run_manager.get_child() if run_manager else None,
                **forced_inputs,
            )
            if isinstance(forced_output, AgentFinish):
                return self._return(forced_output, intermediate_steps, run_manager=run_manager)

        output = self._action_agent.return_stopped_response(
            self.early_stopping_method,
            intermediate_steps,
            **inputs,
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)
