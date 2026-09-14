"""Structured tool outcomes that affect agent-loop accounting."""

from __future__ import annotations

from typing import Any


BUDGET_EXEMPT_KEY = "_omniagent_budget_exempt"


def invalid_tool_input(message: str) -> dict[str, Any]:
    """Return an observation the brain can correct without spending its budget."""
    return {
        "error": message,
        "retryable": True,
        BUDGET_EXEMPT_KEY: True,
    }


def is_budget_exempt_observation(observation: Any) -> bool:
    return isinstance(observation, dict) and observation.get(BUDGET_EXEMPT_KEY) is True
