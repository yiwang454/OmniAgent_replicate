"""Central OpenAI reasoner configuration."""

import httpx
from collections.abc import AsyncIterator, Iterator
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from omni_agent.config import BRAIN_MODEL, OPENAI_API_KEY
from omni_agent.latency import (
    httpx_request_started,
    httpx_response_received,
    model_component,
)


class LatencyTrackedChatOpenAI(ChatOpenAI):
    """Associate OpenAI SDK HTTP attempts with the planner component."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        with model_component("planner"):
            return super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

    def _stream(self, *args, **kwargs) -> Iterator:
        # LangChain's tools agent consumes the planner as a stream even when
        # ``streaming`` is false on the model, so this path needs its own
        # component boundary.
        with model_component("planner"):
            yield from super()._stream(*args, **kwargs)

    async def _astream(self, *args, **kwargs) -> AsyncIterator:
        with model_component("planner"):
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk


def get_brain_llm() -> ChatOpenAI:
    """Return the central planning/reasoning model through the ELM key."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "ELM_API_KEY is not set. Add it to OmniAgent/.env or export it "
            "before starting OmniAgent."
        )

    # Match react-agent-avqa-optimize's ELM route: use the OpenAI SDK default
    # endpoint and pass no custom base_url. BRAIN_MODEL remains unchanged.
    http_client = httpx.Client(
        event_hooks={
            "request": [httpx_request_started],
            "response": [httpx_response_received],
        }
    )
    return LatencyTrackedChatOpenAI(
        model=BRAIN_MODEL,
        temperature=1,
        api_key=OPENAI_API_KEY,
        reasoning_effort="high",
        stream_usage=True,
        http_client=http_client,
    )
