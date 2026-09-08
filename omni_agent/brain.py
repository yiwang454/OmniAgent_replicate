"""Central OpenAI reasoner configuration."""

from langchain_openai import ChatOpenAI

from omni_agent.config import BRAIN_MODEL, OPENAI_API_KEY


def get_brain_llm() -> ChatOpenAI:
    """Return the central planning/reasoning model through the ELM key."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "ELM_API_KEY is not set. Add it to OmniAgent/.env or export it "
            "before starting OmniAgent."
        )

    # Match react-agent-avqa-optimize's ELM route: use the OpenAI SDK default
    # endpoint and pass no custom base_url. BRAIN_MODEL remains unchanged.
    return ChatOpenAI(
        model=BRAIN_MODEL,
        temperature=1,
        api_key=OPENAI_API_KEY,
        reasoning_effort="high",
    )
