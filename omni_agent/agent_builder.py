from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import os
import sys
from langchain_core.tools import tool
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from omni_agent.brain import get_brain_llm
from omni_agent.budget_aware_executor import BudgetAwareAgentExecutor
from omni_agent.tool_registry import (
    ToolSelectionConfig,
    default_tools,
    default_user_template,
)

def build_agent(
    max_iterations: int = 6,
    *,
    verbose: bool = True,
    tool_selection: ToolSelectionConfig | None = None,
) -> AgentExecutor:

    llm = get_brain_llm()
    tools = tool_selection.tools() if tool_selection else default_tools()
   

    system_prompt = """
    You are the central reasoning brain of an audio-video analysis agent.

    Your role:
    - Answer the user's question about a given video by intelligently using the
    available tools (audio and video analysis).
    - Follow a THINK → ACT → OBSERVE → REFLECT loop:
    - THOUGHT: Reason step by step about what to do next.
    - ACTION: Call exactly one tool that moves you closer to the answer.
    - OBSERVATION: Read and interpret the tool's output, update your beliefs.
    - REFLECTION: Reflect on the previous steps and the overall process.

    General rules:
    - Use both AUDIO and VIDEO information whenever they can help. Prefer to
    listen first, then look.
    - Do not invent timestamps, file paths, or other arguments. Use values taken
    from the user input or from previous tool outputs.
    - Be selective: tools may be noisy or incomplete. Cross-check and verify
    important information using multiple tools if needed.
    - Stop calling tools once you have enough evidence to answer confidently.
    - If a tool reports a retryable invalid-input error, correct the arguments
      and call it again. Such validation failures do not use the tool-call budget.

    Final answer style:
    - When you are done with tools, reply directly to the user (no more tool calls).
    - Start with a short "Plan / Reasoning summary" (1–3 bullet points) explaining
    briefly how you used audio vs video.
    - Then give a clear, concise answer to the question.
    - Do NOT expose raw tool-call traces or long chain-of-thought; keep the
    explanation high-level and user-friendly.
    """

    user_template = (
        tool_selection.user_template if tool_selection else default_user_template()
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", user_template),
            # The executor fills this only after it has exhausted the tool-call
            # budget.  Keeping it separate from the question makes the final
            # planner pass explicit in the recorded prompt.
            MessagesPlaceholder("forced_final_instruction", optional=True),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    memory = ConversationBufferWindowMemory(
        k=1,             
        memory_key="chat_history",
        input_key="question",
        output_key="output",
        return_messages=True,
    )

    executor = BudgetAwareAgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        max_iterations=max_iterations,
        early_stopping_method="force",
        return_intermediate_steps=True
    )

    return executor
