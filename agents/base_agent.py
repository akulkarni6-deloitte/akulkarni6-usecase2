"""
Shared ReAct agent scaffolding.

Every specialist agent (Profiler, STTM Generator, Bronze, Silver, Gold,
Reporter) is a single-purpose LangChain ReAct agent: one LLM (chosen via
the LLMClientFactory strategy pattern), a small tool set specific to its
job, and a tightly scoped system prompt. Keeping agents single-purpose
(per the brief's "Efficiency and Precision" principle) keeps token usage
and blast radius small.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from utils.llm_factory import LLMClientFactory


class BaseSpecialistAgent:
    """Wraps an LLM + a fixed tool set + a system prompt into a runnable ReAct agent
    (built on langchain's `create_agent`, a langgraph tool-calling loop)."""

    #: Override in subclasses.
    system_prompt: str = "You are a helpful assistant."
    max_iterations: int = 8

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: Optional[BaseChatModel] = None,
        provider: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.tools = list(tools)
        self.llm = llm or LLMClientFactory.get_client(provider=provider, temperature=temperature)
        self._graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def run(self, input_text: str) -> dict[str, Any]:
        """
        Invoke the agent. Returns a dict shaped like the classic AgentExecutor
        result (`output`, `intermediate_steps`) so callers/tests don't need to
        know which langchain agent runtime is underneath.
        """
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": input_text}]},
            config={"recursion_limit": self.max_iterations * 2 + 4},
        )
        messages = result.get("messages", [])

        final_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        intermediate_steps = [
            (m.name, m.content) for m in messages if isinstance(m, ToolMessage)
        ]
        return {"output": final_text, "intermediate_steps": intermediate_steps, "messages": messages}
