"""Shared graph state.

All agents share the conversation history (``messages``). The other keys are
control fields used by the supervisor / agent / tool nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.graph import MessagesState

FINISH = "FINISH"
SUPERVISOR = "supervisor"
TOOLS_SUFFIX = "__tools"


class AgentState(MessagesState, total=False):
    # Routing decision of the supervisor (agent name or FINISH).
    next: str
    # Optional instructions given by the supervisor to the routed agent.
    task: str | None
    # Last agent that produced a message.
    active_agent: str | None
    # Target of an agent-to-agent handoff requested by the last tool node.
    handoff: str | None
    # When set, the turn targets one agent directly (no supervisor).
    direct: str | None
    # Set when a guard (budget, tool-call limit, max steps) stopped the turn.
    stopped: bool
    # Number of tool calls executed during the current turn.
    tool_calls_count: int
    # Number of supervisor decisions during the current turn (per graph level).
    supervisor_steps: int
    # Tokens consumed during the whole session (accumulated across turns).
    tokens_used: Annotated[int, operator.add]


def turn_input(messages: list, direct: str | None = None) -> dict:
    """State update that starts a new user turn (resets per-turn counters)."""
    return {
        "messages": messages,
        "next": "",
        "task": None,
        "handoff": None,
        "direct": direct,
        "stopped": False,
        "tool_calls_count": 0,
        "supervisor_steps": 0,
    }
