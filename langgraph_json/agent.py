"""Graph construction — the generic LangGraph engine.

The graph is rebuilt from the JSON configuration on every reload:

    START ──► supervisor ──► <agent> ──► <agent>__tools ──► <agent> ...
                  ▲              │                 │
                  └──────────────┘    handoff ─────┴──► <other agent>

* one node per enabled agent (``react``) plus its tools node;
* ``sub_graph`` agents are nested supervisor graphs over their ``sub_agents``;
* a turn may target one agent directly (``direct``), bypassing the supervisor.

This module is also the entry point referenced by ``langgraph.json``
(``make_graph``) for the LangGraph CLI / Studio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .utils.backend_client import BackendClient, BackendEventListener
from .utils.config_loader import AgentDef, ConfigBundle, SupervisorDef, load_config
from .utils.llm_factory import LLMFactory, ProviderConfig
from .utils.nodes import (
    HANDOFF_PREFIX,
    NodeContext,
    make_agent_node,
    make_handoff_tool,
    make_subgraph_node,
    make_supervisor_node,
    make_tools_node,
    tools_node_name,
)
from .utils.state import FINISH, SUPERVISOR, AgentState
from .utils.tools import BUILTIN_TOOL_NAMES, ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class GraphInfo:
    """Description of a built graph (used by /status and the event mapper)."""

    top_level: list[str]
    node_agents: set[str]  # every agent name having a node, at any level
    tool_counts: dict[str, int]


class GraphBuilder:
    def __init__(self, bundle: ConfigBundle, registry: ToolRegistry, llms: LLMFactory):
        self.bundle = bundle
        self.registry = registry
        self.llms = llms
        cfg = bundle.agents
        self.ctx = NodeContext(
            settings=bundle.settings,
            get_llm=self._get_llm,
            default_model=cfg.default_model,
            default_provider=cfg.default_provider,
        )
        self.info = GraphInfo(top_level=[], node_agents=set(), tool_counts={})

    async def _get_llm(self, provider: str | None, model: str | None):
        # Token acquisition (Azure AD) may block: keep the event loop free.
        return await asyncio.to_thread(self.llms.get, provider, model)

    def check_providers(self) -> list[str]:
        """Provider names referenced by agents but missing from the provider config."""
        known = self.llms.provider_names()
        if known is None:
            return []
        cfg = self.bundle.agents
        refs = {cfg.default_provider, cfg.supervisor.provider} | {a.provider for a in cfg.enabled_agents().values()}
        return sorted(p for p in refs if p and p not in known)

    # ------------------------------------------------------------------ #
    def build(self, checkpointer=None):
        missing = self.check_providers()
        if missing:
            raise ValueError(f"agents.json references unknown providers: {missing}")
        cfg = self.bundle.agents
        members = [n for n, a in cfg.agents.items() if a.enabled and a.routable]
        self.info.top_level = members
        builder = self._build_level(members, cfg.supervisor, top=True)
        return builder.compile(checkpointer=checkpointer, name="lang-graph-json")

    def _build_level(self, member_names: list[str], spec: SupervisorDef, top: bool) -> StateGraph:
        agents = self.bundle.agents.agents
        members: dict[str, AgentDef] = {n: agents[n] for n in member_names if n in agents and agents[n].enabled}
        g = StateGraph(AgentState)
        g.add_node(SUPERVISOR, make_supervisor_node(members, spec, self.ctx))

        for name, agent in members.items():
            self.info.node_agents.add(name)
            after = [SUPERVISOR, END]
            if agent.type == "sub_graph":
                sub_spec = SupervisorDef(
                    system_prompt=agent.system_prompt,
                    model=agent.model,
                    provider=agent.provider,
                    max_history_messages=agent.max_history_messages
                    if agent.max_history_messages is not None
                    else self.bundle.settings.default_max_history_messages,
                )
                sub = self._build_level(agent.sub_agents, sub_spec, top=False).compile(name=name)
                g.add_node(name, make_subgraph_node(agent, sub))
                g.add_conditional_edges(name, _route_after_finished(name), after)
                continue

            tools = self.registry.for_agent(agent.allowed_tools)
            self.info.tool_counts[name] = len(tools)
            targets = [t for t in agent.can_handoff_to if t in members and t != name]
            handoff_tools = [make_handoff_tool(t, members[t].description) for t in targets]
            handoff_map = {f"{HANDOFF_PREFIX}{t}": t for t in targets}

            g.add_node(name, make_agent_node(agent, tools, handoff_tools, self.ctx))
            tnode = tools_node_name(name)
            g.add_node(tnode, make_tools_node(agent, tools, handoff_map, self.ctx))
            g.add_conditional_edges(name, _route_after_agent(name, tnode), [tnode, SUPERVISOR, END])
            g.add_conditional_edges(tnode, _route_after_tools(name), [name, END, *targets])

        g.add_conditional_edges(START, _route_start(set(members)), [SUPERVISOR, *members])
        g.add_conditional_edges(SUPERVISOR, _route_supervisor(set(members)), [*members, END])
        return g


# --------------------------------------------------------------------------- #
# Routing functions
# --------------------------------------------------------------------------- #
def _route_start(members: set[str]):
    def route(state: AgentState) -> str:
        direct = state.get("direct")
        return direct if direct in members else SUPERVISOR

    return route


def _route_supervisor(members: set[str]):
    def route(state: AgentState) -> str:
        nxt = state.get("next")
        return nxt if nxt in members else END

    return route


def _route_after_agent(name: str, tnode: str):
    def route(state: AgentState) -> str:
        if state.get("stopped"):
            return END
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return tnode
        return END if state.get("direct") == name else SUPERVISOR

    return route


def _route_after_finished(name: str):
    def route(state: AgentState) -> str:
        if state.get("stopped") or state.get("direct") == name:
            return END
        return SUPERVISOR

    return route


def _route_after_tools(name: str):
    def route(state: AgentState) -> str:
        if state.get("stopped"):
            return END
        target = state.get("handoff")
        return target if target else name

    return route


def build_graph(bundle: ConfigBundle, registry: ToolRegistry, llms: LLMFactory, checkpointer=None):
    """Build and compile the graph. Returns ``(compiled_graph, GraphInfo)``."""
    gb = GraphBuilder(bundle, registry, llms)
    graph = gb.build(checkpointer=checkpointer)
    return graph, gb.info


# --------------------------------------------------------------------------- #
# LangGraph CLI / Studio entry point (development only)
# --------------------------------------------------------------------------- #
def _provider_from_env() -> ProviderConfig:
    raw = os.environ.get("LGJ_PROVIDER_JSON")
    if raw:
        return ProviderConfig.model_validate(json.loads(raw))
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderConfig.model_validate(
            {
                "type": "anthropic",
                "api_key": os.environ["ANTHROPIC_API_KEY"],
                "default_model": os.environ.get("LGJ_MODEL", "claude-sonnet-5-5"),
            }
        )
    return ProviderConfig.model_validate({"type": "fake"})


def make_graph(config=None):
    """Graph factory for ``langgraph dev`` / LangGraph Studio.

    Reads ``LGJ_CONFIG_DIR`` (default ``./config``) and the provider from the
    environment (``LGJ_PROVIDER_JSON`` or ``ANTHROPIC_API_KEY``; ``.env`` is
    for local development only). Persistence is provided by the platform.
    """
    config_dir = Path(os.environ.get("LGJ_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))
    bundle = load_config(config_dir, BUILTIN_TOOL_NAMES)
    s = bundle.settings
    registry = ToolRegistry(
        bundle.tools,
        BackendClient(s.backend_url, s.backend_timeout_s),
        BackendEventListener(s.backend_ws_url),
        max_result_chars=s.max_tool_result_chars,
        allowed_modules=s.python_tool_modules,
        plugins_dir=s.resolve_plugins_dir(config_dir),
    )
    graph, _ = build_graph(bundle, registry, LLMFactory(_provider_from_env()))
    return graph


__all__ = ["build_graph", "make_graph", "GraphBuilder", "GraphInfo", "FINISH"]
