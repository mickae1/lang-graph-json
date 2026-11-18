"""Streaming events: LangGraph ``astream_events(version="v2")`` → unified format.

Every event sent on ``WS /stream`` looks like::

    {"session_id": "...", "event_type": "tool.start",
     "timestamp": "2026-11-18T14:32:01.234Z", "data": {...}}
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .nodes import message_text
from .state import SUPERVISOR, TOOLS_SUFFIX

log = logging.getLogger(__name__)

ALL_SESSIONS = "*"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def envelope(session_id: str, event_type: str, data: dict) -> dict:
    return {"session_id": session_id, "event_type": event_type, "timestamp": now_iso(), "data": data}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    content = getattr(value, "content", None)
    if content is not None:
        return message_text(content)
    return str(value)


def map_langgraph_event(ev: dict, agent_names: set[str]) -> tuple[str, dict] | None:
    """Map one LangGraph v2 event to ``(event_type, data)`` or None (ignored)."""
    kind = ev.get("event")
    meta = ev.get("metadata") or {}
    node = meta.get("langgraph_node") or ""
    agent = node[: -len(TOOLS_SUFFIX)] if node.endswith(TOOLS_SUFFIX) else node
    data = ev.get("data") or {}
    known = agent_names | {SUPERVISOR}

    if kind == "on_chat_model_stream":
        chunk = data.get("chunk")
        token = message_text(getattr(chunk, "content", ""))
        if token and agent != SUPERVISOR:
            return "llm.stream", {"agent_name": agent, "token": token}
        return None
    if kind == "on_chat_model_start":
        return "llm.start", {"agent_name": agent, "model": meta.get("ls_model_name")}
    if kind == "on_chat_model_end":
        out = data.get("output")
        usage = getattr(out, "usage_metadata", None)
        return "llm.end", {"agent_name": agent, "usage": dict(usage) if usage else None}
    if kind == "on_tool_start":
        return "tool.start", {"agent_name": agent, "tool_name": ev.get("name"), "tool_args": _jsonable(data.get("input"))}
    if kind == "on_tool_end":
        result = _jsonable(data.get("output"))
        if isinstance(result, str) and len(result) > 2000:
            result = result[:2000] + "…"
        return "tool.end", {"agent_name": agent, "tool_name": ev.get("name"), "status": "ok", "result": result}
    if kind == "on_tool_error":
        return "tool.end", {"agent_name": agent, "tool_name": ev.get("name"), "status": "error", "result": str(data.get("error"))}
    if kind in ("on_chain_start", "on_chain_end"):
        name = ev.get("name")
        if name and name == node and name in known:
            return ("agent.start" if kind == "on_chain_start" else "agent.end"), {"agent_name": name}
    return None


class EventBus:
    """Fan-out of events to WebSocket subscribers (per session or all)."""

    def __init__(self, queue_size: int = 1000):
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self.queue_size = queue_size

    def subscribe(self, session_id: str | None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._subs[session_id or ALL_SESSIONS].add(q)
        return q

    def unsubscribe(self, session_id: str | None, q: asyncio.Queue) -> None:
        key = session_id or ALL_SESSIONS
        self._subs[key].discard(q)
        if not self._subs[key]:
            self._subs.pop(key, None)

    @property
    def subscriber_count(self) -> int:
        return sum(len(s) for s in self._subs.values())

    def publish(self, event: dict) -> None:
        targets = set(self._subs.get(event.get("session_id"), ())) | set(self._subs.get(ALL_SESSIONS, ()))
        for q in targets:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("event queue full — dropping %s", event.get("event_type"))
