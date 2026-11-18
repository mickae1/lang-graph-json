"""Server runtime: configuration, graph lifecycle, sessions and turns.

* One checkpointer per runtime, so the conversation history survives
  hot-reloads of the graph (only the graph is rebuilt). It is in memory by
  default (nothing on disk), SQLite when ``server.json`` says
  ``"persistence": "sqlite"``, or any checkpointer injected by a library user.
* A maintenance task expires inactive sessions (``session_ttl_hours``) and
  denies confirmations left unanswered (``confirmation_timeout_s``).
* Turns can be cancelled (``cancel``) and are cancelled after
  ``turn_timeout_s``. The history is then repaired, so the session stays
  usable (unanswered tool calls receive a "cancelled" result).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from .. import __version__
from ..agent import GraphBuilder, GraphInfo, build_graph
from ..utils.backend_client import BackendClient, BackendEventListener
from ..utils.config_loader import (
    AGENTS_FILE,
    SERVER_FILE,
    TOOLS_FILE,
    ConfigBundle,
    ConfigError,
    ConfigWatcher,
    ServerSettings,
    config_from_data,
    load_config,
    load_settings,
)
from ..utils.events import EventBus, envelope, map_langgraph_event
from ..utils.llm_factory import LLMFactory, ProviderConfig, ProviderNotConfigured, ProxyConfig, StaticLLMFactory
from ..utils.nodes import message_text
from ..utils.state import FINISH, SUPERVISOR, turn_input
from ..utils.tools import BUILTIN_TOOL_NAMES, ToolRegistry, tooldef_from_function

log = logging.getLogger(__name__)

SQLITE_HINT = 'persistence "sqlite" needs the optional dependency: pip install "lang-graph-json[sqlite]"'


class NotReady(RuntimeError):
    """The server cannot run a turn yet (no provider or invalid config)."""


class Conflict(RuntimeError):
    pass


@dataclass
class Session:
    id: str
    created: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turn_start: int = 0
    turns: int = 0
    budget_warned: bool = False
    # Set while a confirmation is waiting for an answer.
    pending_since: float | None = None
    # Running turn (for cancellation) and why it was cancelled.
    task: asyncio.Task | None = None
    cancel_reason: str | None = None
    # Restored from the checkpointer: check for a pending confirmation once
    # the graph exists (the provider may be pushed after startup).
    check_pending: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


def serialize_message(m: BaseMessage) -> dict:
    d: dict[str, Any] = {"type": m.type, "name": getattr(m, "name", None), "content": message_text(m.content)}
    if isinstance(m, AIMessage) and m.tool_calls:
        d["tool_calls"] = [{"id": c["id"], "name": c["name"], "args": c["args"]} for c in m.tool_calls]
    if isinstance(m, ToolMessage):
        d["tool_call_id"] = m.tool_call_id
        d["status"] = m.status
    return d


def _last_human_index(messages: list) -> int:
    return next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), 0)


def _parse_ts(ts: Any) -> float:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return time.time()


class Runtime:
    """The agent engine. Used by the standalone server, or directly as a library.

    Configuration comes either from a directory (``config_dir`` with
    agents.json / tools.json / server.json, hot-reloaded) or from Python
    objects (``agents=``, ``tools=``, ``settings=``: same schemas as the
    files; change them later with :meth:`update_config`).

    Library extras: ``functions`` (Python callables registered as tools),
    ``llm`` (a LangChain chat model, or ``{provider_name: model}``) and
    ``checkpointer`` (any LangGraph checkpointer; it takes precedence over
    ``server.json`` ``persistence``).

    Use ``async with Runtime(...) as rt:`` (or ``await rt.startup()`` /
    ``await rt.shutdown()``), then ``await rt.chat(...)``.
    """

    def __init__(
        self,
        config_dir: str | Path | None = None,
        backend_transport: httpx.AsyncBaseTransport | None = None,
        *,
        agents: dict | None = None,
        tools: list | None = None,
        settings: dict | None = None,
        functions: list | tuple = (),
        llm: Any = None,
        checkpointer: Any = None,
    ):
        if config_dir is None and agents is None:
            raise ValueError("give either config_dir or agents=")
        if config_dir is not None and agents is not None:
            raise ValueError("config_dir and agents= are mutually exclusive")
        self.config_dir = Path(config_dir) if config_dir is not None else None
        self._data = {"agents": agents, "tools": tools, "settings": settings}
        self.functions = list(functions)
        self.backend_transport = backend_transport
        self.bundle: ConfigBundle | None = None
        self.llms = StaticLLMFactory(llm) if llm is not None else LLMFactory()
        self._own_checkpointer = checkpointer is None
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.persistence = "memory" if checkpointer is None else "custom"
        self.sqlite_file: Path | None = None
        self._sqlite_conn = None
        self.events = EventBus()
        self.shared_paths: dict[str, str] = {}
        self.sessions: dict[str, Session] = {}
        self.graph = None
        self.info: GraphInfo | None = None
        self.client: BackendClient | None = None
        self.listener: BackendEventListener | None = None
        self.registry: ToolRegistry | None = None
        self.watcher = ConfigWatcher(self.config_dir) if self.config_dir else None
        self.last_reload: float | None = None
        self.last_error: str | None = None
        self.started = time.time()
        self._watch_task: asyncio.Task | None = None
        self._maintenance_task: asyncio.Task | None = None

    @property
    def _base_config_dir(self) -> Path:
        """Directory used to resolve relative paths (plugins_dir, sqlite_path)."""
        return self.config_dir or Path.cwd() / "config"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def startup(self, raise_on_error: bool = False) -> None:
        try:
            await self._init_persistence()
            self.reload()
            await self._restore_pending()
        except ConfigError as exc:
            self.last_error = str(exc)
            if raise_on_error:
                raise
            log.error("invalid configuration at startup: %s", exc)
        if self.listener:
            self.listener.start()
        if self.watcher and self.bundle and self.bundle.settings.watch_interval_s > 0:
            self._watch_task = asyncio.create_task(self._watch_loop(), name="config-watcher")
        self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="maintenance")

    async def __aenter__(self) -> "Runtime":
        await self.startup(raise_on_error=True)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.shutdown()

    async def shutdown(self) -> None:
        for task in (self._watch_task, self._maintenance_task):
            if task:
                task.cancel()
        for sess in self.sessions.values():
            if sess.running:
                sess.cancel_reason = "server shutdown"
                sess.task.cancel()
        if self.listener:
            await self.listener.stop()
        if self.client:
            await self.client.aclose()
        if self._sqlite_conn is not None:
            with contextlib.suppress(Exception):
                await self._sqlite_conn.close()
            self._sqlite_conn = None

    async def _watch_loop(self) -> None:
        while True:
            interval = self.bundle.settings.watch_interval_s if self.bundle else 2.0
            await asyncio.sleep(interval or 2.0)
            if self.watcher.changed():
                log.info("configuration files changed — reloading")
                try:
                    self.reload()
                    self.events.publish(envelope("*", "config.reloaded", {"source": "watcher"}))
                except ConfigError as exc:
                    log.error("hot-reload failed, keeping previous configuration: %s", exc)
                    self.events.publish(envelope("*", "error", {"message": f"config reload failed: {exc}", "recoverable": True}))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _peek_settings(self) -> ServerSettings:
        """server.json before the full load (to open the right checkpointer)."""
        try:
            if self.config_dir is not None:
                return load_settings(self.config_dir / SERVER_FILE)
            return ServerSettings.model_validate(self._data["settings"] or {})
        except (ConfigError, ValidationError):
            return ServerSettings()  # the full load reports the error

    async def _init_persistence(self) -> None:
        if not self._own_checkpointer or self._sqlite_conn is not None:
            await self._restore_sessions()
            return
        settings = self._peek_settings()
        if settings.persistence == "sqlite":
            try:
                import aiosqlite
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            except ImportError as exc:
                raise ConfigError([f"{SERVER_FILE}: {SQLITE_HINT}"]) from exc
            path = settings.resolve_sqlite_path(self._base_config_dir)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                conn = await aiosqlite.connect(str(path))
                saver = AsyncSqliteSaver(conn)
                await saver.setup()
            except Exception as exc:  # noqa: BLE001 — unwritable path, corrupt file...
                raise ConfigError([f"{SERVER_FILE}: cannot open the SQLite database {path}: {exc}"]) from exc
            self.checkpointer, self._sqlite_conn = saver, conn
            self.persistence, self.sqlite_file = "sqlite", path
            log.info("conversation history persisted in %s", path)
        await self._restore_sessions()

    async def _restore_sessions(self) -> None:
        """Rebuild the session registry from the checkpointer (after a restart)."""
        spans: dict[str, tuple[float, float]] = {}
        async for item in self.checkpointer.alist(None):
            conf = item.config.get("configurable", {})
            tid = conf.get("thread_id")
            if not tid or conf.get("checkpoint_ns"):
                continue
            ts = _parse_ts(item.checkpoint.get("ts"))
            first, last = spans.get(tid, (ts, ts))
            spans[tid] = (min(first, ts), max(last, ts))
        for tid, (first, last) in spans.items():
            self.sessions.setdefault(tid, Session(id=tid, created=first, last_active=last, check_pending=True))
        if spans:
            log.info("restored %d session(s) from %s", len(spans), self.persistence)

    async def _restore_pending(self) -> None:
        """Restored sessions waiting for a confirmation: restart their timer."""
        if self.graph is None:
            return
        for sess in list(self.sessions.values()):
            if not sess.check_pending or sess.running:
                continue
            sess.check_pending = False
            if sess.pending_since is None:
                cfg = self._config(sess.id, 50)
                if await self.pending_confirmation(self.graph, cfg):
                    snap = await self.graph.aget_state(cfg)
                    sess.turn_start = _last_human_index((snap.values or {}).get("messages", []))
                    sess.pending_since = time.time()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def _load_bundle(self, data: dict | None = None) -> ConfigBundle:
        known = BUILTIN_TOOL_NAMES | {tooldef_from_function(f).name for f in self.functions}
        if data is None and self.config_dir is not None:
            return load_config(self.config_dir, known)
        data = data or self._data
        return config_from_data(data["agents"], data["tools"], data["settings"], known)

    def _make_registry(self, bundle: ConfigBundle, client: BackendClient, listener: BackendEventListener, paths: dict) -> ToolRegistry:
        s = bundle.settings
        return ToolRegistry(
            bundle.tools,
            client,
            listener,
            paths,
            s.max_tool_result_chars,
            allowed_modules=s.python_tool_modules,
            plugins_dir=s.resolve_plugins_dir(self._base_config_dir),
            functions=self.functions,
        )

    def reload(self) -> dict:
        """Re-read the configuration and rebuild the graph. Keeps the previous
        configuration if the new one is invalid (raises ConfigError)."""
        bundle = self._load_bundle()
        s = bundle.settings
        if self._own_checkpointer and self.last_reload is not None:
            wanted = (s.persistence, s.resolve_sqlite_path(self._base_config_dir) if s.persistence == "sqlite" else None)
            if wanted != (self.persistence, self.sqlite_file):
                bundle.warnings.append(f"{SERVER_FILE}: persistence settings changed — restart the server to apply them")
        if self.client is None or self.client.base_url != s.backend_url.rstrip("/"):
            old = self.client
            self.client = BackendClient(s.backend_url, s.backend_timeout_s, transport=self.backend_transport)
            if old and _loop_running():
                asyncio.ensure_future(old.aclose())
        if self.listener is None or self.listener.ws_url != s.backend_ws_url:
            if self.listener and _loop_running():
                asyncio.ensure_future(self.listener.stop())
            self.listener = BackendEventListener(s.backend_ws_url)
            if _loop_running() and self.last_reload is not None:
                self.listener.start()
        registry = self._make_registry(bundle, self.client, self.listener, self.shared_paths)
        previous = (self.bundle, self.registry)
        self.bundle, self.registry = bundle, registry
        try:
            self._rebuild_graph()
        except Exception:
            self.bundle, self.registry = previous
            raise
        if self.watcher:
            self.watcher.mark_seen()
        self.last_reload = time.time()
        return {"status": "reloaded", "agents": self.info.top_level if self.info else [], "warnings": bundle.warnings}

    def update_config(self, agents: dict | None = None, tools: list | None = None, settings: dict | None = None) -> dict:
        """In-memory configuration: replace any part and hot-reload.
        Keeps the previous configuration if the new one is invalid."""
        if self.config_dir is not None:
            raise RuntimeError("this runtime reads its configuration from files: edit them and call reload()")
        previous = dict(self._data)
        for key, value in (("agents", agents), ("tools", tools), ("settings", settings)):
            if value is not None:
                self._data[key] = value
        try:
            return self.reload()
        except ConfigError:
            self._data = previous
            raise

    def _current_data(self) -> dict:
        """The configuration as stored (files on disk, or in-memory objects)."""
        if self.config_dir is None:
            return dict(self._data)
        out: dict[str, Any] = {}
        for key, name, default in (("agents", AGENTS_FILE, None), ("tools", TOOLS_FILE, []), ("settings", SERVER_FILE, {})):
            path = self.config_dir / name
            if not path.exists():
                out[key] = default
                continue
            try:
                out[key] = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                raise ConfigError([f"{name}: invalid JSON at line {exc.lineno} col {exc.colno}: {exc.msg}"]) from exc
        return out

    def validate(self, agents: Any = None, tools: Any = None, settings: Any = None) -> dict:
        """Dry run of a configuration change: nothing is applied or written.

        Parts left to None are taken from the current configuration. The
        candidate goes through the same checks as a reload (schemas,
        cross-file rules, Python tool imports, providers, graph build), so
        ``valid`` means "a reload with this configuration will succeed".
        """
        try:
            current = self._current_data()
            data = {
                "agents": agents if agents is not None else current["agents"],
                "tools": tools if tools is not None else current["tools"],
                "settings": settings if settings is not None else current["settings"],
            }
            if data["agents"] is None:
                raise ConfigError([f"{AGENTS_FILE}: file not found"])
            bundle = self._load_bundle(data)
            s = bundle.settings
            registry = self._make_registry(bundle, BackendClient(s.backend_url), BackendEventListener(None), {})
            GraphBuilder(bundle, registry, self.llms).build()
        except ConfigError as exc:
            return {"valid": False, "errors": exc.errors, "warnings": []}
        except Exception as exc:  # noqa: BLE001 — any build failure makes the config invalid
            return {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"], "warnings": []}
        return {"valid": True, "errors": [], "warnings": bundle.warnings}

    def _rebuild_graph(self) -> None:
        if self.bundle is None or not self.llms.configured:
            self.graph, self.info = None, None
            return
        try:
            self.graph, self.info = build_graph(self.bundle, self.registry, self.llms, checkpointer=self.checkpointer)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            raise ConfigError([str(exc)]) from exc

    def set_provider(self, cfg: ProviderConfig) -> None:
        previous = self.llms
        self.llms = LLMFactory(cfg, previous.proxy)
        try:
            self._rebuild_graph()
        except ConfigError:
            self.llms = previous
            self._rebuild_graph()
            raise
        if _loop_running():  # the graph now exists: look for restored pending confirmations
            asyncio.ensure_future(self._restore_pending())

    def set_proxy(self, cfg: ProxyConfig) -> None:
        if isinstance(self.llms, StaticLLMFactory):
            self.llms.proxy = cfg  # caller-built models manage their own HTTP clients
            return
        self.llms = LLMFactory(self.llms.providers, cfg)
        self._rebuild_graph()

    def set_paths(self, paths: dict[str, str]) -> dict[str, str]:
        missing = [k for k, v in paths.items() if not Path(v).is_dir()]
        self.shared_paths.clear()
        self.shared_paths.update({k: str(Path(v).resolve()) for k, v in paths.items()})
        return {"paths": dict(self.shared_paths), "missing": missing}

    async def test_provider(self, provider: str | None = None, model: str | None = None) -> dict:
        t0 = time.perf_counter()
        try:
            name, entry = self.llms.provider_entry(provider)
            llm = await asyncio.to_thread(self.llms.get, name, model)
            resp = await llm.ainvoke([HumanMessage(content="Reply with the single word: pong")])
        except ProviderNotConfigured as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": provider, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "provider": name,
            "model": entry.resolve_model(model),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "reply": message_text(resp.content)[:200],
        }

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        agents = self.bundle.agents.agents if self.bundle else {}
        if self.graph is not None:
            state = "running"
        elif self.bundle is None:
            state = "error"
        else:
            state = "waiting_provider"
        return {
            "status": state,
            "version": __version__,
            "uptime_s": round(time.time() - self.started),
            "provider_configured": self.llms.configured,
            "agents_active": sum(1 for a in agents.values() if a.enabled),
            "agents_total": len(agents),
            "agents": [
                {
                    "name": n,
                    "display_name": a.display_name or n,
                    "enabled": a.enabled,
                    "type": a.type,
                    "routable": a.routable,
                    "tools": self.info.tool_counts.get(n) if self.info else None,
                }
                for n, a in agents.items()
            ],
            "tools": len(self.registry.tools) if self.registry else 0,
            "persistence": self.persistence,
            "sqlite_path": str(self.sqlite_file) if self.sqlite_file else None,
            "sessions": len(self.sessions),
            "running_turns": sum(1 for s in self.sessions.values() if s.running),
            "pending_confirmations": sum(1 for s in self.sessions.values() if s.pending_since is not None),
            "stream_subscribers": self.events.subscriber_count,
            "backend_events_connected": bool(self.listener and self.listener.connected),
            "last_reload": self.last_reload,
            "last_error": self.last_error,
            "warnings": self.bundle.warnings if self.bundle else [],
        }

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #
    def _require_graph(self):
        if self.bundle is None:
            raise NotReady("configuration is invalid — fix agents.json and call POST /reload")
        if not self.llms.configured:
            raise NotReady("no LLM provider configured — call POST /config/provider first")
        if self.graph is None:
            raise NotReady(self.last_error or "graph not built")
        return self.graph

    def _session(self, session_id: str | None) -> Session:
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        sess = self.sessions.get(sid)
        if sess is None:
            sess = self.sessions[sid] = Session(id=sid)
        return sess

    async def _existing_session(self, session_id: str) -> Session:
        """A known session, possibly only present in the checkpointer. Raises KeyError."""
        sess = self.sessions.get(session_id)
        if sess is not None:
            return sess
        item = await self.checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
        if item is None:
            raise KeyError(session_id)
        sess = self.sessions[session_id] = Session(id=session_id, last_active=_parse_ts(item.checkpoint.get("ts")))
        if self.graph is not None:
            snap = await self.graph.aget_state(self._config(session_id, 50))
            sess.turn_start = _last_human_index((snap.values or {}).get("messages", []))
        return sess

    @staticmethod
    def _config(sid: str, recursion_limit: int) -> dict:
        return {"configurable": {"thread_id": sid}, "recursion_limit": recursion_limit}

    async def pending_confirmation(self, graph, cfg: dict) -> dict | None:
        snap = await graph.aget_state(cfg)
        for task in snap.tasks:
            for intr in task.interrupts:
                value = intr.value if isinstance(intr.value, dict) else {"value": intr.value}
                return {**value, "interrupt_id": intr.id}
        return None

    async def history(self, session_id: str) -> list[dict]:
        graph = self._require_graph()
        await self._existing_session(session_id)
        snap = await graph.aget_state(self._config(session_id, 50))
        return [serialize_message(m) for m in (snap.values or {}).get("messages", [])]

    async def delete_session(self, session_id: str) -> None:
        sess = await self._existing_session(session_id)
        if sess.running:
            raise Conflict("a turn is running in this session — cancel it first")
        self.sessions.pop(session_id, None)
        await self.checkpointer.adelete_thread(session_id)

    # ------------------------------------------------------------------ #
    # Maintenance: session expiry, confirmation timeouts
    # ------------------------------------------------------------------ #
    async def _maintenance_loop(self) -> None:
        while True:
            interval = self.bundle.settings.maintenance_interval_s if self.bundle else 10.0
            await asyncio.sleep(interval)
            try:
                await self.maintenance()
            except Exception:  # noqa: BLE001 — never kill the loop
                log.exception("maintenance failed")

    async def maintenance(self, now: float | None = None) -> dict:
        """One maintenance pass (normally run by a background task)."""
        expired: list[str] = []
        timed_out: list[str] = []
        if self.bundle is None:
            return {"expired": expired, "confirmations_timed_out": timed_out}
        s = self.bundle.settings
        now = time.time() if now is None else now
        await self._restore_pending()

        if s.confirmation_timeout_s and self.graph is not None:
            for sess in list(self.sessions.values()):
                if sess.pending_since is None or sess.lock.locked():
                    continue
                if now - sess.pending_since < s.confirmation_timeout_s:
                    continue
                reason = f"no answer within {s.confirmation_timeout_s:.0f} s"
                self.events.publish(envelope(sess.id, "confirmation.expired", {"reason": reason}))
                try:
                    result = await self.confirm(sess.id, approved=False, reason=reason)
                    self.events.publish(envelope(sess.id, "chat.result", result))
                except Conflict:
                    sess.pending_since = None
                timed_out.append(sess.id)

        if s.session_ttl_hours:
            ttl = s.session_ttl_hours * 3600
            for sess in list(self.sessions.values()):
                if sess.running or sess.lock.locked() or now - sess.last_active < ttl:
                    continue
                await self.delete_session(sess.id)
                self.events.publish(envelope(sess.id, "session.expired", {"inactive_s": round(now - sess.last_active)}))
                expired.append(sess.id)
        return {"expired": expired, "confirmations_timed_out": timed_out}

    # ------------------------------------------------------------------ #
    # Turns
    # ------------------------------------------------------------------ #
    async def chat(self, message: str, session_id: str | None = None, agent: str | None = None) -> dict:
        graph = self._require_graph()
        if agent is not None and (self.info is None or agent not in self.info.top_level):
            raise ValueError(f"unknown or disabled agent '{agent}'")
        sess = self._session(session_id)
        if sess.running:
            raise Conflict("a turn is already running in this session — wait for it or cancel it")
        async with sess.lock:
            cfg = self._config(sess.id, self.bundle.settings.recursion_limit)
            if await self.pending_confirmation(graph, cfg):
                raise Conflict("a confirmation is pending for this session — answer it with POST /confirm (or POST /cancel)")
            snap = await graph.aget_state(cfg)
            values = snap.values or {}
            budget = self.bundle.settings.token_budget_per_session
            if budget and values.get("tokens_used", 0) >= budget:
                return self._result(sess, "budget_exceeded", values, [], [], None,
                                    response=f"The token budget of this session ({budget} tokens) is exhausted.")
            sess.turn_start = len(values.get("messages", []))
            sess.turns += 1
            return await self._execute(graph, sess, cfg, turn_input([HumanMessage(content=message)], direct=agent))

    async def confirm(self, session_id: str, approved: bool, reason: str | None = None) -> dict:
        graph = self._require_graph()
        sess = await self._existing_session(session_id)
        async with sess.lock:
            cfg = self._config(sess.id, self.bundle.settings.recursion_limit)
            if not await self.pending_confirmation(graph, cfg):
                sess.pending_since = None
                raise Conflict("no confirmation is pending for this session")
            sess.pending_since = None
            return await self._execute(graph, sess, cfg, Command(resume={"approved": approved, "reason": reason or ""}))

    async def cancel(self, session_id: str, reason: str | None = None) -> dict:
        """Cancel the running turn, or the pending confirmation, of a session.

        Returns once the session is usable again. Raises KeyError (unknown
        session) or Conflict (nothing to cancel)."""
        graph = self._require_graph()
        sess = await self._existing_session(session_id)
        reason = reason or "cancelled by the user"
        if sess.running:
            sess.cancel_reason = reason
            sess.task.cancel()
            async with sess.lock:  # the turn repairs the history, then releases the lock
                pass
            return {"session_id": sess.id, "status": "cancelled", "reason": reason}
        async with sess.lock:
            cfg = self._config(sess.id, self.bundle.settings.recursion_limit)
            if not await self.pending_confirmation(graph, cfg):
                raise Conflict("nothing to cancel in this session")
            await self._close_turn(graph, sess, cfg, reason)
        return {"session_id": sess.id, "status": "cancelled", "reason": reason}

    async def _execute(self, graph, sess: Session, cfg: dict, payload: Any) -> dict:
        """Run a turn as a cancellable task, with the turn timeout."""
        timeout = self.bundle.settings.turn_timeout_s or None
        sess.cancel_reason = None
        task = asyncio.create_task(self._run(graph, sess, cfg, payload), name=f"turn-{sess.id}")
        sess.task = task
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if not done:
                sess.cancel_reason = f"turn timeout ({timeout:g} s)"
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            elif not task.cancelled():
                return task.result()
        except asyncio.CancelledError:  # pragma: no cover — the caller itself was cancelled
            task.cancel()
            raise
        finally:
            sess.task = None
        return await self._close_turn(graph, sess, cfg, sess.cancel_reason or "cancelled")

    async def _close_turn(self, graph, sess: Session, cfg: dict, reason: str) -> dict:
        """End an interrupted turn: answer dangling tool calls, add a final
        message, clear any pending confirmation. The session stays usable."""
        snap = await graph.aget_state(cfg)
        messages = (snap.values or {}).get("messages", [])
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        patch: list[BaseMessage] = [
            ToolMessage(content=f"Cancelled: {reason}.", tool_call_id=tc["id"], name=tc["name"], status="error")
            for m in messages
            if isinstance(m, AIMessage)
            for tc in m.tool_calls
            if tc["id"] not in answered
        ]
        text = f"The turn was cancelled ({reason})."
        patch.append(AIMessage(content=text, name=SUPERVISOR))
        await graph.aupdate_state(cfg, {"messages": patch, "next": FINISH, "stopped": True}, as_node=SUPERVISOR)
        sess.pending_since = None
        sess.last_active = time.time()
        self.events.publish(envelope(sess.id, "turn.cancelled", {"reason": reason}))
        snap = await graph.aget_state(cfg)
        return self._result(sess, "cancelled", snap.values or {}, [], [], None, response=text)

    async def _run(self, graph, sess: Session, cfg: dict, payload: Any) -> dict:
        sess.last_active = time.time()
        names = self.info.node_agents if self.info else set()
        route: list[str] = []
        tool_calls: list[dict] = []
        publish = self.events.publish
        try:
            async for ev in graph.astream_events(payload, cfg, version="v2"):
                mapped = map_langgraph_event(ev, names)
                if mapped is None:
                    continue
                etype, data = mapped
                publish(envelope(sess.id, etype, data))
                if etype == "agent.start" and data["agent_name"] != SUPERVISOR:
                    if not route or route[-1] != data["agent_name"]:
                        route.append(data["agent_name"])
                elif etype == "tool.start":
                    tool_calls.append({"agent_name": data["agent_name"], "tool_name": data["tool_name"], "tool_args": data["tool_args"], "status": "running"})
                elif etype == "tool.end":
                    for tc in reversed(tool_calls):
                        if tc["tool_name"] == data["tool_name"] and tc["status"] == "running":
                            tc["status"], tc["result"] = data["status"], data.get("result")
                            break
        except Exception as exc:
            log.exception("turn failed for session %s", sess.id)
            publish(envelope(sess.id, "error", {"message": f"{type(exc).__name__}: {exc}", "recoverable": True}))
            raise
        finally:
            sess.last_active = time.time()

        snap = await graph.aget_state(cfg)
        values = snap.values or {}
        confirmation = await self.pending_confirmation(graph, cfg)
        if confirmation:
            sess.pending_since = time.time()
            publish(envelope(sess.id, "confirmation.needed", confirmation))
            return self._result(sess, "confirmation_needed", values, route, tool_calls, confirmation)
        sess.pending_since = None

        budget = self.bundle.settings.token_budget_per_session
        used = values.get("tokens_used", 0)
        if budget and not sess.budget_warned and used >= budget * self.bundle.settings.token_alert_ratio:
            sess.budget_warned = True
            publish(envelope(sess.id, "budget.warning", {"tokens_used": used, "token_budget": budget}))
        return self._result(sess, "completed", values, route, tool_calls, None)

    def _result(self, sess, status, values, route, tool_calls, confirmation, response=None) -> dict:
        new = list(values.get("messages", []))[sess.turn_start:]
        if response is None:
            final = next((m for m in reversed(new) if isinstance(m, AIMessage) and not m.tool_calls and message_text(m.content)), None)
            response = message_text(final.content) if final else ""
            agent = final.name if final else None
        else:
            agent = None
        return {
            "session_id": sess.id,
            "status": status,
            "response": response,
            "agent": agent,
            "route": route,
            "tool_calls": tool_calls,
            "confirmation": confirmation,
            "messages": [serialize_message(m) for m in new],
            "usage": {
                "tokens_used": values.get("tokens_used", 0),
                "token_budget": self.bundle.settings.token_budget_per_session if self.bundle else None,
            },
        }


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
