"""Persistence (memory / SQLite), session expiry, confirmation timeouts,
turn cancellation and timeouts, dry-run validation."""

import asyncio
import json
import sys

import pytest
from conftest import agent, call, fake_provider, route, write_config
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_json import FakeChatModel, Runtime, create_app
from langgraph_json.gateway.runtime import Conflict
from langgraph_json.utils.config_loader import ConfigError

SOLO = {"solo": agent("solo", ["open_item"])}


async def started(cfg, backend, *responses, **kw) -> Runtime:
    rt = Runtime(cfg, backend_transport=backend.transport, **kw)
    await rt.startup(raise_on_error=True)
    rt.set_provider(fake_provider(*responses))
    return rt


def llm_calls(rt):
    return rt.llms.get().calls


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
async def test_memory_is_the_default_and_writes_nothing(tmp_path, backend):
    cfg = write_config(tmp_path / "config", SOLO)
    rt = await started(cfg, backend, "one")
    await rt.chat("hello", "s")
    assert rt.status()["persistence"] == "memory" and rt.status()["sqlite_path"] is None
    await rt.shutdown()
    assert not (tmp_path / "data").exists()
    rt2 = await started(cfg, backend, "two")
    assert rt2.sessions == {}  # history lost on restart, by design
    await rt2.shutdown()


async def test_sqlite_history_survives_restart(tmp_path, backend):
    cfg = write_config(tmp_path / "config", SOLO, server={"persistence": "sqlite"})
    rt = await started(cfg, backend, "first answer")
    await rt.chat("remember 42", "persist")
    assert rt.status()["persistence"] == "sqlite"
    db = tmp_path / "data" / "sessions.db"
    assert rt.status()["sqlite_path"] == str(db.resolve()) and db.exists()
    await rt.shutdown()

    rt2 = await started(cfg, backend, "second answer")
    try:
        assert "persist" in rt2.sessions  # restored from the database
        r = await rt2.chat("what was it?", "persist")
        assert r["response"] == "second answer"
        seen = [m.content for m in llm_calls(rt2)[0]["messages"] if isinstance(m, HumanMessage)]
        assert seen == ["remember 42", "what was it?"]
        assert [m["type"] for m in r["messages"]] == ["human", "ai"]  # only the new turn
        hist = await rt2.history("persist")
        assert [m["content"] for m in hist if m["type"] == "human"] == ["remember 42", "what was it?"]
    finally:
        await rt2.shutdown()


async def test_sqlite_pending_confirmation_survives_restart(tmp_path, backend):
    agents = {"jobs": agent("jobs", ["run_job"], requires_confirmation=["run_job"])}
    cfg = write_config(tmp_path / "config", agents, server={"persistence": "sqlite"})
    rt = await started(cfg, backend, call("run_job", jobId="J1"))
    assert (await rt.chat("run J1", "s"))["status"] == "confirmation_needed"
    await rt.shutdown()

    rt2 = await started(cfg, backend, "J1 started")
    try:
        await rt2.maintenance()  # (also scheduled right after the provider is set)
        assert rt2.status()["pending_confirmations"] == 1  # timer restarted after the restart
        r = await rt2.confirm("s", approved=True)
        assert r["status"] == "completed" and r["response"] == "J1 started"
        assert [m["type"] for m in r["messages"]] == ["human", "ai", "tool", "ai"]  # the turn, not the whole history
        assert backend.calls() == [("POST", "/api/jobs/J1/run")]
    finally:
        await rt2.shutdown()


async def test_sqlite_custom_path_and_errors(tmp_path, backend, monkeypatch):
    db = tmp_path / "elsewhere" / "conv.sqlite"
    cfg = write_config(tmp_path / "config", SOLO, server={"persistence": "sqlite", "sqlite_path": str(db)})
    rt = await started(cfg, backend)
    assert rt.sqlite_file == db and db.exists()
    await rt.shutdown()

    (tmp_path / "isdir").mkdir()
    write_config(cfg, SOLO, server={"persistence": "sqlite", "sqlite_path": str(tmp_path / "isdir")})
    with pytest.raises(ConfigError, match="cannot open the SQLite database"):
        await Runtime(cfg).startup(raise_on_error=True)

    monkeypatch.setitem(sys.modules, "aiosqlite", None)  # simulate the missing optional dependency
    write_config(cfg, SOLO, server={"persistence": "sqlite"})
    rt = Runtime(cfg)
    await rt.startup()  # logged, not raised
    try:
        assert rt.status()["status"] == "error" and 'lang-graph-json[sqlite]' in rt.status()["last_error"]
    finally:
        await rt.shutdown()


async def test_changing_persistence_needs_restart(tmp_path, backend):
    cfg = write_config(tmp_path / "config", SOLO)
    rt = await started(cfg, backend)
    try:
        write_config(cfg, SOLO, server={"persistence": "sqlite"})
        r = rt.reload()
        assert any("restart the server" in w for w in r["warnings"])
        assert rt.persistence == "memory"
    finally:
        await rt.shutdown()


async def test_injected_checkpointer_wins(tmp_path, backend):
    cfg = write_config(tmp_path / "config", SOLO, server={"persistence": "sqlite"})
    saver = InMemorySaver()
    rt = await started(cfg, backend, "x", checkpointer=saver)
    try:
        assert rt.persistence == "custom" and rt.checkpointer is saver
        assert not (tmp_path / "data").exists()
    finally:
        await rt.shutdown()


async def test_session_known_only_by_checkpointer(make_runtime):
    rt = await make_runtime(agents=SOLO, responses=["a"])
    await rt.chat("x", "s")
    rt.sessions.clear()  # e.g. created by another runtime sharing the checkpointer
    assert len(await rt.history("s")) == 2
    assert "s" in rt.sessions
    with pytest.raises(KeyError):
        await rt.history("never-seen")
    await rt.delete_session("s")
    with pytest.raises(KeyError):
        await rt.history("s")


# --------------------------------------------------------------------------- #
# Session expiry
# --------------------------------------------------------------------------- #
async def test_inactive_sessions_expire(make_runtime):
    rt = await make_runtime(agents=SOLO, server={"session_ttl_hours": 1}, responses=["a", "b"])
    await rt.chat("x", "old")
    await rt.chat("y", "fresh")
    rt.sessions["old"].last_active -= 2 * 3600
    q = rt.events.subscribe(None)
    r = await rt.maintenance()
    assert r == {"expired": ["old"], "confirmations_timed_out": []}
    assert set(rt.sessions) == {"fresh"}
    with pytest.raises(KeyError):
        await rt.history("old")  # deleted from the checkpointer too
    assert q.get_nowait()["event_type"] == "session.expired"


async def test_sqlite_sessions_expire_on_disk(tmp_path, backend):
    cfg = write_config(tmp_path / "config", SOLO, server={"persistence": "sqlite", "session_ttl_hours": 1})
    rt = await started(cfg, backend, "a")
    await rt.chat("x", "s")
    await rt.maintenance(now=rt.sessions["s"].last_active + 3601)
    await rt.shutdown()
    rt2 = await started(cfg, backend)
    try:
        assert rt2.sessions == {}
    finally:
        await rt2.shutdown()


async def test_ttl_zero_never_expires(make_runtime):
    rt = await make_runtime(agents=SOLO, server={"session_ttl_hours": 0}, responses=["a"])
    await rt.chat("x", "s")
    assert (await rt.maintenance(now=rt.sessions["s"].last_active + 10**9))["expired"] == []


async def test_maintenance_without_config(tmp_path):
    cfg = write_config(tmp_path / "config")
    (cfg / "agents.json").write_text("{broken")
    rt = Runtime(cfg)
    assert await rt.maintenance() == {"expired": [], "confirmations_timed_out": []}


async def test_maintenance_loop_runs(tmp_path, backend):
    cfg = write_config(tmp_path / "config", SOLO, server={"maintenance_interval_s": 0.05, "session_ttl_hours": 0.00001})
    rt = await started(cfg, backend, "a")
    try:
        await rt.chat("x", "s")
        for _ in range(100):
            if not rt.sessions:
                break
            await asyncio.sleep(0.05)
        assert rt.sessions == {}
    finally:
        await rt.shutdown()


# --------------------------------------------------------------------------- #
# Confirmation timeout
# --------------------------------------------------------------------------- #
async def test_unanswered_confirmation_is_denied(make_runtime, backend):
    rt = await make_runtime(server={"confirmation_timeout_s": 60},
                            responses=[route("jobs"), call("run_job", jobId="J"), "OK, not run.", route("FINISH")])
    assert (await rt.chat("run", "s"))["status"] == "confirmation_needed"
    assert rt.status()["pending_confirmations"] == 1
    q = rt.events.subscribe("s")
    assert (await rt.maintenance(now=rt.sessions["s"].pending_since + 30))["confirmations_timed_out"] == []
    r = await rt.maintenance(now=rt.sessions["s"].pending_since + 61)
    assert r["confirmations_timed_out"] == ["s"]
    assert backend.calls() == []
    assert rt.status()["pending_confirmations"] == 0
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    types = [e["event_type"] for e in events]
    assert "confirmation.expired" in types
    result = next(e for e in events if e["event_type"] == "chat.result")["data"]
    tool = next(m for m in result["messages"] if m["type"] == "tool")
    assert "no answer within 60 s" in tool["content"]
    assert (await rt.chat("next", "s"))["status"] == "completed"  # session usable again


async def test_confirmation_timeout_zero_waits_forever(make_runtime):
    rt = await make_runtime(server={"confirmation_timeout_s": 0, "session_ttl_hours": 0}, responses=[route("jobs"), call("run_job", jobId="J")])
    await rt.chat("run", "s")
    assert (await rt.maintenance(now=10**12))["confirmations_timed_out"] == []
    assert rt.sessions["s"].pending_since is not None


async def test_stale_pending_marker_is_cleared(make_runtime):
    rt = await make_runtime(server={"confirmation_timeout_s": 1}, agents=SOLO, responses=["a"])
    await rt.chat("x", "s")
    rt.sessions["s"].pending_since = 0  # marker without a real interrupt
    r = await rt.maintenance(now=100)
    assert r["confirmations_timed_out"] == ["s"] and rt.sessions["s"].pending_since is None


# --------------------------------------------------------------------------- #
# Cancellation and turn timeout
# --------------------------------------------------------------------------- #
STARTED = asyncio.Event


async def slow_tool(seconds: float = 30) -> str:
    """Waits (a long computation)."""
    await asyncio.sleep(seconds)
    return "finished"


def lib_runtime(responses, **settings):
    return Runtime(
        agents={"agents": {"worker": agent("worker", ["slow_tool"])}},
        functions=[slow_tool],
        llm=FakeChatModel(responses=list(responses)),
        settings={"watch_interval_s": 0, **settings},
    )


async def wait_tool_started(rt, sid):
    q = rt.events.subscribe(sid)
    while (await asyncio.wait_for(q.get(), 5))["event_type"] != "tool.start":
        pass


async def test_cancel_running_turn_and_continue():
    llm = FakeChatModel(responses=[call("slow_tool"), "after cancel"])
    rt = Runtime(agents={"agents": {"worker": agent("worker", ["slow_tool"])}}, functions=[slow_tool], llm=llm)
    async with rt:
        turn = asyncio.create_task(rt.chat("work", "s"))
        await wait_tool_started(rt, "s")
        assert rt.status()["running_turns"] == 1
        with pytest.raises(Conflict, match="already running"):
            await rt.chat("again", "s")
        with pytest.raises(Conflict, match="cancel it first"):
            await rt.delete_session("s")
        c = await rt.cancel("s", "user pressed stop")
        assert c == {"session_id": "s", "status": "cancelled", "reason": "user pressed stop"}
        r = await turn
        assert r["status"] == "cancelled" and "user pressed stop" in r["response"]
        tool = next(m for m in r["messages"] if m["type"] == "tool")
        assert tool["status"] == "error" and tool["content"].startswith("Cancelled")
        assert rt.status()["running_turns"] == 0

        r2 = await rt.chat("continue", "s")
        assert r2["status"] == "completed" and r2["response"] == "after cancel"
        # the history sent to the LLM is valid: every tool call has its result
        msgs = llm.calls[-1]["messages"]
        ids = {tc["id"] for m in msgs for tc in getattr(m, "tool_calls", [])}
        assert ids and ids <= {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}


async def test_turn_timeout():
    async with lib_runtime([call("slow_tool"), "next"], turn_timeout_s=0.2) as rt:
        r = await rt.chat("work", "s")
        assert r["status"] == "cancelled" and "turn timeout (0.2 s)" in r["response"]
        assert (await rt.chat("again", "s"))["status"] == "completed"


async def test_turn_timeout_zero_means_no_limit():
    async with lib_runtime([call("slow_tool", seconds=0.3), "done"], turn_timeout_s=0) as rt:
        assert (await rt.chat("work", "s"))["response"] == "done"


async def test_cancel_pending_confirmation(make_runtime, backend):
    rt = await make_runtime(responses=[route("jobs"), call("run_job", jobId="J"), route("FINISH"), "fresh start"])
    await rt.chat("run", "s")
    r = await rt.cancel("s")
    assert r["status"] == "cancelled" and r["reason"] == "cancelled by the user"
    assert backend.calls() == [] and rt.status()["pending_confirmations"] == 0
    with pytest.raises(Conflict, match="no confirmation"):
        await rt.confirm("s", True)
    hist = await rt.history("s")
    assert hist[-1]["content"].startswith("The turn was cancelled")
    assert any(m["type"] == "tool" and m["content"].startswith("Cancelled") for m in hist)


async def test_cancel_errors(make_runtime):
    rt = await make_runtime(agents=SOLO, responses=["a"])
    with pytest.raises(KeyError):
        await rt.cancel("unknown")
    await rt.chat("x", "s")
    with pytest.raises(Conflict, match="nothing to cancel"):
        await rt.cancel("s")


async def test_shutdown_cancels_running_turns():
    rt = lib_runtime([call("slow_tool")])
    await rt.startup(raise_on_error=True)
    turn = asyncio.create_task(rt.chat("work", "s"))
    await wait_tool_started(rt, "s")
    await rt.shutdown()
    r = await turn
    assert r["status"] == "cancelled" and "server shutdown" in r["response"]


def test_cancel_over_http_and_websocket(tmp_path, backend):
    cfg = write_config(tmp_path / "config")
    rt = Runtime(cfg, backend_transport=backend.transport)
    with TestClient(create_app(rt)) as c:
        rt.set_provider(fake_provider(route("jobs"), call("run_job", jobId="J"), route("jobs"), call("run_job", jobId="K")))
        assert c.post("/cancel", json={"session_id": "nope"}).status_code == 404
        c.post("/chat", json={"message": "run", "session_id": "s"})
        r = c.post("/cancel", json={"session_id": "s", "reason": "changed my mind"})
        assert r.status_code == 200 and r.json()["reason"] == "changed my mind"
        assert c.post("/cancel", json={"session_id": "s"}).status_code == 409

        c.post("/chat", json={"message": "run again", "session_id": "s"})
        with c.websocket_connect("/stream?session_id=s") as ws:
            ws.receive_json()
            ws.send_json({"type": "cancel"})
            while (ev := ws.receive_json())["event_type"] != "turn.cancelled":
                pass
            ws.send_json({"type": "cancel"})  # nothing left to cancel
            while (ev := ws.receive_json())["event_type"] != "error":
                pass
            assert "cancel failed" in ev["data"]["message"]
    assert backend.calls() == []


# --------------------------------------------------------------------------- #
# Dry-run validation
# --------------------------------------------------------------------------- #
def test_validate_current_and_candidate_configs(tmp_path, backend):
    cfg = write_config(tmp_path / "config")
    rt = Runtime(cfg, backend_transport=backend.transport)
    with TestClient(create_app(rt)) as c:
        assert c.post("/config/validate", json={}).json() == {"valid": True, "errors": [], "warnings": []}

        before = (cfg / "agents.json").read_text()
        bad = {"agents": {"x": {"description": "", "system_prompt": "p", "allowed_tools": []}}}
        r = c.post("/agents/validate", json=bad).json()
        assert r["valid"] is False and any("at least one tool" in e for e in r["errors"])
        assert (cfg / "agents.json").read_text() == before  # nothing written
        assert "x" not in rt.bundle.agents.agents  # nothing applied

        ok = {"agents": {"new": agent("new", ["open_item", "ghost"])}}
        r = c.post("/agents/validate", json=ok).json()
        assert r["valid"] is True and any("unknown tool 'ghost'" in w for w in r["warnings"])

        r = c.post("/config/validate", json={"tools": [{"name": "t", "method": "FETCH", "path": "/x", "description": "d"}]}).json()
        assert r["valid"] is False and any("invalid HTTP method" in e for e in r["errors"])
        r = c.post("/config/validate", json={"settings": {"port": 0}}).json()
        assert r["valid"] is False and any("server.json: port" in e for e in r["errors"])


def test_validate_does_the_full_dry_build(tmp_path, backend):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "brokenmod_vx.py").write_text("def f(:\n")
    cfg = write_config(tmp_path / "config")
    rt = Runtime(cfg, backend_transport=backend.transport)
    rt.reload()
    tools = [{"name": "f", "kind": "python", "callable": "brokenmod_vx:f", "description": "d"}]
    r = rt.validate(tools=tools, settings={"python_tool_modules": ["brokenmod_vx"], "plugins_dir": "plugins"})
    assert r["valid"] is False and any("cannot import" in e for e in r["errors"])
    r = rt.validate(tools=tools)
    assert r["valid"] is False and any("not allowed" in e for e in r["errors"])

    rt.set_provider(fake_provider())
    r = rt.validate(agents={"agents": {"a": agent("a", ["open_item"], provider="missing")}})
    assert r["valid"] is False and any("unknown providers" in e for e in r["errors"])


def test_validate_reads_parts_from_disk(tmp_path):
    cfg = write_config(tmp_path / "config")
    rt = Runtime(cfg)
    (cfg / "tools.json").write_text("{broken")
    r = rt.validate(agents={"agents": {"a": agent("a", ["open_item"])}})
    assert r["valid"] is False and "tools.json: invalid JSON" in r["errors"][0]
    (cfg / "tools.json").unlink()
    (cfg / "agents.json").unlink()
    assert rt.validate()["errors"] == ["agents.json: file not found"]
    assert rt.validate(agents={"agents": {"a": agent("a", ["wait_for_event"])}})["valid"] is True


async def test_validate_in_memory_runtime():
    rt = Runtime(agents={"agents": {"solo": agent("solo", ["slow_tool"])}}, functions=[slow_tool], llm=FakeChatModel())
    assert rt.validate()["valid"] is True
    assert rt.validate(agents={"agents": {"solo": agent("solo", [])}})["valid"] is False
    assert rt._data["agents"]["agents"]["solo"]["allowed_tools"] == ["slow_tool"]  # unchanged


def test_shipped_server_json_documents_the_new_settings():
    from conftest import ROOT

    s = json.loads((ROOT / "config" / "server.json").read_text())
    assert s["persistence"] == "memory" and s["session_ttl_hours"] == 24
    assert s["turn_timeout_s"] == 600 and s["confirmation_timeout_s"] == 900


# --------------------------------------------------------------------------- #
# Remaining branches
# --------------------------------------------------------------------------- #
async def test_pending_restored_as_soon_as_provider_is_pushed(tmp_path, backend):
    agents = {"jobs": agent("jobs", ["run_job"], requires_confirmation=["run_job"])}
    cfg = write_config(tmp_path / "config", agents, server={"persistence": "sqlite"})
    rt = await started(cfg, backend, call("run_job", jobId="J1"))
    await rt.chat("run", "s")
    await rt.shutdown()
    rt2 = Runtime(cfg, backend_transport=backend.transport)
    await rt2.startup(raise_on_error=True)  # no provider yet: the graph does not exist
    try:
        assert rt2.graph is None and rt2.status()["pending_confirmations"] == 0
        rt2.set_provider(fake_provider("ok"))  # what the host application does after startup
        for _ in range(50):
            if rt2.status()["pending_confirmations"]:
                break
            await asyncio.sleep(0.02)
        assert rt2.status()["pending_confirmations"] == 1
    finally:
        await rt2.shutdown()


async def test_restore_ignores_sub_graph_checkpoints(make_runtime):
    agents = {
        "team": {"type": "sub_graph", "description": "team", "system_prompt": "p", "sub_agents": ["w"]},
        "w": agent("w", ["open_item"], routable=False),
    }
    rt = await make_runtime(agents=agents, responses=["w answer"])
    await rt.chat("x", "top")
    namespaces = {c.config["configurable"].get("checkpoint_ns") async for c in rt.checkpointer.alist(None)}
    assert any(namespaces - {""})  # the sub-graph wrote its own checkpoints
    rt.sessions.clear()
    await rt._restore_sessions()
    assert set(rt.sessions) == {"top"}


def test_helpers_tolerate_bad_input(tmp_path):
    from langgraph_json.gateway.runtime import _parse_ts

    assert abs(_parse_ts("not a date") - __import__("time").time()) < 5
    assert _parse_ts("2026-01-01T00:00:00+00:00") == 1767225600.0
    cfg = write_config(tmp_path / "config", server={"port": "bad"})
    assert Runtime(cfg)._peek_settings().persistence == "memory"


async def test_maintenance_loop_survives_errors(tmp_path, backend, caplog):
    cfg = write_config(tmp_path / "config", SOLO, server={"maintenance_interval_s": 0.02})
    rt = Runtime(cfg, backend_transport=backend.transport)
    calls = []

    async def boom(now=None):
        calls.append(1)
        raise RuntimeError("maintenance bug")

    rt.maintenance = boom
    await rt.startup()
    try:
        for _ in range(100):
            if len(calls) >= 3:
                break
            await asyncio.sleep(0.02)
        assert len(calls) >= 3 and "maintenance failed" in caplog.text
    finally:
        await rt.shutdown()
