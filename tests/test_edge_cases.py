"""Edge cases and error paths: hot-reload watcher, LangGraph Studio entry
point, guards inside nodes, backend client, provider builders, events."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from conftest import ROOT, TOOLS, agent, call, fake_provider, route, write_config
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_json.gateway import Runtime, create_app
from langgraph_json.utils.backend_client import BackendClient, BackendError, BackendEventListener
from langgraph_json.utils.config_loader import AgentsConfig, ConfigWatcher, ToolDef, validate_semantics
from langgraph_json.utils.events import EventBus, _jsonable, envelope, map_langgraph_event
from langgraph_json.utils.fake_llm import FakeChatModel, _text
from langgraph_json.utils.llm_factory import LLMFactory, ProviderConfig, ProviderNotConfigured, ProxyConfig
from langgraph_json.utils.nodes import _normalise_decision, answered_this_turn, make_handoff_tool, make_route_tool
from langgraph_json.utils.state import turn_input


# --------------------------------------------------------------------------- #
# Hot-reload watcher (NFR-09)
# --------------------------------------------------------------------------- #
async def wait_event(queue: asyncio.Queue, event_type: str, timeout: float = 10) -> dict:
    async def _get():
        while True:
            ev = await queue.get()
            if ev["event_type"] == event_type:
                return ev

    return await asyncio.wait_for(_get(), timeout)


async def test_watcher_reloads_changed_files(tmp_path, backend):
    cfg = write_config(tmp_path / "cfg", server={"watch_interval_s": 0.05})
    rt = Runtime(cfg, backend_transport=backend.transport)
    await rt.startup()
    try:
        q = rt.events.subscribe(None)
        data = json.loads((cfg / "agents.json").read_text())
        data["agents"]["added"] = agent("added agent", ["open_item"])
        (cfg / "agents.json").write_text(json.dumps(data))
        await wait_event(q, "config.reloaded")
        assert "added" in rt.bundle.agents.agents

        (cfg / "agents.json").write_text("{broken json")
        ev = await wait_event(q, "error")
        assert "config reload failed" in ev["data"]["message"]
        assert "added" in rt.bundle.agents.agents  # previous configuration kept
    finally:
        await rt.shutdown()


async def test_reload_switches_backend_and_events(tmp_path, backend):
    cfg = write_config(tmp_path / "cfg")
    rt = Runtime(cfg, backend_transport=backend.transport)
    await rt.startup()
    try:
        first_client, first_listener = rt.client, rt.listener
        write_config(cfg, server={"backend_url": "http://other:1234", "backend_ws_url": "ws://127.0.0.1:9/ws"})
        rt.reload()
        assert rt.client is not first_client and rt.client.base_url == "http://other:1234"
        assert rt.listener is not first_listener and rt.listener.ws_url == "ws://127.0.0.1:9/ws"
        await asyncio.sleep(0.05)
        assert rt.listener._task is not None  # started
    finally:
        await rt.shutdown()


def test_reload_outside_event_loop(tmp_path):
    cfg = write_config(tmp_path / "cfg")
    rt = Runtime(cfg)
    rt.reload()
    write_config(cfg, server={"backend_url": "http://x:1", "backend_ws_url": "ws://x:2"})
    assert rt.reload()["status"] == "reloaded"
    assert rt.client.base_url == "http://x:1"


def test_watcher_handles_missing_files(tmp_path):
    cfg = write_config(tmp_path / "cfg")
    w = ConfigWatcher(cfg)
    (cfg / "server.json").unlink()
    assert w.changed()
    assert ("server.json", None, None) in w.snapshot()


# --------------------------------------------------------------------------- #
# LangGraph CLI / Studio entry point
# --------------------------------------------------------------------------- #
def test_langgraph_json_points_to_factory():
    cfg = json.loads((ROOT / "langgraph.json").read_text())
    module, _, attr = cfg["graphs"]["agents"].partition(":")
    assert module == "./langgraph_json/agent.py" and attr == "make_graph"


def test_provider_from_env(monkeypatch):
    from langgraph_json.agent import _provider_from_env

    for k in ("LGJ_PROVIDER_JSON", "ANTHROPIC_API_KEY", "LGJ_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert _provider_from_env().providers["default"].type == "fake"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    p = _provider_from_env().providers["default"]
    assert p.type == "anthropic" and p.default_model == "claude-sonnet-5-5"
    monkeypatch.setenv("LGJ_PROVIDER_JSON", json.dumps({"type": "ollama", "default_model": "llama3"}))
    assert _provider_from_env().providers["default"].type == "ollama"


async def test_make_graph_runs(monkeypatch):
    from langgraph_json.agent import make_graph

    for k in ("LGJ_PROVIDER_JSON", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LGJ_CONFIG_DIR", str(ROOT / "config"))
    graph = make_graph()
    out = await graph.ainvoke(turn_input([HumanMessage(content="hello studio")]))
    last = out["messages"][-1]
    assert last.name == "supervisor" and last.content.startswith("echo:") and "User: hello studio" in last.content
    assert {"supervisor", "hmi", "hmi__tools", "analysis", "jobs"} <= set(graph.get_graph().nodes)


# --------------------------------------------------------------------------- #
# Node guards
# --------------------------------------------------------------------------- #
async def test_max_steps_message_when_no_answer(make_runtime):
    rt = await make_runtime(responses=[route("hmi"), ""], supervisor={"max_steps": 1})
    r = await rt.chat("x")
    assert "maximum number of routing steps" in r["response"]


async def test_no_routable_agent(make_runtime):
    rt = await make_runtime(agents={"hidden": agent("hidden", ["open_item"], routable=False)}, responses=[])
    assert any("no enabled routable agent" in w for w in rt.bundle.warnings)
    r = await rt.chat("x")
    assert "No agent is enabled" in r["response"]


async def test_budget_exhausted_in_supervisor_mid_turn(make_runtime):
    rt = await make_runtime(server={"token_budget_per_session": 400}, responses=[route("hmi"), "word " * 500])
    r = await rt.chat("x")
    assert r["response"].startswith("The token budget")
    assert r["agent"] == "supervisor"


async def test_budget_exhausted_in_agent(make_runtime):
    rt = await make_runtime(server={"token_budget_per_session": 3}, responses=[route("hmi")])
    r = await rt.chat("a fairly long question to consume the tiny budget")
    assert r["agent"] == "hmi" and r["response"].startswith("The token budget")


def test_node_helpers():
    assert _normalise_decision(True) == (True, "")
    assert _normalise_decision({"approved": 1, "reason": "ok"}) == (True, "ok")
    assert _normalise_decision(" Yes ") == (True, "")
    assert _normalise_decision("no") == (False, "")
    assert _normalise_decision(None) == (False, "")
    assert not answered_this_turn([])
    assert not answered_this_turn([AIMessage(content="x")]) is False  # an answer without user message
    assert make_route_tool({"a": None}).invoke({"next": "a"}) == "routed"
    assert make_handoff_tool("b", "desc").invoke({"reason": "r"}) == "Handing off to b"


# --------------------------------------------------------------------------- #
# Gateway error paths
# --------------------------------------------------------------------------- #
def test_provider_without_model_is_503(tmp_path, backend):
    cfg = write_config(tmp_path / "cfg")
    with TestClient(create_app(Runtime(cfg, backend_transport=backend.transport))) as c:
        c.post("/config/provider", json={"type": "ollama"})
        r = c.post("/chat", json={"message": "x"})
        assert r.status_code == 503 and "no model" in r.json()["error"]


def test_confirm_failure_is_502(tmp_path, backend):
    def explode(messages, tools):
        raise RuntimeError("model crashed")

    cfg = write_config(tmp_path / "cfg")
    rt = Runtime(cfg, backend_transport=backend.transport)
    with TestClient(create_app(rt)) as c:
        rt.set_provider(fake_provider(route("jobs"), call("run_job", jobId="J"), explode))
        assert c.post("/chat", json={"message": "x", "session_id": "s"}).json()["status"] == "confirmation_needed"
        r = c.post("/confirm", json={"session_id": "s", "approved": True})
        assert r.status_code == 502 and "model crashed" in r.json()["error"]


async def test_test_provider_reports_errors(make_runtime):
    def explode(messages, tools):
        raise RuntimeError("unreachable")

    rt = await make_runtime(responses=[explode])
    r = await rt.test_provider()
    assert r["ok"] is False and "unreachable" in r["error"]


async def test_graph_missing_reports_last_error(make_runtime):
    rt = await make_runtime(responses=[])
    rt.graph, rt.last_error = None, "boom"
    from langgraph_json.gateway.runtime import NotReady

    with pytest.raises(NotReady, match="boom"):
        await rt.chat("x")


# --------------------------------------------------------------------------- #
# Backend client
# --------------------------------------------------------------------------- #
async def test_backend_client_responses_and_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/empty":
            return httpx.Response(204)
        if request.url.path == "/text":
            return httpx.Response(200, text="plain text")
        if request.url.path == "/slow":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"q": dict(request.url.params)})

    c = BackendClient("http://b/", transport=httpx.MockTransport(handler))
    assert await c.request("POST", "/empty", body={}) == {"status": "ok"}
    assert await c.request("GET", "/text") == "plain text"
    assert (await c.request("GET", "/json", params={"f": {"a": 1}, "n": 2}))["q"] == {"f": '{"a": 1}', "n": "2"}
    with pytest.raises(BackendError, match="timeout"):
        await c.request("GET", "/slow")
    assert await c.ping("/text") is True
    await c.aclose()
    await c.aclose()

    def down(request):
        raise httpx.ConnectError("refused")

    assert await BackendClient("http://b", transport=httpx.MockTransport(down)).ping() is False


async def test_listener_retries_when_backend_is_down():
    listener = BackendEventListener("ws://127.0.0.1:9/unreachable")
    listener.start()
    listener.start()  # idempotent
    await asyncio.sleep(0.3)
    assert listener.connected is False and listener._task is not None and not listener._task.done()
    await listener.stop()
    assert listener._task is None


# --------------------------------------------------------------------------- #
# Config details
# --------------------------------------------------------------------------- #
def test_config_details(tmp_path):
    with pytest.raises(Exception, match="path must start"):
        ToolDef.model_validate({"name": "a", "method": "GET", "path": "a", "description": "d"})
    assert ToolDef.model_validate({"name": "p", "kind": "python", "callable": "m:f", "description": "d", "method": None}).method is None
    with pytest.raises(Exception, match="needs a 'name'"):
        AgentsConfig.model_validate({"agents": [{"description": "x"}]})
    cfg = AgentsConfig.model_validate({"agents": {
        "loop": {"type": "sub_graph", "description": "d", "system_prompt": "p", "sub_agents": ["loop"]},
        "a": agent("a", ["t"], can_handoff_to=["off"]),
        "off": {**agent("off", ["t"]), "enabled": False},
    }})
    errors, warnings = validate_semantics(cfg, {"t"})
    assert any("its own sub-agent" in e for e in errors)
    assert any("'off' is disabled" in w for w in warnings)
    from langgraph_json.utils.config_loader import load_config

    bundle = load_config(write_config(tmp_path / "c"))
    assert set(bundle.tools_by_name) == {t["name"] for t in TOOLS}


async def test_list_shared_files_missing_folder(make_runtime, tmp_path):
    rt = await make_runtime(responses=[])
    rt.set_paths({"gone": str(tmp_path / "gone")})
    from langgraph_json.utils.tools import ToolExecutionError

    with pytest.raises(ToolExecutionError, match="does not exist"):
        await rt.registry.tools["list_shared_files"].ainvoke({"folder": "gone"})


# --------------------------------------------------------------------------- #
# Tools details
# --------------------------------------------------------------------------- #
async def test_tool_details(backend):
    from langgraph_json.utils.config_loader import ConfigError
    from langgraph_json.utils.tools import ToolExecutionError, ToolRegistry, make_rest_tool, schema_from_signature

    def f(x: "UndefinedType", y=2):  # noqa: F821 — unresolvable annotation
        return x

    schema = schema_from_signature("f", f).model_json_schema()
    assert schema["required"] == ["x"]

    t = ToolDef.model_validate({**TOOLS[1], "returns": {"state": "string"}})
    tool = make_rest_tool(t, BackendClient("http://b", transport=backend.transport), 1000)
    assert "Returns:" in tool.description
    with pytest.raises(ToolExecutionError, match="missing path parameter"):
        await tool.coroutine(verbose=True)

    with pytest.raises(ConfigError, match="duplicate tool name 'wait_for_event'"):
        ToolRegistry([ToolDef.model_validate({**TOOLS[0], "name": "wait_for_event"})], BackendClient("http://b"))


# --------------------------------------------------------------------------- #
# Provider builders
# --------------------------------------------------------------------------- #
def test_builder_options():
    cfg = ProviderConfig.model_validate({
        "default": "a",
        "providers": {
            "a": {"type": "anthropic", "auth": "bearer", "api_key": "tok", "default_model": "m", "temperature": 0.2, "max_tokens": 99},
            "az": {"type": "azure_openai", "api_key": "k", "endpoint": "https://r.openai.azure.com", "default_model": "d"},
            "ol": {"type": "ollama", "endpoint": "https://remote-ollama.example.com", "default_model": "l",
                   "headers": {"X-Key": "v"}, "temperature": 0.1, "max_tokens": 50},
        },
    })
    f = LLMFactory(cfg, ProxyConfig(https_proxy="http://proxy:3128"))
    a = f.get()
    assert a.default_headers["Authorization"] == "Bearer tok" and a.temperature == 0.2 and a.max_tokens == 99
    assert f.get("az").openai_api_key.get_secret_value() == "k"
    ol = f.get("ol")
    assert ol.client_kwargs["proxy"] == "http://proxy:3128" and ol.client_kwargs["headers"] == {"X-Key": "v"}
    assert ol.temperature == 0.1 and ol.num_predict == 50
    assert LLMFactory().available_models() == {}


def test_unsupported_provider_type(monkeypatch):
    monkeypatch.setattr(LLMFactory, "_builders", {})
    with pytest.raises(ProviderNotConfigured, match="unsupported provider type"):
        LLMFactory(ProviderConfig.model_validate({"type": "fake"})).get()


async def test_azure_token_through_proxy(monkeypatch):
    from langgraph_json.utils.llm_factory import AzureAdConfig, AzureAdTokenProvider

    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, data):
            return httpx.Response(200, json={"access_token": "T", "expires_in": 3600})

    monkeypatch.setattr(httpx, "Client", FakeClient)
    tp = AzureAdTokenProvider(AzureAdConfig(tenant_id="t", client_id="c", client_secret="s", scope="x"), ProxyConfig(https_proxy="http://p:1"))
    assert await tp.aget_token() == "T"
    assert seen["proxy"] == "http://p:1"


# --------------------------------------------------------------------------- #
# Events and fake model
# --------------------------------------------------------------------------- #
def test_jsonable_and_long_tool_results():
    assert _jsonable([1, (2, 3)]) == [1, [2, 3]]
    assert _jsonable(AIMessage(content="hi")) == "hi"
    assert _jsonable(Path("x")) == "x"
    ev = {"event": "on_tool_end", "name": "t", "metadata": {"langgraph_node": "a__tools"}, "data": {"output": "x" * 5000}}
    etype, data = map_langgraph_event(ev, {"a"})
    assert etype == "tool.end" and data["agent_name"] == "a" and len(data["result"]) == 2001
    assert map_langgraph_event({"event": "on_chain_start", "name": "other", "metadata": {"langgraph_node": "other"}}, {"a"}) is None


def test_event_bus_drops_when_full():
    bus = EventBus(queue_size=1)
    q = bus.subscribe("s")
    bus.publish(envelope("s", "a", {}))
    bus.publish(envelope("s", "b", {}))
    assert q.qsize() == 1
    bus.unsubscribe("s", q)
    assert bus.subscriber_count == 0


def test_fake_model_details():
    assert _text([{"text": "a"}, "b"]) == "ab" and _text(3) == "3"
    m = FakeChatModel(responses=[AIMessage(content="given")])
    assert m.invoke([HumanMessage(content="x")]).content == "given"

    class RM:
        tokens = []

        def on_llm_new_token(self, token, chunk=None):
            self.tokens.append(token)

    rm = RM()
    chunks = list(FakeChatModel(responses=["a b"])._stream([HumanMessage(content="x")], run_manager=rm))
    assert rm.tokens == ["a ", "b"] and len(chunks) == 3


# --------------------------------------------------------------------------- #
# Remaining branches
# --------------------------------------------------------------------------- #
async def test_listener_survives_abnormal_close():
    import websockets

    async def handler(ws):
        raise RuntimeError("server side failure")  # closes with code 1011

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        listener = BackendEventListener(f"ws://127.0.0.1:{port}")
        listener.start()
        await asyncio.sleep(1.4)  # > first reconnect delay (1 s): exercises the back-off
        assert listener.connected is False and not listener._task.done()
        await listener.stop()


def test_graph_can_be_built_before_provider(tmp_path):
    from langgraph_json.agent import build_graph
    from langgraph_json.utils.config_loader import load_config
    from langgraph_json.utils.tools import BUILTIN_TOOL_NAMES, ToolRegistry

    bundle = load_config(write_config(tmp_path / "c"), BUILTIN_TOOL_NAMES)
    graph, info = build_graph(bundle, ToolRegistry(bundle.tools, BackendClient("http://b")), LLMFactory())
    assert info.top_level == ["hmi", "jobs"] and "hmi__tools" in graph.get_graph().nodes


async def test_direct_call_of_sub_graph_agent(make_runtime):
    agents = {
        "team": {"type": "sub_graph", "description": "team", "system_prompt": "coordinate", "sub_agents": ["w"]},
        "w": agent("worker", ["open_item"], routable=False),
    }
    rt = await make_runtime(agents=agents, responses=["worker answer"])
    r = await rt.chat("x", agent="team")
    assert r["response"] == "worker answer" and r["route"][0] == "team"
    assert len(fake_llm_calls(rt)) == 1  # no supervisor call at all


def fake_llm_calls(rt):
    return rt.llms.get().calls


def test_no_proxy_wildcard_disables_mounts():
    assert ProxyConfig(https_proxy="http://p:1", no_proxy=["*"]).httpx_mounts() == {}
