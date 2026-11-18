"""Library use: in-memory configuration, Python functions as tools, injected
LLM / checkpointer, mounting in a host FastAPI app, public API."""

import json
import subprocess
import sys
from contextlib import asynccontextmanager

import pytest
from conftest import ROOT, TOOLS, agent, call, route
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import langgraph_json
from langgraph_json import AI_METADATA, ConfigError, FakeChatModel, Runtime, create_app
from langgraph_json.utils.llm_factory import StaticLLMFactory


@AI_METADATA({"name": "weather", "description": "Weather of a city", "read_only": True})
def weather_tool(city: str) -> dict:
    return {"city": city, "temp": 21}


def add(a: int, b: int = 1) -> int:
    """Adds two integers.

    More text."""
    return a + b


async def shout(text: str, ctx) -> str:
    return text.upper() + f" ({len(ctx.shared_paths)} folders)"


AGENTS = {
    "agents": {
        "meteo": agent("weather agent", ["weather"]),
        "calc": agent("calculator", ["add", "shout"], requires_confirmation=["add"]),
    }
}


def test_public_api_is_lazy():
    code = "import sys, langgraph_json; print('langgraph' in sys.modules, langgraph_json.__version__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT).stdout.split()
    assert out == ["False", langgraph_json.__version__]
    assert langgraph_json.Runtime is Runtime
    with pytest.raises(AttributeError):
        langgraph_json.does_not_exist  # noqa: B018
    for name in langgraph_json.__all__:
        assert getattr(langgraph_json, name) is not None


async def test_in_memory_config_with_functions_and_injected_llm(backend):
    llm = FakeChatModel(responses=[route("meteo"), call("weather", city="Paris"), "21 degrees in Paris", route("FINISH")])
    async with Runtime(agents=AGENTS, functions=[weather_tool, add, shout], llm=llm) as rt:
        assert rt.status()["status"] == "running" and rt.status()["provider_configured"]
        r = await rt.chat("weather in Paris?", "s")
        assert r["response"] == "21 degrees in Paris"
        assert json.loads(r["tool_calls"][0]["result"]) == {"city": "Paris", "temp": 21}
        tools = {t["name"]: t for t in rt.registry.describe()}
        assert tools["add"]["description"] == "Adds two integers."
        assert tools["weather"]["read_only"] is True
        assert tools["add"]["callable"].endswith(":add")
    assert llm.calls[1]["tools"] == ["weather"]


async def test_function_tools_confirmation_and_ctx():
    llm = FakeChatModel(responses=[route("calc"), {"tool_calls": [{"name": "add", "args": {"a": 2}}, {"name": "shout", "args": {"text": "hi"}}]}, "3 and HI", route("FINISH")])
    async with Runtime(agents=AGENTS, functions=[weather_tool, add, shout], llm=llm) as rt:
        rt.set_paths({})
        r = await rt.chat("compute", "s")
        assert r["status"] == "confirmation_needed" and r["confirmation"]["tool_name"] == "add"
        r = await rt.confirm("s", True)
        results = [m["content"] for m in r["messages"] if m["type"] == "tool"]
        assert results == ["3", "HI (0 folders)"]


async def test_rest_tools_from_in_memory_tools_list(backend):
    llm = FakeChatModel(responses=[call("open_item", itemType="Document", itemId="7"), "opened"])
    async with Runtime(agents={"agents": {"solo": agent("solo", ["open_item"])}}, tools=TOOLS, llm=llm,
                       settings={"backend_url": "http://my-backend:9999"}, backend_transport=backend.transport) as rt:
        r = await rt.chat("open 7")
        assert r["response"] == "opened"
    assert str(backend.requests[0].url) == "http://my-backend:9999/api/items/open"


async def test_invalid_in_memory_config_raises_in_context_manager():
    with pytest.raises(ConfigError, match="at least one tool"):
        async with Runtime(agents={"agents": {"a": agent("a", [])}}, llm=FakeChatModel()):
            pass
    with pytest.raises(ConfigError, match="system_prompt"):
        async with Runtime(agents={"agents": {"a": {"description": "x", "allowed_tools": ["t"]}}}, llm=FakeChatModel()):
            pass
    with pytest.raises(ConfigError, match="duplicate tool names"):
        async with Runtime(agents=AGENTS, tools=[TOOLS[0], TOOLS[0]], functions=[weather_tool, add, shout], llm=FakeChatModel()):
            pass


def test_constructor_errors(tmp_path):
    with pytest.raises(ValueError, match="either"):
        Runtime()
    with pytest.raises(ValueError, match="mutually exclusive"):
        Runtime(tmp_path, agents={})


async def test_function_tool_errors():
    def weird():
        pass

    weird.__ai_metadata__ = {"name": "bad name!", "description": "x"}
    with pytest.raises(ConfigError, match="invalid tool function"):
        async with Runtime(agents={"agents": {"a": agent("a", ["x"])}}, functions=[weird], llm=FakeChatModel()):
            pass

    def wait_for_event():
        """Shadows a built-in."""

    with pytest.raises(ConfigError, match="duplicate tool name 'wait_for_event'"):
        async with Runtime(agents={"agents": {"a": agent("a", ["x"])}}, functions=[wait_for_event], llm=FakeChatModel()):
            pass


async def test_update_config_hot_reload():
    llm = FakeChatModel(responses=["first", "second"])
    async with Runtime(agents={"agents": {"solo": agent("solo", ["add"])}}, functions=[add], llm=llm) as rt:
        await rt.chat("a", "s")
        new_agents = {"agents": {"solo": {**agent("solo", ["add"]), "system_prompt": "Updated prompt."}}}
        assert rt.update_config(agents=new_agents)["status"] == "reloaded"
        await rt.chat("b", "s")
        assert llm.calls[-1]["messages"][0].content.startswith("Updated prompt.")
        with pytest.raises(ConfigError):
            rt.update_config(agents={"agents": {"solo": agent("solo", [])}})
        assert rt._data["agents"] == new_agents  # previous configuration kept
        rt.update_config(settings={"max_tool_calls_per_turn": 1})
        assert rt.bundle.settings.max_tool_calls_per_turn == 1


async def test_update_config_refused_for_file_runtime(make_runtime):
    rt = await make_runtime()
    with pytest.raises(RuntimeError, match="from files"):
        rt.update_config(agents={})


async def test_several_models_by_provider_name():
    big = FakeChatModel(responses=["from big"], model_name="big")
    small = FakeChatModel(responses=["from small"], model_name="small")
    agents = {"agents": {"a": agent("a", ["add"], provider="small"), "b": agent("b", ["add"])}}
    async with Runtime(agents=agents, functions=[add], llm={"big": big, "small": small}) as rt:
        assert (await rt.chat("x", agent="a"))["response"] == "from small"
        assert (await rt.chat("y", agent="b"))["response"] == "from big"
        assert rt.llms.available_models() == {"big": ["big"], "small": ["small"]}
        # proxy settings do not replace caller-built models
        from langgraph_json import ProxyConfig

        rt.set_proxy(ProxyConfig(https_proxy="http://p:1"))
        assert isinstance(rt.llms, StaticLLMFactory) and rt.llms.proxy.https_proxy == "http://p:1"
    with pytest.raises(ConfigError, match="unknown providers"):
        async with Runtime(agents={"agents": {"a": agent("a", ["add"], provider="ghost")}}, functions=[add], llm=big):
            pass


def test_static_factory_validation():
    from langgraph_json.utils.llm_factory import ProviderNotConfigured

    with pytest.raises(ValueError):
        StaticLLMFactory({})
    with pytest.raises(ValueError):
        StaticLLMFactory({"a": FakeChatModel()}, default="b")
    f = StaticLLMFactory(FakeChatModel())
    with pytest.raises(ProviderNotConfigured):
        f.get("other")
    assert f.provider_names() == {"default"}


async def test_shared_checkpointer_persists_across_runtimes():
    saver = InMemorySaver()  # stands for any persistent LangGraph checkpointer (SQLite, Postgres...)
    cfg = {"agents": {"solo": agent("solo", ["add"])}}
    async with Runtime(agents=cfg, functions=[add], llm=FakeChatModel(responses=["noted"]), checkpointer=saver) as rt:
        await rt.chat("remember 42", "persistent")
    llm2 = FakeChatModel(responses=["it was 42"])
    async with Runtime(agents=cfg, functions=[add], llm=llm2, checkpointer=saver) as rt2:
        await rt2.chat("what was it?", "persistent")
    human = [m.content for m in llm2.calls[0]["messages"] if m.type == "human"]
    assert human == ["remember 42", "what was it?"]


def test_mount_in_host_fastapi_app():
    rt = Runtime(agents=AGENTS, functions=[weather_tool, add, shout], llm=FakeChatModel(responses=[route("FINISH")]))

    @asynccontextmanager
    async def lifespan(app):
        async with rt:
            yield

    host = FastAPI(lifespan=lifespan)
    host.mount("/agents", create_app(rt, manage_lifespan=False))

    @host.get("/")
    def home():
        return {"host": True}

    with TestClient(host) as c:
        assert c.get("/").json() == {"host": True}
        assert c.get("/agents/status").json()["status"] == "running"
        r = c.post("/agents/chat", json={"message": "hello"}).json()
        assert r["status"] == "completed"
        with c.websocket_connect("/agents/stream") as ws:
            assert ws.receive_json()["event_type"] == "stream.connected"


def test_example_scripts_run():
    import os

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "embedded" / "inprocess.py")], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "[weather] It is 21" in r.stdout and "[calculator] 2 + 3 = 5" in r.stdout

    code = (
        "from fastapi.testclient import TestClient\n"
        "from examples.embedded.mounted_fastapi import app\n"
        "with TestClient(app) as c:\n"
        "    print(c.get('/').json()['agents'], c.get('/agents/status').json()['agents_active'])\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["waiting_provider", "3"]
