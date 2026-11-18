"""Python tools (kind = "python"): whitelist, plugins, ctx injection, schemas,
hot-reload, parser support and end-to-end use by an agent."""

import json
import os
import threading
import time
import uuid
from pathlib import Path

import pytest
from conftest import ROOT, TOOLS, Backend, agent, call, fake_llm, route, write_config
from pydantic import ValidationError

from langgraph_json import tools_parser as tp
from langgraph_json.metadata import AI_METADATA, ToolContext
from langgraph_json.utils.backend_client import BackendClient
from langgraph_json.utils.config_loader import ConfigError, ToolDef, load_config
from langgraph_json.utils.tools import BUILTIN_TOOL_NAMES, PythonToolLoader, ToolExecutionError, ToolRegistry

EXAMPLE = ROOT / "examples" / "python"


def uniq(prefix: str = "plug") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def write_plugin(plugins: Path, module: str, code: str) -> Path:
    plugins.mkdir(parents=True, exist_ok=True)
    f = plugins / f"{module}.py"
    f.write_text(code, encoding="utf-8")
    return f


def pytool(name: str, spec: str, **kw) -> dict:
    return {"name": name, "kind": "python", "callable": spec, "description": f"{name} tool", **kw}


def registry(backend: Backend, defs: list[dict], allowed=(), plugins=None, **kw) -> ToolRegistry:
    return ToolRegistry(
        [ToolDef.model_validate(d) for d in defs],
        BackendClient("http://backend", transport=backend.transport),
        allowed_modules=list(allowed),
        plugins_dir=plugins,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Decorator and ToolDef
# --------------------------------------------------------------------------- #
def test_decorator_defaults():
    @AI_METADATA({"name": "x", "description": "d"})
    def f():
        pass

    assert f.__ai_metadata__ == {"name": "x", "description": "d", "kind": "python", "callable": f"{__name__}:f"}

    @AI_METADATA({"name": "y", "method": "GET", "path": "/y", "description": "d", "parameters": {}})
    def handler():
        pass

    assert handler.__ai_metadata__["kind"] == "rest" and "callable" not in handler.__ai_metadata__
    with pytest.raises(ValueError):
        AI_METADATA({"description": "no name"})


def test_tooldef_kind_validation():
    assert ToolDef.model_validate(pytool("a", "pkg.mod:fn", read_only=True)).read_only
    assert not ToolDef.model_validate(pytool("a", "pkg.mod:fn")).read_only
    with pytest.raises(ValidationError, match="requires 'callable'"):
        ToolDef.model_validate({"name": "a", "kind": "python", "description": "d"})
    with pytest.raises(ValidationError, match="requires 'callable'"):
        ToolDef.model_validate(pytool("a", "not a callable"))
    with pytest.raises(ValidationError, match="requires 'method' and 'path'"):
        ToolDef.model_validate({"name": "a", "description": "d"})
    assert ToolDef.model_validate(pytool("a", "pkg.mod:fn")).path_params == []


def test_config_rejects_modules_not_whitelisted(tmp_path):
    cfg = write_config(tmp_path, tools=TOOLS + [pytool("evil", "os:system")])
    with pytest.raises(ConfigError, match="module 'os' is not allowed"):
        load_config(cfg, BUILTIN_TOOL_NAMES)
    write_config(tmp_path, tools=TOOLS + [pytool("ok", "my_tools.units:convert_distance")], server={"python_tool_modules": ["my_tools"]})
    load_config(cfg, BUILTIN_TOOL_NAMES)
    # sub-modules are allowed, look-alike prefixes are not
    write_config(tmp_path, tools=[pytool("bad", "my_tools_evil:x")], server={"python_tool_modules": ["my_tools"]})
    with pytest.raises(ConfigError, match="not allowed"):
        load_config(cfg, BUILTIN_TOOL_NAMES)


def test_config_rejects_builtin_name_conflict(tmp_path):
    cfg = write_config(tmp_path, tools=[pytool("wait_for_event", "my_tools.x:y")], server={"python_tool_modules": ["my_tools"]})
    with pytest.raises(ConfigError, match="conflicts with a built-in"):
        load_config(cfg, BUILTIN_TOOL_NAMES)


def test_plugins_dir_resolution(tmp_path):
    from langgraph_json.utils.config_loader import ServerSettings

    s = ServerSettings(plugins_dir="plugins")
    assert s.resolve_plugins_dir(tmp_path / "config") == (tmp_path / "plugins").resolve()
    assert ServerSettings(plugins_dir=str(tmp_path)).resolve_plugins_dir(Path("x")) == tmp_path
    assert ServerSettings().resolve_plugins_dir(tmp_path) is None


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def test_loader_whitelist_and_errors(tmp_path):
    mod = uniq()
    write_plugin(tmp_path, mod, "def ok():\n    return 1\nVALUE = 3\n")
    loader = PythonToolLoader([mod], tmp_path)
    assert loader.load(f"{mod}:ok")() == 1
    with pytest.raises(ConfigError, match="not allowed"):
        loader.load("os:getcwd")
    with pytest.raises(ConfigError, match="is not a function"):
        loader.load(f"{mod}:VALUE")
    with pytest.raises(ConfigError, match="is not a function"):
        loader.load(f"{mod}:missing")
    broken = uniq()
    write_plugin(tmp_path, broken, "import does_not_exist_xyz\n")
    with pytest.raises(ConfigError, match="cannot import python tool module.*ModuleNotFoundError"):
        PythonToolLoader([broken], tmp_path).load(f"{broken}:f")
    with pytest.raises(ConfigError, match="plugins_dir does not exist"):
        PythonToolLoader([], tmp_path / "nope")


def test_builtins_are_always_allowed(backend):
    reg = registry(backend, [])
    assert {"wait_for_event", "list_shared_files"} <= set(reg.tools)
    assert reg.tools["wait_for_event"].metadata["kind"] == "python"


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
PLUGIN = '''
import threading
from typing import Literal

from langgraph_json.metadata import ToolContext

CALLS = []

def sync_tool(a: int, ctx: ToolContext, b: int = 10):
    """Adds numbers.

    Long description ignored."""
    CALLS.append(threading.current_thread().name)
    return {"sum": a + b, "has_ctx": isinstance(ctx, ToolContext)}

async def async_tool(text: str, mode: Literal["upper", "lower"] = "upper"):
    return text.upper() if mode == "upper" else text.lower()

async def rest_tool(job_id: str, ctx):
    status = await ctx.client.request("GET", f"/api/jobs/{job_id}")
    return {"wrapped": status}

def failing():
    raise RuntimeError("plugin exploded")

def big():
    return "x" * 10000

def returns_coroutine():
    async def inner():
        return "from coroutine"
    return inner()
'''


@pytest.fixture
def plugin(tmp_path):
    mod = uniq()
    write_plugin(tmp_path / "plugins", mod, PLUGIN)
    return mod, tmp_path / "plugins"


async def test_sync_tool_runs_in_thread_with_ctx(backend, plugin):
    mod, plugins = plugin
    reg = registry(backend, [pytool("sync_tool", f"{mod}:sync_tool")], [mod], plugins)
    tool = reg.tools["sync_tool"]
    schema = tool.args_schema.model_json_schema()
    assert set(schema["properties"]) == {"a", "b"} and schema["required"] == ["a"]  # ctx hidden
    out = json.loads(await tool.ainvoke({"a": 1}))
    assert out == {"sum": 11, "has_ctx": True}
    import sys

    assert sys.modules[mod].CALLS[-1] != threading.current_thread().name


async def test_async_tool_and_inferred_enum(backend, plugin):
    mod, plugins = plugin
    tool = registry(backend, [pytool("async_tool", f"{mod}:async_tool")], [mod], plugins).tools["async_tool"]
    props = tool.args_schema.model_json_schema()["properties"]
    assert props["mode"]["enum"] == ["upper", "lower"]
    assert await tool.ainvoke({"text": "Hi"}) == "HI"
    assert await tool.ainvoke({"text": "Hi", "mode": "lower"}) == "hi"
    with pytest.raises(ValidationError):
        await tool.ainvoke({"text": "Hi", "mode": "other"})
    with pytest.raises(ValidationError):
        await tool.ainvoke({"text": "Hi", "unexpected": 1})


async def test_explicit_parameters_override_signature(backend, plugin):
    mod, plugins = plugin
    spec = pytool("sync_tool", f"{mod}:sync_tool", parameters={"a": {"type": "integer", "required": True, "description": "first"}})
    tool = registry(backend, [spec], [mod], plugins).tools["sync_tool"]
    schema = tool.args_schema.model_json_schema()
    assert set(schema["properties"]) == {"a"} and schema["properties"]["a"]["description"] == "first"
    assert json.loads(await tool.ainvoke({"a": 5}))["sum"] == 15  # function default b=10 applies


async def test_python_tool_can_call_backend(backend, plugin):
    mod, plugins = plugin
    backend.responses[("GET", "/api/jobs/J1")] = {"state": "done"}
    tool = registry(backend, [pytool("rest_tool", f"{mod}:rest_tool")], [mod], plugins).tools["rest_tool"]
    assert json.loads(await tool.ainvoke({"job_id": "J1"})) == {"wrapped": {"state": "done"}}
    assert backend.calls() == [("GET", "/api/jobs/J1")]


async def test_errors_truncation_and_coroutine_results(backend, plugin):
    mod, plugins = plugin
    defs = [pytool("failing", f"{mod}:failing"), pytool("big", f"{mod}:big"), pytool("returns_coroutine", f"{mod}:returns_coroutine")]
    reg = registry(backend, defs, [mod], plugins, max_result_chars=300)
    with pytest.raises(ToolExecutionError, match="RuntimeError: plugin exploded"):
        await reg.tools["failing"].ainvoke({})
    assert "truncated" in await reg.tools["big"].ainvoke({})
    assert await reg.tools["returns_coroutine"].ainvoke({}) == "from coroutine"


def test_description_from_metadata_or_docstring(backend, plugin):
    mod, plugins = plugin
    reg = registry(backend, [pytool("sync_tool", f"{mod}:sync_tool", description="", returns={"sum": "int"})], [mod], plugins)
    desc = reg.tools["sync_tool"].description
    assert desc.startswith("Adds numbers.") and "Long description" not in desc and "Returns:" in desc


def test_describe_includes_kind_and_callable(backend, plugin):
    mod, plugins = plugin
    d = {x["name"]: x for x in registry(backend, [*TOOLS, pytool("async_tool", f"{mod}:async_tool")], [mod], plugins).describe()}
    assert d["async_tool"]["kind"] == "python" and d["async_tool"]["callable"] == f"{mod}:async_tool"
    assert d["async_tool"]["read_only"] is False and "text" in d["async_tool"]["parameters"]
    assert d["open_item"]["kind"] == "rest" and d["open_item"]["method"] == "POST"


async def test_plugin_hot_reload(backend, tmp_path):
    mod = uniq()
    plugins = tmp_path / "plugins"
    f = write_plugin(plugins, mod, "def v():\n    return 'one'\n")
    spec = [pytool("v", f"{mod}:v")]
    assert await registry(backend, spec, [mod], plugins).tools["v"].ainvoke({}) == "one"
    f.write_text("def v():\n    return 'second version'\n", encoding="utf-8")
    later = time.time() + 5
    os.utime(f, (later, later))
    assert await registry(backend, spec, [mod], plugins).tools["v"].ainvoke({}) == "second version"


# --------------------------------------------------------------------------- #
# Shipped example package
# --------------------------------------------------------------------------- #
def example_registry(backend, tmp_path):
    res = tp.parse_sources([EXAMPLE], lang="python")
    assert res.errors == []
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "data.csv").write_text("id,value\n1,10\n2,20\n3,30\n", encoding="utf-8")
    return registry(backend, res.tools, ["my_tools"], EXAMPLE, shared_paths={"inbox": str(inbox)})


async def test_example_csv_summary(backend, tmp_path):
    tool = example_registry(backend, tmp_path).tools["csv_summary"]
    out = json.loads(await tool.ainvoke({"folder": "inbox", "file": "data.csv", "sample_rows": 2}))
    assert out == {"file": "data.csv", "rows": 3, "columns": ["id", "value"], "sample": [{"id": "1", "value": "10"}, {"id": "2", "value": "20"}]}
    with pytest.raises(ToolExecutionError, match="inside the shared folder"):
        await tool.ainvoke({"folder": "inbox", "file": "../secret.txt"})
    with pytest.raises(ToolExecutionError, match="unknown shared folder"):
        await tool.ainvoke({"folder": "outbox", "file": "a.csv"})


async def test_example_jobs_overview(backend, tmp_path):
    backend.responses[("GET", "/api/jobs/A")] = {"state": "done"}
    backend.responses[("GET", "/api/jobs/B")] = {"state": "running"}
    out = json.loads(await example_registry(backend, tmp_path).tools["jobs_overview"].ainvoke({"job_ids": ["A", "B"]}))
    assert out == {"total": 2, "done": 1, "states": {"A": "done", "B": "running"}}


async def test_example_convert_distance_inferred_schema(backend, tmp_path):
    tool = example_registry(backend, tmp_path).tools["convert_distance"]
    assert tool.args_schema.model_json_schema()["required"] == ["value", "from_unit"]
    assert json.loads(await tool.ainvoke({"value": 1, "from_unit": "nm"})) == {"value": 1852.0, "unit": "m"}
    assert json.loads(await tool.ainvoke({"value": 1000, "from_unit": "m", "to_unit": "km"}))["value"] == 1.0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def test_parser_python_example():
    res = tp.parse_sources([EXAMPLE], lang="python")
    assert res.errors == []
    tools = {t["name"]: t for t in res.tools}
    assert tools["csv_summary"]["callable"] == "my_tools.files:csv_summary"
    assert tools["csv_summary"]["read_only"] is True and tools["csv_summary"]["kind"] == "python"
    assert tools["convert_distance"]["callable"] == "my_tools.units:convert_distance" and "parameters" not in tools["convert_distance"]
    for t in res.tools:
        ToolDef.model_validate(t)


def test_parser_builtins_match_runtime_metadata():
    from langgraph_json.builtin_tools import BUILTINS

    res = tp.parse_sources([ROOT / "langgraph_json" / "builtin_tools.py"], lang="python", python_root=ROOT)
    parsed = {t["name"]: t["callable"] for t in res.tools}
    assert parsed == {fn.__ai_metadata__["name"]: fn.__ai_metadata__["callable"] for fn in BUILTINS}


def test_parser_python_kinds_and_errors(tmp_path):
    pkg = tmp_path / "app" / "api"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        '@AI_METADATA({"name": "pkg_tool", "description": "in __init__"})\n'
        "async def pkg_tool(): ...\n"
    )
    (pkg / "routes.py").write_text(
        "@app.get('/items')\n"
        '@AI_METADATA({"name": "list_items", "method": "GET", "path": "/items", "description": "Flask-style", "parameters": {}, "domain": "web"})\n'
        "def list_items(): ...\n"
        '@AI_METADATA({"name": "explicit", "kind": "python", "callable": "other.mod:fn", "description": "given"})\n'
        "def whatever(): ...\n"
        '@AI_METADATA({"name": "orphan", "description": "no function follows"})\n'
        "x = 1\n"
        '@AI_METADATA({"name": "badkind", "kind": "grpc", "description": "d"})\n'
        "def f(): ...\n"
    )
    res = tp.parse_sources([tmp_path / "app"], lang="python")
    tools = {t["name"]: t for t in res.tools}
    assert tools["pkg_tool"]["callable"] == "api:pkg_tool"
    assert tools["list_items"].get("kind", "rest") == "rest" and tools["list_items"]["method"] == "GET"
    assert tools["explicit"]["callable"] == "other.mod:fn"
    text = "\n".join(res.errors)
    assert "orphan" not in tools and "without 'callable'" in text
    assert "invalid kind 'grpc'" in text
    # an explicit import root changes the module path
    res2 = tp.parse_sources([tmp_path / "app"], lang="python", python_root=tmp_path)
    assert {t["name"]: t.get("callable") for t in res2.tools}["pkg_tool"] == "app.api:pkg_tool"


def test_parser_python_literals_only_in_python_profile(tmp_path):
    (tmp_path / "a.py").write_text('@AI_METADATA({"name": "lit", "description": "d", "read_only": True, "parameters": {"x": {"type": "string", "required": True}}})\ndef lit(x): ...\n')
    res = tp.parse_sources([tmp_path], lang="python")
    assert res.errors == [] and res.tools[0]["read_only"] is True
    (tmp_path / "b.cpp").write_text('AI_METADATA({"name": "lit", "method": "GET", "path": "/a", "description": "d", "parameters": {"x": {"required": True}}})')
    assert any("malformed JSON" in e for e in tp.parse_sources([tmp_path / "b.cpp"], lang="cpp").errors)


def test_parser_cli_python_root(tmp_path):
    out = tmp_path / "tools.json"
    assert tp.main(["--source-dir", str(EXAMPLE / "my_tools"), "--lang", "python", "--python-root", str(EXAMPLE), "--output", str(out)]) == 0
    callables = {t["callable"] for t in json.loads(out.read_text())["tools"]}
    assert callables == {"my_tools.files:csv_summary", "my_tools.jobs:jobs_overview", "my_tools.units:convert_distance"}


# --------------------------------------------------------------------------- #
# End to end: an agent using Python + REST tools
# --------------------------------------------------------------------------- #
@pytest.fixture
def py_runtime(make_runtime, tmp_path):
    mod = uniq("e2e")
    write_plugin(tmp_path / "plugins", mod, PLUGIN)
    tools = TOOLS + [pytool("sync_tool", f"{mod}:sync_tool"), pytool("async_tool", f"{mod}:async_tool")]
    agents = {
        "mixed": agent("mixed", ["open_item", "sync_tool", "async_tool"], requires_confirmation=["sync_tool"]),
        "other": agent("other", ["open_item"]),
    }

    async def _make(responses):
        return await make_runtime(agents=agents, tools=tools, responses=responses,
                                  server={"python_tool_modules": [mod], "plugins_dir": "plugins"})

    return _make, mod


async def test_agent_uses_python_and_rest_tools(py_runtime, backend):
    make, _ = py_runtime
    rt = await make([
        route("mixed"),
        {"tool_calls": [{"name": "async_tool", "args": {"text": "abc"}}, {"name": "open_item", "args": {"itemType": "Project", "itemId": "P"}}]},
        "done ABC",
        route("FINISH"),
    ])
    r = await rt.chat("go", "s")
    assert r["status"] == "completed" and r["response"] == "done ABC"
    assert [(t["tool_name"], t["status"]) for t in r["tool_calls"]] == [("async_tool", "ok"), ("open_item", "ok")]
    assert r["tool_calls"][0]["result"] == "ABC"
    assert backend.calls() == [("POST", "/api/items/open")]
    assert fake_llm(rt).calls[1]["tools"] == ["open_item", "sync_tool", "async_tool"]


async def test_python_tool_confirmation(py_runtime):
    make, _ = py_runtime
    rt = await make([route("mixed"), call("sync_tool", a=2, b=3), "sum is 5", route("FINISH")])
    r = await rt.chat("add", "s")
    assert r["status"] == "confirmation_needed" and r["confirmation"]["tool_name"] == "sync_tool"
    r = await rt.confirm("s", True)
    tool_msg = next(m for m in r["messages"] if m["type"] == "tool")
    assert json.loads(tool_msg["content"]) == {"sum": 5, "has_ctx": True}


async def test_python_tool_error_reaches_llm(make_runtime, tmp_path):
    mod = uniq("err")
    write_plugin(tmp_path / "plugins", mod, PLUGIN)
    rt = await make_runtime(
        agents={"solo": agent("solo", ["failing"])},
        tools=[pytool("failing", f"{mod}:failing")],
        server={"python_tool_modules": [mod], "plugins_dir": "plugins"},
        responses=[call("failing"), "it failed"],
    )
    r = await rt.chat("x")
    tool_msg = next(m for m in r["messages"] if m["type"] == "tool")
    assert tool_msg["status"] == "error" and "plugin exploded" in tool_msg["content"]
    assert r["response"] == "it failed"


async def test_reload_picks_up_plugin_changes_and_reports_import_errors(make_runtime, tmp_path):
    mod = uniq("reload")
    f = write_plugin(tmp_path / "plugins", mod, "def v():\n    return 'one'\n")
    rt = await make_runtime(
        agents={"solo": agent("solo", ["v"])},
        tools=[pytool("v", f"{mod}:v")],
        server={"python_tool_modules": [mod], "plugins_dir": "plugins"},
        responses=[call("v"), "first", call("v"), "second"],
    )
    r = await rt.chat("x", "s")
    assert r["tool_calls"][0]["result"] == "one"
    f.write_text("def v():\n    return 'version two'\n", encoding="utf-8")
    later = time.time() + 5
    os.utime(f, (later, later))
    rt.reload()
    r = await rt.chat("y", "s")
    assert r["tool_calls"][0]["result"] == "version two"
    # a broken plugin makes the reload fail and keeps the previous tools
    f.write_text("def v(:\n", encoding="utf-8")
    os.utime(f, (later + 5, later + 5))
    with pytest.raises(ConfigError, match="cannot import"):
        rt.reload()
    assert rt.registry.tools["v"] is not None and rt.graph is not None
