import json

import httpx
import pytest
from conftest import TOOLS, Backend
from pydantic import ValidationError

from langgraph_json.utils.backend_client import BackendClient, BackendEventListener
from langgraph_json.utils.config_loader import ToolDef
from langgraph_json.utils.tools import ToolExecutionError, ToolRegistry, build_args_schema, sanitize_result

DEFS = [ToolDef.model_validate(t) for t in TOOLS]


def registry(backend: Backend, **kw) -> ToolRegistry:
    return ToolRegistry(DEFS, BackendClient("http://backend", transport=backend.transport), **kw)


def test_args_schema_types_required_and_enum():
    schema = build_args_schema(DEFS[0]).model_json_schema()
    assert set(schema["required"]) == {"itemType", "itemId"}
    assert schema["properties"]["itemType"]["enum"] == ["Document", "Project"]
    Model = build_args_schema(DEFS[0])
    with pytest.raises(ValidationError):
        Model(itemType="Unknown", itemId="1")
    with pytest.raises(ValidationError):
        Model(itemType="Document", itemId="1", extra="nope")


def test_array_items_and_path_params_are_required():
    t = ToolDef.model_validate(
        {"name": "t", "method": "GET", "path": "/x/{id}", "description": "d",
         "parameters": {"id": {"type": "string"}, "tags": {"type": "array", "items": {"type": "integer"}}}}
    )
    M = build_args_schema(t)
    assert M.model_json_schema()["required"] == ["id"]
    assert M(id="a", tags=[1, 2]).tags == [1, 2]
    assert t.path_params == ["id"] and t.read_only


async def test_post_tool_sends_json_body(backend):
    out = await registry(backend).tools["open_item"].ainvoke({"itemType": "Document", "itemId": "42"})
    req = backend.requests[-1]
    assert (req.method, req.url.path) == ("POST", "/api/items/open")
    assert json.loads(req.content) == {"itemType": "Document", "itemId": "42"}  # None args dropped
    assert json.loads(out)["ok"] is True


async def test_get_tool_substitutes_path_and_uses_query(backend):
    await registry(backend).tools["get_job_status"].ainvoke({"jobId": "J 1", "verbose": True})
    req = backend.requests[-1]
    assert req.method == "GET" and req.url.path == "/api/jobs/J 1"
    assert req.url.params["verbose"] == "true" and req.content == b""


async def test_delete_tool(backend):
    await registry(backend).tools["delete_thing"].ainvoke({"id": "x"})
    assert backend.calls()[-1] == ("DELETE", "/api/thing/x")


async def test_backend_http_error_becomes_tool_error(backend):
    backend.responses[("POST", "/api/items/open")] = httpx.Response(404, text="no such item")
    with pytest.raises(ToolExecutionError, match="HTTP 404.*no such item"):
        await registry(backend).tools["open_item"].ainvoke({"itemType": "Document", "itemId": "1"})


async def test_backend_unreachable():
    def boom(request):
        raise httpx.ConnectError("refused")

    reg = ToolRegistry(DEFS, BackendClient("http://backend", transport=httpx.MockTransport(boom)))
    with pytest.raises(ToolExecutionError, match="unreachable"):
        await reg.tools["get_job_status"].ainvoke({"jobId": "1"})


async def test_large_results_are_truncated(backend):
    backend.responses[("GET", "/api/jobs/1")] = {"rows": ["x" * 100] * 100}
    out = await registry(backend, max_result_chars=500).tools["get_job_status"].ainvoke({"jobId": "1"})
    assert len(out) < 700 and "truncated" in out


def test_sanitize_strips_control_characters():
    assert sanitize_result("a\x00b\x1bc\nd\te", 100) == "abc\nd\te"
    assert sanitize_result({"k": "é"}, 100) == '{"k": "é"}'


def test_whitelist(backend):
    reg = registry(backend)
    tools = reg.for_agent(["open_item", "ghost", "open_item", "wait_for_event"])
    assert [t.name for t in tools] == ["open_item", "wait_for_event"]


def test_describe(backend):
    desc = {d["name"]: d for d in registry(backend).describe()}
    assert desc["get_job_status"]["read_only"] is True
    assert desc["open_item"]["read_only"] is False and desc["open_item"]["domain"] == "hmi"
    assert desc["wait_for_event"]["domain"] == "builtin"
    assert "itemId" in desc["open_item"]["parameters"]


async def test_list_shared_files(backend, tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.csv").write_text("1")
    (tmp_path / "in" / "b.txt").write_text("2")
    paths = {"inbox": str(tmp_path / "in")}
    tool = registry(backend, shared_paths=paths).tools["list_shared_files"]
    out = json.loads(await tool.ainvoke({"folder": "inbox", "pattern": "*.csv"}))
    assert [f["name"] for f in out["files"]] == ["a.csv"]
    with pytest.raises(ToolExecutionError, match="inside"):
        await tool.ainvoke({"folder": "inbox", "pattern": "../*"})
    with pytest.raises(ToolExecutionError, match="unknown shared folder"):
        await tool.ainvoke({"folder": "outbox"})


async def test_wait_for_event_builtin(backend):
    listener = BackendEventListener("ws://unused")
    tool = registry(backend, listener=listener).tools["wait_for_event"]
    await listener.push({"type": "job.completed", "data": {"jobId": "J1"}})
    out = json.loads(await tool.ainvoke({"event_type": "job.completed", "match": {"jobId": "J1"}, "timeout_s": 1}))
    assert out["data"]["jobId"] == "J1"
    with pytest.raises(ToolExecutionError, match="no 'job.failed'"):
        await tool.ainvoke({"event_type": "job.failed", "timeout_s": 1})


async def test_wait_for_event_without_ws_configured(backend):
    with pytest.raises(ToolExecutionError, match="not configured"):
        await registry(backend).tools["wait_for_event"].ainvoke({"event_type": "x", "timeout_s": 1})
