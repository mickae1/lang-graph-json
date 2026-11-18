"""HTTP + WebSocket API of the gateway (FastAPI TestClient, scripted LLM)."""

import json

import pytest
from conftest import agent, call, route, write_config
from fastapi.testclient import TestClient

from langgraph_json.gateway import Runtime, create_app


@pytest.fixture
def make_client(backend, tmp_path):
    clients = []

    def _make(agents=None, server=None, **extra):
        cfg = write_config(tmp_path / f"cfg{len(clients)}", agents, None, server, **extra)
        client = TestClient(create_app(Runtime(cfg, backend_transport=backend.transport)))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def push_fake(client, *responses):
    r = client.post("/config/provider", json={"type": "fake", "fake_responses": list(responses)})
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- #
def test_status_before_and_after_provider(make_client):
    c = make_client()
    s = c.get("/status").json()
    assert s["status"] == "waiting_provider" and not s["provider_configured"]
    assert s["agents_total"] == 2 and s["agents_active"] == 2 and s["sessions"] == 0
    assert c.post("/chat", json={"message": "hi"}).status_code == 503
    out = push_fake(c, "x")
    assert out["graph_ready"] is True
    s = c.get("/status").json()
    assert s["status"] == "running" and {a["name"]: a["tools"] for a in s["agents"]} == {"hmi": 1, "jobs": 2}


def test_invalid_config_at_startup(make_client, tmp_path):
    c = make_client()
    (c.app.state.runtime.config_dir / "agents.json").write_text("{broken")
    c2 = TestClient(create_app(Runtime(c.app.state.runtime.config_dir)))
    with c2:
        assert c2.get("/status").json()["status"] == "error"
        push_fake(c2, "x")
        r = c2.post("/chat", json={"message": "hi"})
        assert r.status_code == 503 and "invalid" in r.json()["error"]
        assert c2.get("/agents").status_code == 503


def test_chat_round_trip_and_sessions(make_client, backend):
    c = make_client()
    push_fake(c, route("hmi"), call("open_item", itemType="Document", itemId="1"), "opened", route("FINISH"))
    r = c.post("/chat", json={"message": "open doc 1", "session_id": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["response"] == "opened" and body["session_id"] == "abc"
    assert [m["type"] for m in body["messages"]] == ["human", "ai", "tool", "ai"]
    assert c.get("/sessions").json()["sessions"][0]["turns"] == 1
    hist = c.get("/sessions/abc").json()["messages"]
    assert hist[0] == {"type": "human", "name": None, "content": "open doc 1"}
    assert c.delete("/sessions/abc").json()["status"] == "deleted"
    assert c.get("/sessions/abc").status_code == 404
    assert c.delete("/sessions/abc").status_code == 404


def test_chat_generates_session_id_and_validates_input(make_client):
    c = make_client()
    push_fake(c, route("FINISH"))
    sid = c.post("/chat", json={"message": "hi"}).json()["session_id"]
    assert sid.startswith("sess_")
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={"message": "x", "session_id": "bad id!"}).status_code == 422
    assert c.post("/chat", json={"message": "x", "agent": "ghost"}).status_code == 400


def test_confirm_over_http(make_client, backend):
    c = make_client()
    push_fake(c, route("jobs"), call("run_job", jobId="J"), "started", route("FINISH"))
    r = c.post("/chat", json={"message": "run", "session_id": "s"}).json()
    assert r["status"] == "confirmation_needed" and r["confirmation"]["interrupt_id"]
    assert c.post("/chat", json={"message": "again", "session_id": "s"}).status_code == 409
    assert c.post("/confirm", json={"session_id": "nope", "approved": True}).status_code == 404
    r = c.post("/confirm", json={"session_id": "s", "approved": True}).json()
    assert r["status"] == "completed" and backend.calls() == [("POST", "/api/jobs/J/run")]
    assert c.post("/confirm", json={"session_id": "s", "approved": True}).status_code == 409


def test_llm_failure_returns_502(make_client):
    c = make_client()

    def explode(messages, tools):
        raise RuntimeError("provider down")

    c.app.state.runtime.set_provider(__import__("conftest").fake_provider(explode))
    r = c.post("/chat", json={"message": "x", "session_id": "s"})
    assert r.status_code == 502 and "provider down" in r.json()["error"]


# --------------------------------------------------------------------------- #
def test_stream_events_for_a_turn(make_client):
    c = make_client()
    push_fake(c, route("hmi"), call("open_item", "Opening now", itemType="Document", itemId="1"), "All done", route("FINISH"))
    with c.websocket_connect("/stream?session_id=s1") as ws:
        assert ws.receive_json()["event_type"] == "stream.connected"
        c.post("/chat", json={"message": "open", "session_id": "s1"})
        events = []
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["event_type"] == "agent.end" and ev["data"]["agent_name"] == "supervisor" and len(events) > 10:
                break
    types = [e["event_type"] for e in events]
    for t in ("agent.start", "llm.start", "llm.stream", "llm.end", "tool.start", "tool.end", "agent.end"):
        assert t in types
    tool_start = next(e for e in events if e["event_type"] == "tool.start")
    assert tool_start["data"] == {"agent_name": "hmi", "tool_name": "open_item", "tool_args": {"itemType": "Document", "itemId": "1"}}
    assert next(e for e in events if e["event_type"] == "tool.end")["data"]["status"] == "ok"
    tokens = "".join(e["data"]["token"] for e in events if e["event_type"] == "llm.stream")
    assert "Opening now" in tokens and "All done" in tokens
    assert all(e["session_id"] == "s1" and e["timestamp"].endswith("Z") for e in events)
    routed = [e["data"]["agent_name"] for e in events if e["event_type"] == "agent.start"]
    assert routed[:2] == ["supervisor", "hmi"]


def test_stream_confirmation_over_websocket(make_client, backend):
    c = make_client()
    push_fake(c, route("jobs"), call("run_job", jobId="J"), "started J", route("FINISH"))
    with c.websocket_connect("/stream") as ws:  # all sessions
        ws.receive_json()
        c.post("/chat", json={"message": "run", "session_id": "s2"})
        while (ev := ws.receive_json())["event_type"] != "confirmation.needed":
            pass
        assert ev["data"]["tool_name"] == "run_job" and ev["session_id"] == "s2"
        ws.send_json({"type": "confirm", "session_id": "s2", "approved": True})
        while (ev := ws.receive_json())["event_type"] != "chat.result":
            pass
        assert ev["data"]["response"] == "started J"
        ws.send_json({"type": "confirm", "session_id": "s2", "approved": True})
        while (ev := ws.receive_json())["event_type"] != "error":
            pass
        assert "confirmation failed" in ev["data"]["message"]
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["event_type"] == "pong"
        ws.send_json(["not", "a", "dict"])
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["event_type"] == "pong"
    assert backend.calls() == [("POST", "/api/jobs/J/run")]


def test_stream_only_receives_its_session(make_client):
    c = make_client()
    push_fake(c, route("FINISH"), route("FINISH"))
    with c.websocket_connect("/stream?session_id=mine") as ws:
        ws.receive_json()
        c.post("/chat", json={"message": "x", "session_id": "other"})
        c.post("/chat", json={"message": "x", "session_id": "mine"})
        assert ws.receive_json()["session_id"] == "mine"
    assert c.get("/status").json()["stream_subscribers"] == 0


def test_stream_invalid_json_closes(make_client):
    c = make_client()
    with c.websocket_connect("/stream") as ws:
        ws.receive_json()
        ws.send_text("{not json")
        with pytest.raises(Exception):
            ws.receive_json()


# --------------------------------------------------------------------------- #
def test_reload_endpoints_and_invalid_config_keeps_previous(make_client):
    c = make_client()
    push_fake(c, "x")
    cfg = c.app.state.runtime.config_dir / "agents.json"
    data = json.loads(cfg.read_text())
    data["agents"]["new_agent"] = agent("new agent", ["open_item"])
    cfg.write_text(json.dumps(data))
    r = c.post("/agent/reload")
    assert r.status_code == 200 and "new_agent" in r.json()["agents"]
    cfg.write_text(json.dumps({"agents": {"bad": {"description": "x", "system_prompt": "y", "allowed_tools": []}}}))
    r = c.post("/reload")
    assert r.status_code == 422 and any("at least one tool" in e for e in r.json()["errors"])
    assert "new_agent" in [a["name"] for a in c.get("/status").json()["agents"]]


def test_unknown_provider_reference_is_rejected(make_client):
    c = make_client(agents={"a": agent("a", ["open_item"], provider="missing")})
    r = c.post("/config/provider", json={"type": "fake"})
    assert r.status_code == 422 and "unknown providers" in r.json()["errors"][0]
    assert c.get("/status").json()["provider_configured"] is False


def test_provider_endpoints(make_client):
    c = make_client()
    assert c.get("/config/provider").json() == {"configured": False, "models": {}}
    assert c.get("/config/provider/test").json()["ok"] is False
    r = c.post("/config/provider", json={"type": "anthropic"})
    assert r.status_code == 422
    payload = {
        "default": "main",
        "providers": {
            "main": {"type": "fake", "models": ["m1", "m2"]},
            "cloud": {"type": "anthropic", "api_key": "sk-very-secret", "default_model": "m"},
        },
    }
    assert c.post("/config/provider", json=payload).status_code == 200
    got = c.get("/config/provider").json()
    assert "sk-very-secret" not in json.dumps(got)
    assert got["models"] == {"main": ["m1", "m2"], "cloud": ["m"]}
    t = c.get("/config/provider/test").json()
    assert t["ok"] and t["provider"] == "main" and t["reply"].startswith("echo")
    assert c.get("/config/provider/test?provider=ghost").json()["ok"] is False


def test_proxy_and_paths_endpoints(make_client, tmp_path):
    c = make_client()
    r = c.post("/config/proxy", json={"https_proxy": "http://proxy:3128", "no_proxy": "a.com,b.com"})
    assert r.json() == {"status": "configured", "http_proxy": False, "https_proxy": True, "no_proxy": ["a.com", "b.com"]}
    assert c.post("/config/proxy", json={"no_proxy": 5}).status_code == 422
    (tmp_path / "inbox").mkdir()
    r = c.post("/config/paths", json={"paths": {"inbox": str(tmp_path / "inbox"), "outbox": str(tmp_path / "missing")}}).json()
    assert r["missing"] == ["outbox"]
    assert set(c.get("/config/paths").json()["paths"]) == {"inbox", "outbox"}


def test_tools_and_agents_listing(make_client):
    c = make_client()
    tools = {t["name"]: t for t in c.get("/tools").json()["tools"]}
    assert {"open_item", "get_job_status", "run_job", "delete_thing", "wait_for_event", "list_shared_files"} == set(tools)
    assert tools["get_job_status"]["read_only"] is True
    agents = c.get("/agents").json()
    assert agents["agents"]["jobs"]["requires_confirmation"] == ["run_job"]


def test_shutdown(make_client):
    c = make_client()
    assert c.post("/shutdown").status_code == 501

    class FakeServer:
        should_exit = False

    c.app.state.server = FakeServer()
    assert c.post("/shutdown").json() == {"status": "shutting_down"}
    assert c.app.state.server.should_exit


def test_openapi_lists_spec_endpoints(make_client):
    paths = make_client().get("/openapi.json").json()["paths"]
    for p in ("/chat", "/confirm", "/reload", "/agent/reload", "/status", "/config/provider", "/config/provider/test", "/config/proxy", "/config/paths"):
        assert p in paths
