"""Live tests against a real LLM (full stack: HTTP API, supervisor, REST and
Python tools, confirmation, streaming, memory).

Skipped unless credentials are available, from the environment or ``.env``:

* ``ANTHROPIC_API_KEY`` (+ optional ``LGJ_TEST_MODEL``, ``ANTHROPIC_BASE_URL``), or
* ``LGJ_TEST_PROVIDER_JSON`` — any ``POST /config/provider`` body.

Run them only:  ``pytest -m live -v``

The backend is still mocked: only the LLM is real. Assertions check the
observable behaviour (tools called, arguments, routing), not exact wording.
"""

import json
import os

import pytest
from conftest import TOOLS, agent, write_config
from fastapi.testclient import TestClient

from langgraph_json.gateway import Runtime, create_app

ROOT_EXAMPLE = __import__("conftest").ROOT / "examples" / "python"
DEFAULT_MODEL = "claude-haiku-5-5"


def provider_payload() -> dict | None:
    raw = os.environ.get("LGJ_TEST_PROVIDER_JSON")
    if raw:
        return json.loads(raw)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    payload = {
        "type": "anthropic",
        "api_key": key,
        "default_model": os.environ.get("LGJ_TEST_MODEL", DEFAULT_MODEL),
        "temperature": 0,
        "max_tokens": 1024,
    }
    if os.environ.get("ANTHROPIC_BASE_URL"):
        payload["endpoint"] = os.environ["ANTHROPIC_BASE_URL"]
    return payload


PROVIDER = provider_payload()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(PROVIDER is None, reason="no LLM credentials (set ANTHROPIC_API_KEY or LGJ_TEST_PROVIDER_JSON, e.g. in .env)"),
]

AGENTS = {
    "hmi": agent(
        "the HMI agent: opens items (documents, projects, datasets) in viewer windows",
        ["open_item"],
    ),
    "jobs": agent(
        "the jobs agent: checks the status of computation jobs and runs jobs",
        ["get_job_status", "run_job"],
        requires_confirmation=["run_job"],
    ),
    "utils": agent(
        "the utilities agent: converts distances between units",
        ["convert_distance"],
    ),
}
SUPERVISOR = {
    "system_prompt": "You route requests: opening items -> hmi; job status or running jobs -> jobs; "
    "unit conversions -> utils. Route one step at a time; FINISH when the request is fully answered."
}
PY_TOOLS = [{"name": "convert_distance", "kind": "python", "callable": "my_tools.units:convert_distance",
             "description": "Converts a distance between units (m, km, ft, nm)."}]


@pytest.fixture
def client(backend, tmp_path):
    backend.responses[("GET", "/api/jobs/J7")] = {"jobId": "J7", "state": "done", "progress": 100}
    backend.responses[("GET", "/api/jobs/J1")] = {"jobId": "J1", "state": "running", "progress": 40}
    cfg = write_config(
        tmp_path / "config",
        AGENTS,
        TOOLS + PY_TOOLS,
        {"python_tool_modules": ["my_tools"], "plugins_dir": str(ROOT_EXAMPLE), "token_budget_per_session": 200_000},
        supervisor=SUPERVISOR,
    )
    with TestClient(create_app(Runtime(cfg, backend_transport=backend.transport))) as c:
        r = c.post("/config/provider", json=PROVIDER)
        assert r.status_code == 200, r.text
        yield c


def chat(client, message: str, session: str, **kw) -> dict:
    r = client.post("/chat", json={"message": message, "session_id": session, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_provider_connectivity(client):
    r = client.get("/config/provider/test").json()
    assert r["ok"], r
    assert "pong" in r["reply"].lower()


def test_supervisor_routes_and_calls_rest_tool(client, backend):
    r = chat(client, "Open the document with id doc-42.", "live-1")
    assert r["status"] == "completed", r
    assert "hmi" in r["route"]
    req = next(q for q in backend.requests if q.url.path == "/api/items/open")
    body = json.loads(req.content)
    assert body["itemType"] == "Document" and body["itemId"] == "doc-42"
    assert r["response"].strip()
    assert r["usage"]["tokens_used"] > 0


def test_confirmation_denied_then_approved(client, backend):
    r = chat(client, "Run job J7 now.", "live-2")
    assert r["status"] == "confirmation_needed", r
    assert r["confirmation"]["tool_name"] == "run_job" and r["confirmation"]["tool_args"]["jobId"] == "J7"
    r = client.post("/confirm", json={"session_id": "live-2", "approved": False, "reason": "not now"}).json()
    assert not any(q.url.path == "/api/jobs/J7/run" for q in backend.requests)
    assert r["status"] in ("completed", "confirmation_needed")
    if r["status"] == "confirmation_needed":  # the model may retry once: deny again
        r = client.post("/confirm", json={"session_id": "live-2", "approved": False}).json()
    assert r["status"] == "completed" and r["response"].strip()

    r = chat(client, "OK, I changed my mind: run job J7.", "live-2")
    assert r["status"] == "confirmation_needed", r
    r = client.post("/confirm", json={"session_id": "live-2", "approved": True}).json()
    assert r["status"] == "completed", r
    assert any(q.method == "POST" and q.url.path == "/api/jobs/J7/run" for q in backend.requests)


def test_python_tool(client):
    r = chat(client, "Convert 3 nautical miles to meters.", "live-3")
    assert r["status"] == "completed", r
    conv = [t for t in r["tool_calls"] if t["tool_name"] == "convert_distance"]
    assert conv and conv[0]["status"] == "ok"
    assert json.loads(conv[0]["result"])["value"] == 5556.0
    assert "5556" in r["response"].replace(",", "").replace(" ", "").replace(" ", "")


def test_multi_agent_chain(client, backend):
    r = chat(client, "First open the project with id P-7, then tell me the status of job J7.", "live-4")
    assert r["status"] == "completed", r
    assert {"hmi", "jobs"} <= set(r["route"])
    paths = [q.url.path for q in backend.requests]
    assert "/api/items/open" in paths and "/api/jobs/J7" in paths
    assert "done" in r["response"].lower() or "complete" in r["response"].lower()


def test_multi_turn_memory(client):
    chat(client, "Remember this code word for later: PINEAPPLE-93. Just acknowledge.", "live-5")
    r = chat(client, "What code word did I give you earlier? Answer with the code word only.", "live-5")
    assert "PINEAPPLE-93" in r["response"].upper()
    other = chat(client, "What code word did I give you earlier? If none, say NONE.", "live-5-other")
    assert "PINEAPPLE-93" not in other["response"].upper()


def test_direct_agent(client, backend):
    r = chat(client, "What is the status of job J1?", "live-6", agent="jobs")
    assert r["route"] == ["jobs"] and r["agent"] == "jobs"
    assert ("GET", "/api/jobs/J1") in backend.calls()
    assert "40" in r["response"] or "running" in r["response"].lower()


def test_streaming_events(client):
    with client.websocket_connect("/stream?session_id=live-7") as ws:
        ws.receive_json()
        r = chat(client, "Convert 10 km to meters.", "live-7")
        assert r["status"] == "completed"
        events = []
        while not any(e["event_type"] == "agent.end" and e["data"]["agent_name"] == "supervisor" for e in events[-1:]) or len(events) < 5:
            events.append(ws.receive_json())
            if len(events) > 2000:
                break
    types = {e["event_type"] for e in events}
    assert {"agent.start", "llm.start", "llm.end", "tool.start", "tool.end"} <= types
    usage = [e["data"]["usage"] for e in events if e["event_type"] == "llm.end" and e["data"]["usage"]]
    assert usage and usage[0]["total_tokens"] > 0
