"""End-to-end live test over real HTTP, as a host application would drive the server.

* the server runs in its own process (``python -m langgraph_json``), with no
  LLM credentials in its environment: the key is only sent through the API;
* the backend is a real HTTP server (a thread of this test);
* the client is plain ``httpx`` + a WebSocket client: no in-process shortcut.

Run with the other live tests:  ``pytest -m live -v``
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from conftest import TOOLS, agent, write_config
from test_live import PROVIDER
from test_main import start_server

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(PROVIDER is None, reason="no LLM credentials (set ANTHROPIC_API_KEY or LGJ_TEST_PROVIDER_JSON, e.g. in .env)"),
]

SECRET_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "LGJ_TEST_PROVIDER_JSON", "LGJ_PROVIDER_JSON", "OPENAI_API_KEY")


class BackendHandler(BaseHTTPRequestHandler):
    calls: list[tuple[str, str]] = []

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self.calls.append(("GET", self.path))
        if self.path == "/api/jobs/J7":
            self._reply(200, {"jobId": "J7", "state": "done", "progress": 100})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.calls.append(("POST", self.path))
        self._reply(200, {"ok": True})

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def http_backend():
    BackendHandler.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", BackendHandler.calls
    server.shutdown()


@pytest.fixture
def server(tmp_path, http_backend):
    backend_url, _ = http_backend
    cfg = write_config(
        tmp_path / "config",
        {"hmi": agent("the HMI agent: opens items (documents, projects)", ["open_item"]),
         "jobs": agent("the jobs agent: checks the status of computation jobs", ["get_job_status"])},
        TOOLS,
        {"backend_url": backend_url},
        supervisor={"system_prompt": "Opening items -> hmi; anything about jobs -> jobs. FINISH when answered."},
    )
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}  # the key only travels through the API
    proc, url, status = start_server([sys.executable, "-m", "langgraph_json", "--config-dir", str(cfg)], env)
    assert status["status"] == "waiting_provider"
    yield url
    try:
        httpx.post(url + "/shutdown", timeout=10)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:  # pragma: no cover — only if the shutdown failed
            proc.kill()


def test_key_sent_by_api_then_prompt(server, http_backend):
    url = server
    _, calls = http_backend

    # 1. A wrong key is accepted as configuration but fails the connectivity test.
    bad = {**PROVIDER, "api_key": "sk-ant-invalid-key-for-test"} if PROVIDER.get("type") == "anthropic" else None
    if bad:
        assert httpx.post(url + "/config/provider", json=bad).status_code == 200
        r = httpx.get(url + "/config/provider/test", timeout=60).json()
        assert r["ok"] is False
        assert "401" in r["error"] or "authentication" in r["error"].lower(), r

    # 2. The real key: authentication works, and the key never comes back.
    assert httpx.post(url + "/config/provider", json=PROVIDER).status_code == 200
    r = httpx.get(url + "/config/provider/test", timeout=60).json()
    assert r["ok"] is True and "pong" in r["reply"].lower(), r
    shown = httpx.get(url + "/config/provider").text + httpx.get(url + "/status").text
    if PROVIDER.get("api_key"):
        assert PROVIDER["api_key"] not in shown

    # 3. A prompt through the API, with the events followed on the WebSocket.
    from websockets.sync.client import connect

    ws_url = url.replace("http://", "ws://") + "/stream?session_id=http-1"
    with connect(ws_url, open_timeout=10) as ws:
        r = httpx.post(url + "/chat", json={"message": "What is the status of job J7?", "session_id": "http-1"}, timeout=180).json()
        assert r["status"] == "completed", r
        events = []
        while True:  # read until the end of the turn
            ev = json.loads(ws.recv(timeout=30))
            events.append(ev)
            if ev["event_type"] == "agent.end" and ev["data"].get("agent_name") == "supervisor" and len(events) > 5:
                break
    assert "jobs" in r["route"]
    assert ("GET", "/api/jobs/J7") in calls  # the real backend received the REST call
    assert "done" in r["response"].lower() or "complete" in r["response"].lower() or "100" in r["response"]
    assert r["usage"]["tokens_used"] > 0
    types = {e["event_type"] for e in events}
    assert {"agent.start", "tool.start", "tool.end", "llm.end"} <= types

    # 4. The history is available through the API.
    hist = httpx.get(url + "/sessions/http-1").json()
    assert any(m["type"] == "human" for m in hist["messages"])
