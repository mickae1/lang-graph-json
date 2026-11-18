from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from langgraph_json.gateway.runtime import Runtime
from langgraph_json.utils.llm_factory import ProviderConfig

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides the environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


# Live tests (tests/test_live.py) read their credentials from .env or the environment.
load_dotenv(ROOT / ".env")

TOOLS = [
    {
        "name": "open_item",
        "method": "POST",
        "path": "/api/items/open",
        "domain": "hmi",
        "description": "Open an item",
        "parameters": {
            "itemType": {"type": "string", "required": True, "enum": ["Document", "Project"]},
            "itemId": {"type": "string", "required": True},
            "tab": {"type": "string"},
        },
    },
    {
        "name": "get_job_status",
        "method": "GET",
        "path": "/api/jobs/{jobId}",
        "domain": "jobs",
        "description": "Job status",
        "parameters": {"jobId": {"type": "string", "required": True}, "verbose": {"type": "boolean"}},
    },
    {
        "name": "run_job",
        "method": "POST",
        "path": "/api/jobs/{jobId}/run",
        "domain": "jobs",
        "description": "Run a job",
        "parameters": {"jobId": {"type": "string", "required": True}},
    },
    {
        "name": "delete_thing",
        "method": "DELETE",
        "path": "/api/thing/{id}",
        "domain": "analysis",
        "description": "Delete a thing",
        "parameters": {"id": {"type": "string", "required": True}},
    },
]


def agent(description: str, tools: list[str], **kw: Any) -> dict:
    return {"description": description, "system_prompt": f"You are {description}.", "allowed_tools": tools, **kw}


DEFAULT_AGENTS = {
    "hmi": agent("the HMI agent", ["open_item"]),
    "jobs": agent("the jobs agent", ["get_job_status", "run_job"], requires_confirmation=["run_job"]),
}


class Backend:
    """Mock backend REST API recording every request."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.responses: dict[tuple[str, str], Any] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key in self.responses:
            r = self.responses[key]
            return r if isinstance(r, httpx.Response) else httpx.Response(200, json=r)
        body = json.loads(request.content) if request.content else None
        return httpx.Response(200, json={"ok": True, "path": request.url.path, "query": dict(request.url.params), "body": body})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]


def write_config(path: Path, agents: dict | None = None, tools: list | None = None, server: dict | None = None, **agents_extra) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "agents.json").write_text(json.dumps({"agents": agents if agents is not None else DEFAULT_AGENTS, **agents_extra}), encoding="utf-8")
    (path / "tools.json").write_text(json.dumps({"tools": tools if tools is not None else TOOLS}), encoding="utf-8")
    (path / "server.json").write_text(json.dumps({"watch_interval_s": 0, **(server or {})}), encoding="utf-8")
    return path


def route(name: str, instructions: str | None = None) -> dict:
    args = {"next": name}
    if instructions:
        args["instructions"] = instructions
    return {"tool_calls": [{"name": "route", "args": args}]}


def call(name: str, content: str = "", **args) -> dict:
    return {"content": content, "tool_calls": [{"name": name, "args": args}]}


def fake_provider(*responses) -> ProviderConfig:
    return ProviderConfig.model_validate({"type": "fake", "fake_responses": list(responses)})


@pytest.fixture
def backend() -> Backend:
    return Backend()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return write_config(tmp_path / "config")


@pytest.fixture
async def make_runtime(backend: Backend, tmp_path: Path):
    """Factory: make_runtime(agents=..., responses=[...], server={...})."""
    created: list[Runtime] = []

    async def _make(agents: dict | None = None, responses: list | None = None, server: dict | None = None, tools: list | None = None, **extra) -> Runtime:
        cfg = write_config(tmp_path / f"config{len(created)}", agents, tools, server, **extra)
        rt = Runtime(cfg, backend_transport=backend.transport)
        await rt.startup()
        if responses is not None:
            rt.set_provider(fake_provider(*responses))
        created.append(rt)
        return rt

    yield _make
    for rt in created:
        await rt.shutdown()


def fake_llm(rt: Runtime):
    """The (shared) scripted model of a runtime using the fake provider."""
    return rt.llms.get()
