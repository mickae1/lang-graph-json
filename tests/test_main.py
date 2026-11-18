"""CLI entry point, including a real server started in a subprocess."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from conftest import ROOT, write_config

from langgraph_json import __main__ as cli


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_is_loopback():
    assert cli.is_loopback("127.0.0.1") and cli.is_loopback("localhost") and cli.is_loopback("::1")
    assert not cli.is_loopback("0.0.0.0") and not cli.is_loopback("192.168.1.10") and not cli.is_loopback("example.com")


def test_default_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LGJ_CONFIG_DIR", str(tmp_path))
    assert cli.default_config_dir() == tmp_path
    monkeypatch.delenv("LGJ_CONFIG_DIR")
    assert cli.default_config_dir() == Path.cwd() / "config"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "app" / "lang-graph-json.exe"))
    assert cli.default_config_dir() == (tmp_path / "app").resolve() / "config"


def test_parse_args():
    a = cli.parse_args(["--port", "9001", "--host", "localhost", "--config-dir", "x", "--log-level", "debug"])
    assert (a.port, a.host, a.config_dir, a.log_level, a.allow_remote) == (9001, "localhost", Path("x"), "debug", False)


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["--version"])
    assert "lang-graph-json" in capsys.readouterr().out


def test_refuses_remote_bind(tmp_path, capsys):
    cfg = write_config(tmp_path)
    assert cli.main(["--host", "0.0.0.0", "--config-dir", str(cfg)]) == 2
    assert "non-loopback" in capsys.readouterr().err


def test_invalid_server_json(tmp_path, capsys):
    cfg = write_config(tmp_path, server={"port": "not-a-port"})
    assert cli.main(["--config-dir", str(cfg)]) == 2
    assert "server.json" in capsys.readouterr().err


def test_main_runs_uvicorn(monkeypatch, tmp_path):
    """main() builds the app and runs uvicorn with the resolved settings."""
    import uvicorn

    seen = {}

    def fake_run(self):
        seen["host"], seen["port"] = self.config.host, self.config.port
        seen["app"] = self.config.app

    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    cfg = write_config(tmp_path, server={"port": 8123})
    assert cli.main(["--config-dir", str(cfg)]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8123
    assert seen["app"].state.server is not None
    assert cli.main(["--config-dir", str(cfg), "--port", "8124", "--host", "localhost"]) == 0
    assert (seen["host"], seen["port"]) == ("localhost", 8124)


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"server exited early (code {proc.returncode}): {proc.stdout.read()}")
        try:
            return httpx.get(url + "/status", timeout=1).json()
        except httpx.HTTPError:
            time.sleep(0.3)
    raise AssertionError("server did not start")


def start_server(cmd: list[str], env: dict | None = None, attempts: int = 3):
    """Start a server process on a free port (retry on a port collision)."""
    for attempt in range(attempts):
        port = free_port()
        proc = subprocess.Popen([*cmd, "--port", str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        url = f"http://127.0.0.1:{port}"
        try:
            return proc, url, wait_ready(url, proc)
        except AssertionError as exc:
            collision = "10048" in str(exc) or "address already in use" in str(exc).lower()
            if not collision or attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def run_server_scenario(cmd: list[str], env: dict | None = None) -> dict:
    """Start a server process, run a scripted conversation, shut it down cleanly."""
    proc, url, status = start_server(cmd, env)
    try:
        assert status["status"] == "waiting_provider"
        r = httpx.post(url + "/config/provider", json={"type": "fake", "fake_responses": [
            {"tool_calls": [{"name": "route", "args": {"next": "hmi"}}]},
            "hello from the hmi agent",
            {"tool_calls": [{"name": "route", "args": {"next": "FINISH"}}]},
        ]})
        assert r.status_code == 200, r.text
        r = httpx.post(url + "/chat", json={"message": "hi", "session_id": "smoke"}, timeout=60).json()
        assert r["status"] == "completed" and r["response"] == "hello from the hmi agent", r
        assert httpx.post(url + "/shutdown").json()["status"] == "shutting_down"
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return status


def test_real_server_subprocess():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    status = run_server_scenario([sys.executable, "-m", "langgraph_json", "--config-dir", str(ROOT / "config")], env=env)
    assert status["agents_active"] == 3 and status["tools"] == 18
