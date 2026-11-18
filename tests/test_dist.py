"""Tests of the compiled PyInstaller bundle (dist/lang-graph-json/).

Skipped when the bundle is not built:  python scripts/build.py exe
"""

import json
import os
import subprocess
import time

import httpx
import pytest
from conftest import ROOT, TOOLS, agent, write_config
from test_main import run_server_scenario, start_server

BUNDLE = ROOT / "dist" / "lang-graph-json"
EXE = BUNDLE / ("lang-graph-json.exe" if os.name == "nt" else "lang-graph-json")

pytestmark = [pytest.mark.dist, pytest.mark.skipif(not EXE.exists(), reason="bundle not built (python scripts/build.py exe)")]


def test_layout():
    assert (BUNDLE / "_internal").is_dir()
    for name in ("agents.json", "tools.json", "server.json"):
        assert (BUNDLE / "config" / name).is_file()
    assert json.loads((BUNDLE / "config" / "server.json").read_text())["plugins_dir"] == "plugins"
    assert (BUNDLE / "plugins" / "README.txt").is_file()
    assert (BUNDLE / "generate_tools_json.py").is_file()


def test_version():
    out = subprocess.run([str(EXE), "--version"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "lang-graph-json" in out.stdout


def test_bundle_runs_with_its_own_config():
    """No --config-dir: the bundle uses config/ next to the executable."""
    status = run_server_scenario([str(EXE)])
    assert status["agents_active"] == 3 and status["tools"] == 18


def test_bundle_loads_python_plugins_and_reloads(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "data.csv").write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
    tools = TOOLS + [
        {"name": "csv_summary", "kind": "python", "callable": "my_tools.files:csv_summary", "description": "CSV summary",
         "parameters": {"folder": {"type": "string", "required": True}, "file": {"type": "string", "required": True}}},
        {"name": "convert_distance", "kind": "python", "callable": "my_tools.units:convert_distance", "description": "convert"},
    ]
    cfg = write_config(
        tmp_path / "config",
        {"files": agent("files agent", ["csv_summary", "convert_distance"])},
        tools,
        {"python_tool_modules": ["my_tools"], "plugins_dir": str(ROOT / "examples" / "python"), "watch_interval_s": 0},
    )
    proc, url, status = start_server([str(EXE), "--config-dir", str(cfg)])
    try:
        assert status["status"] == "waiting_provider" and status["tools"] == len(tools) + 2, status
        httpx.post(url + "/config/paths", json={"paths": {"inbox": str(inbox)}})
        httpx.post(url + "/config/provider", json={"type": "fake", "fake_responses": [
            {"tool_calls": [{"name": "csv_summary", "args": {"folder": "inbox", "file": "data.csv"}},
                            {"name": "convert_distance", "args": {"value": 2, "from_unit": "km"}}]},
            "2 rows; 2000 m",
        ]})
        r = httpx.post(url + "/chat", json={"message": "summarise", "session_id": "p"}, timeout=60).json()
        assert r["status"] == "completed", r
        results = {t["tool_name"]: json.loads(t["result"]) for t in r["tool_calls"]}
        assert results["csv_summary"]["rows"] == 2 and results["csv_summary"]["columns"] == ["id", "value"]
        assert results["convert_distance"] == {"value": 2000.0, "unit": "m"}
        # hot-reload through the API still works in the frozen executable
        assert httpx.post(url + "/reload").status_code == 200
        httpx.post(url + "/shutdown")
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_bundle_refuses_remote_bind():
    out = subprocess.run([str(EXE), "--host", "0.0.0.0"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 2 and "non-loopback" in out.stderr


def test_bundled_parser_is_standalone(tmp_path):
    import sys

    out = tmp_path / "tools.json"
    r = subprocess.run([sys.executable, "-I", str(BUNDLE / "generate_tools_json.py"), "--source-dir", str(ROOT / "examples" / "cpp"), "--output", str(out)],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert out.read_text() == (ROOT / "config" / "tools.json").read_text()


def _scenario(url, message, session):
    httpx.post(url + "/config/provider", json={"type": "fake"})
    return httpx.post(url + "/chat", json={"message": message, "session_id": session}, timeout=60).json()


def test_bundle_sqlite_persistence_across_restart(tmp_path):
    cfg = write_config(tmp_path / "config", {"solo": agent("solo", ["open_item"])}, None, {"persistence": "sqlite"})
    for i, message in enumerate(["first message", "second message"]):
        proc, url, status = start_server([str(EXE), "--config-dir", str(cfg)])
        try:
            assert status["persistence"] == "sqlite"
            assert status["sessions"] == i  # the session of the first run is restored
            r = _scenario(url, message, "kept")
            assert r["status"] == "completed", r
            hist = httpx.get(url + "/sessions/kept").json()["messages"]
            assert [m["content"] for m in hist if m["type"] == "human"] == ["first message", "second message"][: i + 1]
            httpx.post(url + "/shutdown")
            assert proc.wait(timeout=30) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert (tmp_path / "data" / "sessions.db").exists()
