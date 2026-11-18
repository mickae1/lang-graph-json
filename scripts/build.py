#!/usr/bin/env python3
"""Build the deliverables.

    python scripts/build.py exe      # PyInstaller --onedir bundle in dist/lang-graph-json/ + smoke test
    python scripts/build.py wheel    # wheel + sdist in dist/, installed in a fresh venv + smoke test
    python scripts/build.py all      # both (default)

Options: --skip-smoke (build only), --zip (also archive the bundle).

Bundle layout (dist/lang-graph-json/):

    lang-graph-json(.exe)   executable launched by the host application
    _internal/              Python runtime + dependencies (do not edit)
    config/                 agents.json, tools.json, server.json (shared volume)
    plugins/                optional Python tools (kind "python"), see README there
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUNDLE = DIST / "lang-graph-json"
EXE = BUNDLE / ("lang-graph-json.exe" if os.name == "nt" else "lang-graph-json")

PLUGINS_README = """Python tools (kind "python") loaded at startup and on every hot-reload.

1. Put a package here, e.g. plugins/my_tools/__init__.py + plugins/my_tools/stats.py
2. Allow it in config/server.json:   "python_tool_modules": ["my_tools"]
3. Declare the tools in config/tools.json:
     {"name": "csv_summary", "kind": "python", "callable": "my_tools.stats:csv_summary", ...}
   or generate them:  generate_tools_json.py --source-dir plugins --lang python --output config/tools.json

Plugins may use the standard library and every library bundled with the server
(langchain, httpx, pydantic...). Other third-party packages must be embedded at
build time (LGJ_BUNDLE_MODULES / LGJ_HIDDEN_IMPORTS, see lang-graph-json.spec).
"""


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, body: dict | None = None, timeout: float = 30) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — localhost only
        return json.loads(r.read())


def start_server(cmd: list[str], env: dict | None = None, attempts: int = 3):
    """Start the server on a free port; retry on a port collision."""
    for attempt in range(attempts):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen([*cmd, "--port", str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 120
        while True:
            if proc.poll() is not None:
                output = proc.stdout.read()
                if ("10048" in output or "address already in use" in output.lower()) and attempt < attempts - 1:
                    break  # port taken meanwhile: try another one
                raise SystemExit(f"server exited early:\n{output}")
            try:
                return proc, url, http("GET", url + "/status", timeout=1)
            except OSError:
                if time.time() > deadline:
                    proc.kill()
                    raise SystemExit("server did not start")
                time.sleep(0.3)
    raise SystemExit("could not start the server")


def smoke_test_server(cmd: list[str], env: dict | None = None) -> None:
    """Start the server, run a scripted conversation (fake LLM), stop it."""
    proc, url, status = start_server(cmd, env)
    try:
        print(f"  status: {status['status']}, agents {status['agents_active']}/{status['agents_total']}, tools {status['tools']}")
        http("POST", url + "/config/provider", {"type": "fake", "fake_responses": [
            {"tool_calls": [{"name": "route", "args": {"next": "hmi"}}]},
            "hello from the bundled server",
            {"tool_calls": [{"name": "route", "args": {"next": "FINISH"}}]},
        ]})
        r = http("POST", url + "/chat", {"message": "hi", "session_id": "smoke"})
        assert r["status"] == "completed" and r["response"] == "hello from the bundled server", r
        print(f"  chat: [{r['agent']}] {r['response']}")
        http("POST", url + "/shutdown")
        code = proc.wait(timeout=30)
        assert code == 0, f"exit code {code}"
        print("  smoke test OK")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# --------------------------------------------------------------------------- #
def build_exe(smoke: bool, make_zip: bool) -> None:
    shutil.rmtree(BUNDLE, ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(DIST),
         "--workpath", str(ROOT / "build" / "pyinstaller"), str(ROOT / "lang-graph-json.spec")], cwd=ROOT)

    shutil.copytree(ROOT / "config", BUNDLE / "config")
    server = json.loads((BUNDLE / "config" / "server.json").read_text(encoding="utf-8"))
    server["plugins_dir"] = "plugins"
    (BUNDLE / "config" / "server.json").write_text(json.dumps(server, indent=2) + "\n", encoding="utf-8")
    (BUNDLE / "plugins").mkdir()
    (BUNDLE / "plugins" / "README.txt").write_text(PLUGINS_README, encoding="utf-8")
    # Standalone copy of the (stdlib-only) parser, usable with any Python.
    (BUNDLE / "generate_tools_json.py").write_text(
        (ROOT / "langgraph_json" / "tools_parser.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    size = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file()) / 1e6
    print(f"bundle: {BUNDLE} ({size:.0f} MB)")

    if smoke:
        print("smoke test of the bundle:")
        smoke_test_server([str(EXE)])
    if make_zip:
        archive = shutil.make_archive(str(DIST / f"lang-graph-json-{sys.platform}"), "zip", DIST, "lang-graph-json")
        print(f"archive: {archive}")


def build_wheel(smoke: bool) -> None:
    for f in DIST.glob("lang_graph_json-*"):
        f.unlink()
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(DIST), str(ROOT)])
    wheel = next(DIST.glob("lang_graph_json-*.whl"))
    print(f"wheel: {wheel}")
    if not smoke:
        return
    print("smoke test of the wheel in a fresh virtual environment:")
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        run([sys.executable, "-m", "venv", str(venv)])
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(py), "-m", "pip", "install", "-q", str(wheel)])
        # library mode, run from a neutral directory (no access to the sources)
        run([str(py), str(ROOT / "examples" / "embedded" / "inprocess.py")], cwd=tmp,
            env={k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "PYTHONPATH")})
        # console scripts
        cli = venv / ("Scripts/lang-graph-json.exe" if os.name == "nt" else "bin/lang-graph-json")
        smoke_test_server([str(cli), "--config-dir", str(ROOT / "config")])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="all", choices=["exe", "wheel", "all"])
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    DIST.mkdir(exist_ok=True)
    if args.target in ("wheel", "all"):
        build_wheel(not args.skip_smoke)
    if args.target in ("exe", "all"):
        build_exe(not args.skip_smoke, args.zip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
