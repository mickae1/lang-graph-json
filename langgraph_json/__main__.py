"""Command line entry point: ``lang-graph-json --port 8000 --config-dir config``.

The host application launches this executable as a subprocess, waits for
``GET /status``, then pushes the provider / proxy configuration.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
import sys
from pathlib import Path

from . import __version__


def default_config_dir() -> Path:
    env = os.environ.get("LGJ_CONFIG_DIR")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):  # PyInstaller bundle: config/ next to the executable
        return Path(sys.executable).resolve().parent / "config"
    return Path.cwd() / "config"


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lang-graph-json", description="JSON-driven LangGraph multi-agent server")
    p.add_argument("--host", default=None, help="bind address (default from server.json, 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="port (default from server.json, 8000)")
    p.add_argument("--config-dir", type=Path, default=None, help="directory holding agents.json / tools.json / server.json")
    p.add_argument("--log-level", default=None, choices=["critical", "error", "warning", "info", "debug"])
    p.add_argument("--allow-remote", action="store_true", help="allow binding a non-loopback address (not recommended)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_dir = (args.config_dir or default_config_dir()).resolve()

    from .utils.config_loader import ConfigError, SERVER_FILE, load_settings

    try:
        settings = load_settings(config_dir / SERVER_FILE)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    host = args.host or settings.host
    port = args.port or settings.port
    level = args.log_level or settings.log_level
    if not is_loopback(host) and not args.allow_remote:
        print(f"error: refusing to bind non-loopback address {host} (use --allow-remote)", file=sys.stderr)
        return 2

    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from .gateway import Runtime, create_app

    app = create_app(Runtime(config_dir))
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level=level, ws="websockets-sansio", lifespan="on"))
    app.state.server = server
    logging.getLogger(__name__).info("lang-graph-json %s — config %s — http://%s:%s", __version__, config_dir, host, port)
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
