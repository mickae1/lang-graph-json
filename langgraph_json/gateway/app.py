"""FastAPI gateway.

Endpoints (localhost only):

* ``POST /chat``                 — run a user turn
* ``POST /confirm``              — approve / deny a pending tool call
* ``POST /cancel``               — cancel the running turn / pending confirmation
* ``WS   /stream``               — real-time events (and confirmations)
* ``POST /reload`` (alias ``POST /agent/reload``) — hot-reload JSON config
* ``POST /config/validate``, ``POST /agents/validate`` — dry-run validation
* ``GET  /status``               — health, agents, sessions
* ``GET|POST /config/provider``  — LLM providers (in memory only)
* ``POST /config/provider/token`` — renew a host-managed token (user sign-in)
* ``GET  /config/provider/test`` — test LLM connectivity
* ``POST /config/proxy``         — proxy settings (in memory only)
* ``GET|POST /config/paths``     — shared exchange folders
* ``GET  /tools``, ``GET /agents`` — what the server loaded
* ``GET|DELETE /sessions/{id}``  — history / reset of a conversation
* ``POST /shutdown``             — graceful stop (used by the host application)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from .. import __version__
from ..utils.config_loader import ConfigError
from ..utils.events import envelope
from ..utils.llm_factory import ProviderConfig, ProviderNotConfigured, ProxyConfig
from .runtime import Conflict, NotReady, Runtime

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(None, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    # Address one agent directly (bypass the supervisor), e.g. from the editor test chat.
    agent: str | None = None


class ConfirmRequest(BaseModel):
    session_id: str
    approved: bool
    reason: str | None = None


class PathsRequest(BaseModel):
    paths: dict[str, str]


class TokenRequest(BaseModel):
    token: str = Field(..., min_length=1)
    provider: str | None = None
    expires_in: float | None = Field(None, gt=0)


class CancelRequest(BaseModel):
    session_id: str
    reason: str | None = None


class ValidateRequest(BaseModel):
    # Content of agents.json / tools.json / server.json; omitted parts are
    # taken from the current configuration.
    agents: dict[str, Any] | list | None = None
    tools: dict[str, Any] | list | None = None
    settings: dict[str, Any] | None = None


def create_app(runtime: Runtime, manage_lifespan: bool = True) -> FastAPI:
    """HTTP/WebSocket API around a :class:`Runtime`.

    To mount it inside an existing FastAPI application
    (``host.mount("/agents", create_app(rt, manage_lifespan=False))``), note that
    Starlette does not run the lifespan of mounted apps: start the runtime from
    the host lifespan (``async with rt: yield``).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if manage_lifespan:
            await runtime.startup()
        yield
        if manage_lifespan:
            await runtime.shutdown()

    app = FastAPI(title="lang-graph-json", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime
    app.state.server = None  # set by __main__ (uvicorn.Server) for /shutdown

    @app.exception_handler(NotReady)
    async def _not_ready(_: Request, exc: NotReady):
        return JSONResponse(status_code=503, content={"status": "error", "error": str(exc)})

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, exc: Conflict):
        return JSONResponse(status_code=409, content={"status": "error", "error": str(exc)})

    @app.exception_handler(ConfigError)
    async def _config_error(_: Request, exc: ConfigError):
        return JSONResponse(status_code=422, content={"status": "error", "errors": exc.errors})

    # ------------------------------------------------------------------ #
    @app.get("/status")
    async def status():
        return runtime.status()

    @app.post("/chat")
    async def chat(req: ChatRequest):
        try:
            return await runtime.chat(req.message, req.session_id, req.agent)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (NotReady, Conflict):
            raise
        except ProviderNotConfigured as exc:
            raise NotReady(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"status": "error", "session_id": req.session_id, "error": f"{type(exc).__name__}: {exc}"})

    @app.post("/confirm")
    async def confirm(req: ConfirmRequest):
        try:
            return await runtime.confirm(req.session_id, req.approved, req.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown session {req.session_id}") from exc
        except (NotReady, Conflict):
            raise
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"status": "error", "session_id": req.session_id, "error": f"{type(exc).__name__}: {exc}"})

    @app.post("/cancel")
    async def cancel(req: CancelRequest):
        try:
            return await runtime.cancel(req.session_id, req.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown session {req.session_id}") from exc

    @app.post("/reload")
    @app.post("/agent/reload")
    async def reload():
        return runtime.reload()

    @app.post("/config/validate")
    async def validate_config(req: ValidateRequest):
        """Dry run: would a reload with this configuration succeed? Nothing is applied."""
        return runtime.validate(req.agents, req.tools, req.settings)

    @app.post("/agents/validate")
    async def validate_agents(payload: dict[str, Any]):
        """Validate the content of agents.json before writing it (editor "Save")."""
        return runtime.validate(agents=payload)

    # ------------------------------------------------------------------ #
    @app.get("/config/provider")
    async def get_provider():
        if not runtime.llms.configured:
            return {"configured": False, "models": {}}
        return {
            "configured": True,
            **runtime.llms.providers.redacted(),
            "models": runtime.llms.available_models(),
            "token_expires_in_s": runtime.token_status(),
        }

    @app.post("/config/provider")
    async def set_provider(payload: dict[str, Any]):
        try:
            cfg = ProviderConfig.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_input=False, include_context=False)) from exc
        runtime.set_provider(cfg)
        return {"status": "configured", "default": cfg.default, "providers": sorted(cfg.providers), "graph_ready": runtime.graph is not None}

    @app.post("/config/provider/token")
    async def set_token(req: TokenRequest):
        """Renew a host-managed token (e.g. the user's Azure AD token) without rebuilding anything."""
        try:
            return runtime.set_token(req.token, req.provider, req.expires_in)
        except ProviderNotConfigured as exc:
            raise NotReady(str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/config/provider/test")
    async def test_provider(provider: str | None = None, model: str | None = None):
        return await runtime.test_provider(provider, model)

    @app.post("/config/proxy")
    async def set_proxy(payload: dict[str, Any]):
        try:
            cfg = ProxyConfig.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_input=False, include_context=False)) from exc
        runtime.set_proxy(cfg)
        return {"status": "configured", "http_proxy": bool(cfg.http_proxy), "https_proxy": bool(cfg.https_proxy), "no_proxy": cfg.no_proxy}

    @app.get("/config/paths")
    async def get_paths():
        return {"paths": dict(runtime.shared_paths)}

    @app.post("/config/paths")
    async def set_paths(req: PathsRequest):
        return runtime.set_paths(req.paths)

    # ------------------------------------------------------------------ #
    @app.get("/tools")
    async def tools():
        return {"tools": runtime.registry.describe() if runtime.registry else []}

    @app.get("/agents")
    async def agents():
        if runtime.bundle is None:
            raise NotReady("configuration is invalid")
        return runtime.bundle.agents.model_dump()

    @app.get("/sessions")
    async def sessions():
        return {
            "sessions": [
                {"session_id": s.id, "created": s.created, "last_active": s.last_active, "turns": s.turns}
                for s in runtime.sessions.values()
            ]
        }

    @app.get("/sessions/{session_id}")
    async def session_history(session_id: str):
        try:
            return {"session_id": session_id, "messages": await runtime.history(session_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown session") from exc

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            await runtime.delete_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown session") from exc
        return {"status": "deleted", "session_id": session_id}

    @app.post("/shutdown")
    async def shutdown():
        server = app.state.server
        if server is None:
            raise HTTPException(status_code=501, detail="shutdown is only available when started from the CLI")
        server.should_exit = True
        return {"status": "shutting_down"}

    # ------------------------------------------------------------------ #
    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        """Events of one session (``?session_id=``) or of all sessions.

        Client messages: ``{"type": "confirm", "session_id": ..., "approved": bool, "reason": ...}``
        and ``{"type": "ping"}``.
        """
        await ws.accept()
        session_id = ws.query_params.get("session_id")
        queue = runtime.events.subscribe(session_id)
        await ws.send_json(envelope(session_id or "*", "stream.connected", {"session_id": session_id}))

        async def sender():
            while True:
                await ws.send_json(await queue.get())

        send_task = asyncio.create_task(sender())
        pending: set[asyncio.Task] = set()
        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "ping":
                    await ws.send_json(envelope(session_id or "*", "pong", {}))
                elif msg.get("type") in ("confirm", "confirmation"):
                    sid = msg.get("session_id") or session_id
                    task = asyncio.create_task(_ws_confirm(runtime, sid, bool(msg.get("approved")), msg.get("reason")))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                elif msg.get("type") == "cancel":
                    sid = msg.get("session_id") or session_id
                    task = asyncio.create_task(_ws_cancel(runtime, sid, msg.get("reason")))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except ValueError:  # invalid JSON
            with contextlib.suppress(Exception):
                await ws.close(code=1003)
        finally:
            send_task.cancel()
            runtime.events.unsubscribe(session_id, queue)

    return app


async def _ws_cancel(runtime: Runtime, session_id: str | None, reason: str | None) -> None:
    try:
        await runtime.cancel(session_id, reason)
    except Exception as exc:  # noqa: BLE001
        runtime.events.publish(envelope(session_id or "*", "error", {"message": f"cancel failed: {exc}", "recoverable": True}))


async def _ws_confirm(runtime: Runtime, session_id: str | None, approved: bool, reason: str | None) -> None:
    try:
        result = await runtime.confirm(session_id, approved, reason)
        runtime.events.publish(envelope(session_id, "chat.result", result))
    except Exception as exc:  # noqa: BLE001
        runtime.events.publish(envelope(session_id or "*", "error", {"message": f"confirmation failed: {exc}", "recoverable": True}))
