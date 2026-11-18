"""Client of the host application (the "backend").

* :class:`BackendClient` — async HTTP calls to the backend REST API (the tools).
* :class:`BackendEventListener` — optional WebSocket subscription to backend
  events (job progress, completion…), so agents can *wait* for an
  event instead of polling.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from typing import Any

import httpx

log = logging.getLogger(__name__)


class BackendError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class BackendClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # The backend is on localhost: never go through a proxy.
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_s,
                transport=self._transport,
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def request(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> Any:
        client = self._get_client()
        method = method.upper()
        kwargs: dict[str, Any] = {}
        if params:
            kwargs["params"] = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()}
        if body is not None and method not in ("GET", "DELETE"):
            kwargs["json"] = body
        try:
            resp = await client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise BackendError(f"backend timeout on {method} {path}") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"backend unreachable on {method} {path}: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"backend returned HTTP {resp.status_code} on {method} {path}: {resp.text[:500]}", resp.status_code)
        if not resp.content:
            return {"status": "ok"}
        try:
            return resp.json()
        except ValueError:
            return resp.text

    async def ping(self, path: str = "/") -> bool:
        try:
            await self._get_client().get(path, timeout=2)
            return True
        except httpx.HTTPError:
            return False


class BackendEventListener:
    """Keeps a WebSocket connection to the backend and buffers its events.

    Events are JSON objects; their type is read from ``type`` or ``event_type``.
    """

    def __init__(self, ws_url: str | None, buffer_size: int = 500):
        self.ws_url = ws_url
        self.events: deque[tuple[int, float, dict]] = deque(maxlen=buffer_size)
        self._seq = 0
        self.connected = False
        self._cond = asyncio.Condition()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self.ws_url and self._task is None:
            self._task = asyncio.create_task(self._run(), name="backend-ws-listener")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    async def _run(self) -> None:
        import websockets

        delay = 1.0
        while True:
            try:
                async with websockets.connect(self.ws_url, proxy=None) as ws:
                    self.connected = True
                    delay = 1.0
                    log.info("connected to backend events %s", self.ws_url)
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(event, dict):
                            await self.push(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — reconnect on any failure
                log.debug("backend events connection lost: %s", exc)
            self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    async def push(self, event: dict) -> None:
        async with self._cond:
            self._seq += 1
            self.events.append((self._seq, time.monotonic(), event))
            self._cond.notify_all()

    @staticmethod
    def event_type(event: dict) -> str | None:
        return event.get("type") or event.get("event_type") or event.get("event")

    @classmethod
    def matches(cls, event: dict, event_type: str, match: dict | None) -> bool:
        if cls.event_type(event) != event_type:
            return False
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        for k, v in (match or {}).items():
            if event.get(k, data.get(k)) != v:
                return False
        return True

    async def wait_for(self, event_type: str, match: dict | None = None, timeout_s: float = 300.0, lookback_s: float = 30.0) -> dict:
        """Return the first matching event (recent buffered events included)."""
        if not self.ws_url:
            raise BackendError("backend events are not configured (server.json: backend_ws_url)")
        since = time.monotonic() - lookback_s
        deadline = time.monotonic() + timeout_s
        last_seq = 0
        async with self._cond:
            while True:
                for seq, ts, ev in list(self.events):
                    if seq > last_seq and ts >= since and self.matches(ev, event_type, match):
                        return ev
                last_seq = self._seq
                remaining = deadline - time.monotonic()
                if remaining <= 0:  # pragma: no cover — timing dependent
                    raise TimeoutError(f"no '{event_type}' event received within {timeout_s:.0f}s")
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except TimeoutError:
                    raise TimeoutError(f"no '{event_type}' event received within {timeout_s:.0f}s") from None
