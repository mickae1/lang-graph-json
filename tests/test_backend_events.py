import asyncio
import json

import pytest
import websockets

from langgraph_json.utils.backend_client import BackendEventListener


async def test_wait_for_receives_future_event():
    listener = BackendEventListener("ws://unused")

    async def later():
        await asyncio.sleep(0.05)
        await listener.push({"type": "other"})
        await listener.push({"event_type": "job.completed", "jobId": "B"})
        await listener.push({"event_type": "job.completed", "jobId": "A"})

    asyncio.create_task(later())
    ev = await listener.wait_for("job.completed", {"jobId": "A"}, timeout_s=2)
    assert ev["jobId"] == "A"


async def test_recent_events_are_found_and_timeout():
    listener = BackendEventListener("ws://unused")
    await listener.push({"type": "done", "data": {"id": 1}})
    assert (await listener.wait_for("done", {"id": 1}, timeout_s=1))["data"]["id"] == 1
    with pytest.raises(TimeoutError):
        await listener.wait_for("done", {"id": 2}, timeout_s=0.1)


async def test_buffer_rotation_does_not_lose_new_events():
    listener = BackendEventListener("ws://unused", buffer_size=3)
    for i in range(10):
        await listener.push({"type": "noise", "i": i})

    async def later():
        await asyncio.sleep(0.05)
        for i in range(5):
            await listener.push({"type": "noise", "i": i})
        await listener.push({"type": "target"})

    asyncio.create_task(later())
    assert (await listener.wait_for("target", timeout_s=2))["type"] == "target"


async def test_listener_connects_to_real_websocket():
    async def handler(ws):
        await ws.send(json.dumps({"type": "job.progress", "data": {"progress": 50}}))
        await ws.send("not json")
        await ws.send(json.dumps({"type": "job.completed", "data": {"jobId": "J9"}}))
        await asyncio.sleep(1)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        listener = BackendEventListener(f"ws://127.0.0.1:{port}")
        listener.start()
        try:
            ev = await listener.wait_for("job.completed", {"jobId": "J9"}, timeout_s=5)
            assert ev["data"]["jobId"] == "J9"
            assert listener.connected
        finally:
            await listener.stop()
