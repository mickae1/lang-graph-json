"""Mount the lang-graph-json HTTP API inside an existing FastAPI application.

    pip install lang-graph-json
    uvicorn examples.embedded.mounted_fastapi:app --port 9000
    # -> POST http://127.0.0.1:9000/agents/chat, WS /agents/stream, GET /agents/status ...

The configuration is read from a directory (hot-reloaded); the provider can
be pushed with POST /agents/config/provider, or given in code (llm=...).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from langgraph_json import Runtime, create_app

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

runtime = Runtime(CONFIG_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Mounted apps have no lifespan of their own: start the runtime here.
    async with runtime:
        yield


app = FastAPI(title="My application", lifespan=lifespan)
app.mount("/agents", create_app(runtime, manage_lifespan=False))


@app.get("/")
async def home():
    return {"app": "my application", "agents": runtime.status()["status"]}
