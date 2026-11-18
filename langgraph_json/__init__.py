"""lang-graph-json — a generic LangGraph multi-agent engine driven by JSON.

Agents are entries in ``agents.json``; tools come from ``tools.json`` (REST
endpoints or Python functions). No Python code is written per agent.

Two ways to use it:

* **standalone server** (PyInstaller bundle or ``lang-graph-json`` CLI),
  driven over REST / WebSocket by an application written in any language;
* **Python library**, embedded in your own project::

      from langgraph_json import Runtime

      async with Runtime(agents={...}, functions=[my_tool], llm=my_chat_model) as rt:
          result = await rt.chat("Open document 42", session_id="s1")

  or mount the HTTP API in an existing FastAPI app with :func:`create_app`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_EXPORTS = {
    "Runtime": "langgraph_json.gateway.runtime",
    "create_app": "langgraph_json.gateway.app",
    "AI_METADATA": "langgraph_json.metadata",
    "ToolContext": "langgraph_json.metadata",
    "ProviderConfig": "langgraph_json.utils.llm_factory",
    "ProxyConfig": "langgraph_json.utils.llm_factory",
    "LLMFactory": "langgraph_json.utils.llm_factory",
    "ConfigError": "langgraph_json.utils.config_loader",
    "load_config": "langgraph_json.utils.config_loader",
    "config_from_data": "langgraph_json.utils.config_loader",
    "build_graph": "langgraph_json.agent",
    "FakeChatModel": "langgraph_json.utils.fake_llm",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:  # PEP 562: keep "import langgraph_json" light
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'langgraph_json' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


if TYPE_CHECKING:  # pragma: no cover
    from .agent import build_graph
    from .gateway.app import create_app
    from .gateway.runtime import Runtime
    from .metadata import AI_METADATA, ToolContext
    from .utils.config_loader import ConfigError, config_from_data, load_config
    from .utils.fake_llm import FakeChatModel
    from .utils.llm_factory import LLMFactory, ProviderConfig, ProxyConfig
