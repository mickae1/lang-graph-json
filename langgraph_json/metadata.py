"""Python side of the tool metadata marker.

Decorating a function with :func:`AI_METADATA` does two things:

* at **build time**, ``scripts/generate_tools_json.py --lang python`` finds the
  decorator (like the C++ macro), and writes the entry to ``tools.json``
  with ``kind: "python"``. It also fills ``callable`` from the module path
  and the function name;
* at **run time**, the metadata is attached to the function
  (``fn.__ai_metadata__``), so it can be registered without ``tools.json``
  (this is how the built-in tools are declared).

Example::

    from langgraph_json.metadata import AI_METADATA, ToolContext

    @AI_METADATA({
        "name": "count_rows",
        "domain": "analysis",
        "description": "Counts the rows of a CSV file of a shared folder",
        "parameters": {"folder": {"type": "string", "required": True},
                       "file": {"type": "string", "required": True}}
    })
    async def count_rows(folder: str, file: str, ctx: ToolContext) -> dict:
        ...

A parameter named ``ctx`` receives a :class:`ToolContext` (backend client,
shared folders, backend events). It is never shown to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from .utils.backend_client import BackendClient, BackendEventListener

F = TypeVar("F", bound=Callable[..., Any])

CTX_PARAM = "ctx"


def AI_METADATA(meta: dict[str, Any]) -> Callable[[F], F]:  # noqa: N802 — mirrors the C++ macro name
    """Attach tool metadata to a function.

    Without an explicit ``kind``, metadata naming an endpoint (``method`` and
    ``path``) describes a REST tool — e.g. a Flask/FastAPI handler — otherwise a
    Python tool whose ``callable`` is this function.
    """
    if not isinstance(meta, dict) or "name" not in meta:
        raise ValueError("AI_METADATA expects a dict with at least a 'name'")

    def decorate(fn: F) -> F:
        kind = meta.get("kind") or ("rest" if "method" in meta and "path" in meta else "python")
        data = {**meta, "kind": kind}
        if kind == "python":
            data.setdefault("callable", f"{fn.__module__}:{fn.__name__}")
        fn.__ai_metadata__ = data  # type: ignore[attr-defined]
        return fn

    return decorate


@dataclass
class ToolContext:
    """Services available to Python tools through their ``ctx`` parameter."""

    client: "BackendClient"
    listener: "BackendEventListener"
    shared_paths: dict[str, str] = field(default_factory=dict)
    max_result_chars: int = 8000

    def shared_folder(self, name: str) -> Path:
        """Resolved path of a configured shared folder (raises ValueError)."""
        if name not in self.shared_paths:
            raise ValueError(f"unknown shared folder '{name}' (configured: {sorted(self.shared_paths)})")
        return Path(self.shared_paths[name]).resolve()

    def shared_file(self, folder: str, relative: str) -> Path:
        """A file inside a shared folder; refuses any path escaping it."""
        root = self.shared_folder(folder)
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise ValueError("path must stay inside the shared folder")
        return target
