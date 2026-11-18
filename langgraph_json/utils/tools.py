"""Tool registry: turns ``tools.json`` entries into LangChain tools.

Two kinds of tools, same interface for the agents (whitelist, confirmation,
limits and streaming apply identically):

* ``kind: "rest"``   — calls an endpoint of the backend REST API;
* ``kind: "python"`` — calls a Python function ``"package.module:function"``.
  Only modules listed in ``server.json`` ``python_tool_modules`` may be used
  (``agents.json`` / ``tools.json`` are editable from a UI: no arbitrary code).
  Modules are imported from the application or from ``plugins_dir``.

No Python code is written per REST endpoint. Tool results are sanitised
(control characters stripped, size capped) before they reach the LLM.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import re
import sys
import typing
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from ..builtin_tools import BUILTINS
from ..metadata import CTX_PARAM, ToolContext
from .backend_client import BackendClient, BackendError, BackendEventListener
from .config_loader import ConfigError, ParamDef, ToolDef

log = logging.getLogger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_PY_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

BUILTIN_DEFS = [ToolDef.model_validate(fn.__ai_metadata__) for fn in BUILTINS]
BUILTIN_TOOL_NAMES = {d.name for d in BUILTIN_DEFS}
BUILTIN_MODULE = BUILTINS[0].__module__


class ToolExecutionError(Exception):
    pass


def sanitize_result(result: Any, max_chars: int) -> str:
    """Serialise a tool result for the LLM: JSON, no control chars, capped size."""
    if isinstance(result, str):
        text = result
    else:
        text = json.dumps(result, ensure_ascii=False, default=str)
    text = _CONTROL_CHARS.sub("", text)
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        text = text[:max_chars] + f"... [truncated {omitted} characters — request a filtered or smaller result]"
    return text


# --------------------------------------------------------------------------- #
# Argument schemas
# --------------------------------------------------------------------------- #
def _field_type(p: ParamDef) -> Any:
    if p.enum:
        return Literal[tuple(p.enum)]  # type: ignore[valid-type]
    if p.type == "array" and p.items and p.items.get("type") in _PY_TYPES:
        return list[_PY_TYPES[p.items["type"]]]  # type: ignore[index]
    return _PY_TYPES[p.type]


def _model_name(tool_name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_-]", tool_name)) + "Args"


def build_args_schema(tool: ToolDef) -> type[BaseModel]:
    """Schema from the JSON ``parameters`` of a tool."""
    fields: dict[str, Any] = {}
    for pname, p in tool.parameters.items():
        t = _field_type(p)
        desc = p.description or None
        if p.required or pname in tool.path_params:
            fields[pname] = (t, Field(..., description=desc))
        else:
            fields[pname] = (Optional[t], Field(p.default, description=desc))
    return create_model(_model_name(tool.name), __config__=ConfigDict(extra="forbid"), **fields)


def schema_from_signature(tool_name: str, fn: Callable) -> type[BaseModel]:
    """Schema inferred from a function signature (``ctx`` excluded)."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 — unresolvable annotations
        hints = {}
    fields: dict[str, Any] = {}
    for p in inspect.signature(fn).parameters.values():
        if p.name == CTX_PARAM or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = hints.get(p.name, Any)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (ann, Field(default))
    return create_model(_model_name(tool_name), __config__=ConfigDict(extra="forbid"), **fields)


def tool_description(tool: ToolDef) -> str:
    desc = tool.description.strip()
    if tool.returns:
        desc += f"\nReturns: {json.dumps(tool.returns, ensure_ascii=False)}"
    return desc[:1024]


def _metadata(tool: ToolDef) -> dict:
    return {
        "kind": tool.kind,
        "domain": tool.domain,
        "method": tool.method,
        "path": tool.path,
        "callable": tool.callable,
        "read_only": tool.read_only,
    }


# --------------------------------------------------------------------------- #
# REST tools
# --------------------------------------------------------------------------- #
def make_rest_tool(tool: ToolDef, client: BackendClient, max_chars: int) -> StructuredTool:
    async def _call(**kwargs: Any) -> str:
        args = {k: v for k, v in kwargs.items() if v is not None}
        path = tool.path
        for pp in tool.path_params:
            if pp not in args:
                raise ToolExecutionError(f"missing path parameter '{pp}'")
            path = path.replace("{" + pp + "}", str(args.pop(pp)))
        try:
            if tool.method in ("GET", "DELETE"):
                result = await client.request(tool.method, path, params=args or None)
            else:
                result = await client.request(tool.method, path, body=args)
        except BackendError as exc:
            raise ToolExecutionError(str(exc)) from exc
        return sanitize_result(result, max_chars)

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool.name,
        description=tool_description(tool),
        args_schema=build_args_schema(tool),
        metadata=_metadata(tool),
    )


# --------------------------------------------------------------------------- #
# Python tools
# --------------------------------------------------------------------------- #
class PythonToolLoader:
    """Resolves ``"module:function"`` with a module whitelist and an optional
    plugins directory (re-imported on every hot-reload)."""

    def __init__(self, allowed_modules: list[str] | tuple = (), plugins_dir: Path | None = None):
        self.allowed = list(allowed_modules)
        self.plugins_dir = Path(plugins_dir).resolve() if plugins_dir else None
        self._reloaded: set[str] = set()
        if self.plugins_dir:
            if not self.plugins_dir.is_dir():
                raise ConfigError([f"plugins_dir does not exist: {self.plugins_dir}"])
            if str(self.plugins_dir) not in sys.path:
                sys.path.insert(0, str(self.plugins_dir))
            importlib.invalidate_caches()

    def allowed_module(self, module: str) -> bool:
        if module == BUILTIN_MODULE:
            return True
        return any(module == m or module.startswith(m + ".") for m in self.allowed)

    def _from_plugins(self, mod) -> bool:
        f = getattr(mod, "__file__", None)
        return bool(self.plugins_dir and f and self.plugins_dir in Path(f).resolve().parents)

    def _import(self, module: str):
        mod = sys.modules.get(module)
        if mod is not None and self._from_plugins(mod) and module not in self._reloaded:
            # Hot-reload: pick up the new code of plugin modules (and parents).
            parts = module.split(".")
            for i in range(1, len(parts) + 1):
                name = ".".join(parts[:i])
                m = sys.modules.get(name)
                if m is not None and self._from_plugins(m) and name not in self._reloaded:
                    importlib.reload(m)
                    self._reloaded.add(name)
            return sys.modules[module]
        mod = importlib.import_module(module)
        self._reloaded.add(module)
        return mod

    def load(self, spec: str) -> Callable:
        module, _, attr = spec.partition(":")
        if not self.allowed_module(module):
            raise ConfigError([f"python tool module '{module}' is not allowed (server.json 'python_tool_modules')"])
        try:
            mod = self._import(module)
        except Exception as exc:  # noqa: BLE001 — report any import failure
            raise ConfigError([f"cannot import python tool module '{module}': {type(exc).__name__}: {exc}"]) from exc
        fn = getattr(mod, attr, None)
        if fn is None or not callable(fn):
            raise ConfigError([f"python tool '{spec}': '{attr}' is not a function of module '{module}'"])
        return fn


def make_python_tool(tool: ToolDef, fn: Callable, ctx: ToolContext) -> StructuredTool:
    sig = inspect.signature(fn)
    accepts_ctx = CTX_PARAM in sig.parameters
    is_async = inspect.iscoroutinefunction(fn)
    schema = build_args_schema(tool) if tool.parameters else schema_from_signature(tool.name, fn)
    description = tool.description.strip() or (inspect.getdoc(fn) or tool.name).split("\n\n")[0]

    async def _call(**kwargs: Any) -> str:
        # None = "not provided": let the function's own defaults apply.
        args = {k: v for k, v in kwargs.items() if v is not None}
        if accepts_ctx:
            args[CTX_PARAM] = ctx
        try:
            if is_async:
                result = await fn(**args)
            else:
                result = await asyncio.to_thread(fn, **args)
            if inspect.isawaitable(result):
                result = await result
        except (ToolExecutionError, asyncio.CancelledError):  # pragma: no cover
            raise
        except Exception as exc:  # noqa: BLE001 — returned to the LLM as a tool error
            raise ToolExecutionError(f"{type(exc).__name__}: {exc}") from exc
        return sanitize_result(result, ctx.max_result_chars)

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool.name,
        description=(description + (f"\nReturns: {json.dumps(tool.returns, ensure_ascii=False)}" if tool.returns else ""))[:1024],
        args_schema=schema,
        metadata=_metadata(tool),
    )


def tooldef_from_function(fn: Callable) -> ToolDef:
    """ToolDef of a function: its ``@AI_METADATA`` if any, else name + docstring."""
    meta = getattr(fn, "__ai_metadata__", None)
    if meta is None:
        doc = (inspect.getdoc(fn) or "").split("\n\n")[0].strip()
        meta = {"name": fn.__name__, "description": doc or fn.__name__, "kind": "python"}
    meta = {**meta, "kind": "python", "callable": meta.get("callable") or f"{fn.__module__}:{fn.__name__}"}
    try:
        return ToolDef.model_validate(meta)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError([f"invalid tool function {fn!r}: {exc}"]) from exc


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """All tools available to agents. Agents only get their whitelist."""

    def __init__(
        self,
        tools: list[ToolDef],
        client: BackendClient,
        listener: BackendEventListener | None = None,
        shared_paths: dict[str, str] | None = None,
        max_result_chars: int = 8000,
        allowed_modules: list[str] | tuple = (),
        plugins_dir: Path | None = None,
        functions: list[Callable] | tuple = (),
    ):
        """``functions``: Python callables registered directly (library use).
        They are trusted code, so the module whitelist does not apply."""
        self.ctx = ToolContext(
            client=client,
            listener=listener or BackendEventListener(None),
            shared_paths=shared_paths if shared_paths is not None else {},
            max_result_chars=max_result_chars,
        )
        loader = PythonToolLoader(allowed_modules, plugins_dir)
        self.defs: dict[str, ToolDef] = {}
        self.tools: dict[str, BaseTool] = {}
        for t in [*tools, *BUILTIN_DEFS]:
            if t.name in self.tools:
                raise ConfigError([f"duplicate tool name '{t.name}'"])
            if t.kind == "rest":
                self.tools[t.name] = make_rest_tool(t, client, max_result_chars)
            else:
                self.tools[t.name] = make_python_tool(t, loader.load(t.callable), self.ctx)
            self.defs[t.name] = t
        for fn in functions:
            t = tooldef_from_function(fn)
            if t.name in self.tools:
                raise ConfigError([f"duplicate tool name '{t.name}'"])
            self.tools[t.name] = make_python_tool(t, fn, self.ctx)
            self.defs[t.name] = t

    def for_agent(self, allowed: list[str]) -> list[BaseTool]:
        """The whitelist IS the permission system: unknown names are skipped."""
        return [self.tools[n] for n in dict.fromkeys(allowed) if n in self.tools]

    def describe(self) -> list[dict]:
        out = []
        for name, tool in self.tools.items():
            d = self.defs[name]
            params = (
                {k: v.model_dump(exclude_none=True) for k, v in d.parameters.items()}
                if d.parameters
                else tool.args_schema.model_json_schema().get("properties", {})
            )
            out.append({"name": name, **tool.metadata, "description": d.description, "parameters": params})
        return out
