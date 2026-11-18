"""Loading and validation of the JSON configuration files.

Three files live in the configuration directory (a volume shared with the
host application):

* ``agents.json`` — agent definitions (written by the host application UI)
* ``tools.json``  — REST endpoint metadata (generated at build time)
* ``server.json`` — operational settings, never secrets (optional)

Secrets (LLM credentials, proxy) are never read from disk: they are pushed
through the REST API (``POST /config/provider``, ``POST /config/proxy``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .state import FINISH, SUPERVISOR, TOOLS_SUFFIX

log = logging.getLogger(__name__)

AGENTS_FILE = "agents.json"
TOOLS_FILE = "tools.json"
SERVER_FILE = "server.json"

AGENT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
PARAM_TYPES = {"string", "integer", "number", "boolean", "array", "object"}
RESERVED_AGENT_NAMES = {SUPERVISOR, FINISH, "__start__", "__end__"}

DEFAULT_SUPERVISOR_PROMPT = (
    "You are the supervisor of a team of specialised agents. "
    "Read the conversation and decide which agent must act next, or FINISH when "
    "the user's request has been fully answered. Route one step at a time."
)


class ConfigError(Exception):
    """Raised when the configuration is invalid. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


# --------------------------------------------------------------------------- #
# server.json
# --------------------------------------------------------------------------- #
class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    # Base URL of the host application's REST API (the tools target it).
    backend_url: str = "http://127.0.0.1:8080"
    # Optional WebSocket of the host application (progress / completion events).
    backend_ws_url: str | None = None
    backend_timeout_s: float = Field(30.0, gt=0)
    # NFR-07 / NFR-08
    max_tool_calls_per_turn: int = Field(20, ge=0)
    token_budget_per_session: int = Field(100_000, ge=0)
    token_alert_ratio: float = Field(0.8, gt=0, le=1)
    default_max_history_messages: int = Field(40, ge=0)
    # Tool results larger than this are truncated before reaching the LLM.
    max_tool_result_chars: int = Field(8000, ge=200)
    # NFR-09: poll config files for changes (seconds, 0 = disabled).
    watch_interval_s: float = Field(2.0, ge=0)
    recursion_limit: int = Field(200, ge=10)
    log_level: str = "info"
    # Conversation history: "memory" (default, nothing written to disk, lost
    # on restart) or "sqlite" (local file, survives restarts; needs the
    # optional dependency: pip install "lang-graph-json[sqlite]").
    # Changing it requires a restart.
    persistence: Literal["memory", "sqlite"] = "memory"
    # SQLite file; relative paths are resolved against the application root
    # (parent of the configuration directory).
    sqlite_path: str = "data/sessions.db"
    # Sessions inactive for longer are deleted (memory and disk); 0 = never.
    session_ttl_hours: float = Field(24.0, ge=0)
    # A turn running longer is cancelled; 0 = no limit.
    turn_timeout_s: float = Field(600.0, ge=0)
    # A confirmation left unanswered longer is treated as denied; 0 = wait forever.
    confirmation_timeout_s: float = Field(900.0, ge=0)
    # Period of the maintenance task (session expiry, confirmation timeouts).
    maintenance_interval_s: float = Field(10.0, gt=0)
    # Python tools (kind = "python"): only these modules (and their
    # sub-modules) may be referenced by a tool's "callable".
    python_tool_modules: list[str] = Field(default_factory=list)
    # Optional directory of Python plugins, added to the import path and
    # reloaded on hot-reload. Relative paths are resolved against the parent
    # of the configuration directory (the application root).
    plugins_dir: str | None = None

    def module_allowed(self, module: str) -> bool:
        return any(module == m or module.startswith(m + ".") for m in self.python_tool_modules)

    def resolve_sqlite_path(self, config_dir: Path) -> Path:
        p = Path(self.sqlite_path)
        return p if p.is_absolute() else (Path(config_dir).resolve().parent / p).resolve()

    def resolve_plugins_dir(self, config_dir: Path) -> Path | None:
        if not self.plugins_dir:
            return None
        p = Path(self.plugins_dir)
        return p if p.is_absolute() else (Path(config_dir).resolve().parent / p).resolve()


# --------------------------------------------------------------------------- #
# tools.json
# --------------------------------------------------------------------------- #
class ParamDef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "string"
    required: bool = False
    description: str = ""
    enum: list[Any] | None = None
    default: Any = None
    items: dict[str, Any] | None = None

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in PARAM_TYPES:
            raise ValueError(f"unsupported parameter type '{v}' (expected one of {sorted(PARAM_TYPES)})")
        return v


CALLABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")


class ToolDef(BaseModel):
    """One tool. ``kind="rest"`` calls a backend endpoint (``method`` + ``path``);
    ``kind="python"`` calls a Python function (``callable = "module:function"``)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str
    kind: Literal["rest", "python"] = "rest"
    method: str | None = None
    path: str | None = None
    callable: str | None = None
    description: str
    domain: str = "general"
    parameters: dict[str, ParamDef] = Field(default_factory=dict)
    returns: Any = None
    # Python tools: explicit read-only flag (REST tools: GET = read-only).
    read_only_hint: bool | None = Field(None, alias="read_only")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not TOOL_NAME_RE.match(v):
            raise ValueError(f"invalid tool name '{v}' (allowed: letters, digits, _ and -, max 64)")
        return v

    @field_validator("method")
    @classmethod
    def _check_method(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.upper()
        if v not in HTTP_METHODS:
            raise ValueError(f"invalid HTTP method '{v}'")
        return v

    @field_validator("path")
    @classmethod
    def _check_path(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("/"):
            raise ValueError(f"path must start with '/': '{v}'")
        return v

    @model_validator(mode="after")
    def _check_kind(self) -> "ToolDef":
        if self.kind == "rest":
            if not self.method or not self.path:
                raise ValueError(f"REST tool '{self.name}' requires 'method' and 'path'")
        else:
            if not self.callable or not CALLABLE_RE.match(self.callable):
                raise ValueError(f"python tool '{self.name}' requires 'callable' as 'package.module:function'")
        return self

    @property
    def module(self) -> str | None:
        return self.callable.split(":", 1)[0] if self.callable else None

    @property
    def read_only(self) -> bool:
        if self.kind == "python":
            return bool(self.read_only_hint)
        return self.method == "GET"

    @property
    def path_params(self) -> list[str]:
        return re.findall(r"\{([A-Za-z0-9_]+)\}", self.path) if self.path else []


class ToolsFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    tools: list[ToolDef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"tools": data}
        return data


# --------------------------------------------------------------------------- #
# agents.json
# --------------------------------------------------------------------------- #
class AgentDef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    display_name: str | None = None
    enabled: bool = True
    type: Literal["react", "sub_graph"] = "react"
    description: str
    system_prompt: str
    allowed_tools: list[str] = Field(default_factory=list)
    requires_confirmation: list[str] = Field(default_factory=list)
    max_history_messages: int | None = Field(None, ge=0)
    model: str | None = None
    provider: str | None = None
    can_handoff_to: list[str] = Field(default_factory=list)
    sub_agents: list[str] = Field(default_factory=list)
    # When false the agent is not offered to the top-level supervisor
    # (useful for agents only used as sub-agents of a sub_graph).
    routable: bool = True


class SupervisorDef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    system_prompt: str = DEFAULT_SUPERVISOR_PROMPT
    model: str | None = None
    provider: str | None = None
    max_steps: int = Field(10, ge=1)
    max_history_messages: int = Field(40, ge=0)


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    default_model: str | None = None
    default_provider: str | None = None
    supervisor: SupervisorDef = Field(default_factory=SupervisorDef)
    agents: dict[str, AgentDef] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalise_agents(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("agents"), list):
            agents = {}
            for entry in data["agents"]:
                if not isinstance(entry, dict) or not entry.get("name"):
                    raise ValueError("each agent of the 'agents' list needs a 'name'")
                agents[entry["name"]] = entry
            data = {**data, "agents": agents}
        return data

    @model_validator(mode="after")
    def _fill_names(self) -> "AgentsConfig":
        for key, agent in self.agents.items():
            agent.name = key
        return self

    def enabled_agents(self) -> dict[str, AgentDef]:
        return {k: a for k, a in self.agents.items() if a.enabled}


# --------------------------------------------------------------------------- #
# Bundle + validation
# --------------------------------------------------------------------------- #
@dataclass
class ConfigBundle:
    agents: AgentsConfig
    tools: list[ToolDef]
    settings: ServerSettings
    warnings: list[str] = field(default_factory=list)

    @property
    def tools_by_name(self) -> dict[str, ToolDef]:
        return {t.name: t for t in self.tools}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ConfigError([f"{path.name}: invalid JSON at line {exc.lineno} col {exc.colno}: {exc.msg}"]) from exc


def _pydantic_errors(filename: str, exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        out.append(f"{filename}: {loc}: {err['msg']}")
    return out


def validate_semantics(
    agents: AgentsConfig, tool_names: set[str], builtin_tool_names: set[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """Cross-file checks. Returns (errors, warnings).

    Unknown tool names / handoff targets are *warnings* (they are ignored when
    the graph is built) so a stale ``tools.json`` never blocks the server.
    """
    errors: list[str] = []
    warnings: list[str] = []
    known_tools = set(tool_names) | set(builtin_tool_names)
    all_agents = agents.agents

    for name, agent in all_agents.items():
        where = f"agents.json: agent '{name}'"
        if not AGENT_NAME_RE.match(name):
            errors.append(f"{where}: invalid name (letters, digits, _ and -, must start with a letter)")
        if name in RESERVED_AGENT_NAMES or name.endswith(TOOLS_SUFFIX):
            errors.append(f"{where}: name is reserved")
        if not agent.enabled:
            continue
        if not agent.description.strip():
            errors.append(f"{where}: 'description' must not be empty")
        if not agent.system_prompt.strip():
            errors.append(f"{where}: 'system_prompt' must not be empty")

        if agent.type == "react":
            if not agent.allowed_tools:
                errors.append(f"{where}: at least one tool must be assigned in 'allowed_tools'")
            for t in agent.allowed_tools:
                if t not in known_tools:
                    warnings.append(f"{where}: unknown tool '{t}' ignored")
            for t in agent.requires_confirmation:
                if t not in agent.allowed_tools:
                    warnings.append(f"{where}: '{t}' in requires_confirmation but not in allowed_tools")
        else:  # sub_graph
            if not agent.sub_agents:
                errors.append(f"{where}: a sub_graph agent needs 'sub_agents'")
            for sub in agent.sub_agents:
                if sub not in all_agents:
                    errors.append(f"{where}: unknown sub-agent '{sub}'")
                elif sub == name:
                    errors.append(f"{where}: an agent cannot be its own sub-agent")

        for target in agent.can_handoff_to:
            if target not in all_agents:
                warnings.append(f"{where}: unknown handoff target '{target}' ignored")
            elif not all_agents[target].enabled:
                warnings.append(f"{where}: handoff target '{target}' is disabled")

    # Cycles between sub_graph agents.
    def visit(n: str, stack: tuple[str, ...]) -> None:
        a = all_agents.get(n)
        if a is None or a.type != "sub_graph":
            return
        for sub in a.sub_agents:
            if sub in stack:
                errors.append(f"agents.json: sub_graph cycle {' -> '.join(stack + (sub,))}")
                return
            visit(sub, stack + (sub,))

    for n in all_agents:
        visit(n, (n,))

    if not any(a.enabled and a.routable for a in all_agents.values()):
        warnings.append("agents.json: no enabled routable agent — the supervisor can only answer directly")
    return sorted(set(errors), key=errors.index), warnings


def load_tools(path: Path) -> list[ToolDef]:
    if not path.exists():
        return []
    try:
        tf = ToolsFile.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ConfigError(_pydantic_errors(path.name, exc)) from exc
    seen: set[str] = set()
    dups = [t.name for t in tf.tools if t.name in seen or seen.add(t.name)]
    if dups:
        raise ConfigError([f"{path.name}: duplicate tool names: {sorted(set(dups))}"])
    return tf.tools


def load_settings(path: Path) -> ServerSettings:
    if not path.exists():
        return ServerSettings()
    try:
        return ServerSettings.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ConfigError(_pydantic_errors(path.name, exc)) from exc


def load_agents(path: Path) -> AgentsConfig:
    if not path.exists():
        raise ConfigError([f"{path.name}: file not found in {path.parent}"])
    try:
        return AgentsConfig.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ConfigError(_pydantic_errors(path.name, exc)) from exc


def load_config(config_dir: str | Path, builtin_tool_names: set[str] = frozenset()) -> ConfigBundle:
    """Load and validate the whole configuration directory.

    Raises :class:`ConfigError` listing every problem found.
    """
    config_dir = Path(config_dir)
    errors: list[str] = []
    warnings: list[str] = []

    settings = tools = agents = None
    for loader, fname in ((load_settings, SERVER_FILE), (load_tools, TOOLS_FILE), (load_agents, AGENTS_FILE)):
        try:
            value = loader(config_dir / fname)
        except ConfigError as exc:
            errors.extend(exc.errors)
            continue
        if fname == SERVER_FILE:
            settings = value
        elif fname == TOOLS_FILE:
            tools = value
            if not (config_dir / fname).exists():
                warnings.append(f"{TOOLS_FILE} not found — no backend tools available")
        else:
            agents = value

    return _finalize(agents, tools, settings, errors, warnings, builtin_tool_names)


def _finalize(agents, tools, settings, errors: list[str], warnings: list[str], builtin_tool_names) -> ConfigBundle:
    if settings is not None and tools is not None:
        for t in tools:
            if t.kind == "python" and not settings.module_allowed(t.module):
                errors.append(
                    f"{TOOLS_FILE}: tool '{t.name}': module '{t.module}' is not allowed "
                    f"(add it to server.json 'python_tool_modules')"
                )
            if t.name in builtin_tool_names:
                errors.append(f"{TOOLS_FILE}: tool '{t.name}' conflicts with a built-in tool")

    if agents is not None and tools is not None:
        errs, warns = validate_semantics(agents, {t.name for t in tools}, builtin_tool_names)
        errors.extend(errs)
        warnings.extend(warns)

    if errors:
        raise ConfigError(errors)
    for w in warnings:
        log.warning(w)
    return ConfigBundle(agents=agents, tools=tools, settings=settings, warnings=warnings)


def config_from_data(
    agents: dict | AgentsConfig,
    tools: list | ToolsFile | None = None,
    settings: dict | ServerSettings | None = None,
    builtin_tool_names: set[str] = frozenset(),
) -> ConfigBundle:
    """Same validation as :func:`load_config`, from Python objects (library use).

    ``agents`` is the content of ``agents.json`` (or an :class:`AgentsConfig`),
    ``tools`` the content of ``tools.json`` (list or ``{"tools": [...]}``),
    ``settings`` the content of ``server.json``.
    """
    errors: list[str] = []
    out: dict[str, Any] = {}
    specs = (
        ("settings", SERVER_FILE, ServerSettings, settings if settings is not None else {}),
        ("tools", TOOLS_FILE, ToolsFile, tools if tools is not None else []),
        ("agents", AGENTS_FILE, AgentsConfig, agents),
    )
    for key, label, model, data in specs:
        try:
            out[key] = data if isinstance(data, model) else model.model_validate(data)
        except ValidationError as exc:
            errors.extend(_pydantic_errors(label, exc))
    tools_list = out["tools"].tools if "tools" in out else None
    if tools_list is not None:
        seen: set[str] = set()
        dups = sorted({t.name for t in tools_list if t.name in seen or seen.add(t.name)})
        if dups:
            errors.append(f"{TOOLS_FILE}: duplicate tool names: {dups}")
    return _finalize(out.get("agents"), tools_list, out.get("settings"), errors, [], builtin_tool_names)


class ConfigWatcher:
    """Detects modifications of the configuration files (mtime + size)."""

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self._stamp = self.snapshot()

    def snapshot(self) -> tuple:
        out = []
        for name in (AGENTS_FILE, TOOLS_FILE, SERVER_FILE):
            p = self.config_dir / name
            try:
                st = p.stat()
                out.append((name, st.st_mtime_ns, st.st_size))
            except FileNotFoundError:
                out.append((name, None, None))
        return tuple(out)

    def changed(self) -> bool:
        current = self.snapshot()
        if current != self._stamp:
            self._stamp = current
            return True
        return False

    def mark_seen(self) -> None:
        self._stamp = self.snapshot()
