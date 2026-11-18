# CLAUDE.md

Guidance for Claude Code (and other contributors) working in this repository.

## What this is

`lang-graph-json` is a **generic** LangGraph multi-agent engine driven by JSON files:
`agents.json` (agents), `tools.json` (tools = REST endpoints or Python functions) and `server.json`
(operational settings). It ships in two forms:

- a **standalone server**: a PyInstaller `--onedir` bundle, driven over REST/WebSocket by a host
  application written in any language;
- a **Python library**: `Runtime` used in-process, or `create_app()` mounted in a FastAPI app.

The project is domain-agnostic. Do not add references to a specific product, customer or business domain
in code, tests, examples or docs. Examples use neutral concepts (items, projects, jobs, reports).

## Commands

```bash
pip install -e ".[dev]"                                   # setup (Python >= 3.11)
pytest                                                    # offline suite, must stay green
pytest --cov=langgraph_json --cov-report=term-missing     # coverage must stay at 100 %
pytest -m live -v                                         # real LLM; needs ANTHROPIC_API_KEY (or LGJ_TEST_PROVIDER_JSON) in env/.env
python scripts/build.py all                               # wheel + PyInstaller bundle, with smoke tests
pytest -m dist -v                                         # tests of dist/lang-graph-json (after build)
python scripts/generate_tools_json.py --source-dir examples/cpp --output config/tools.json   # regenerate
python scripts/generate_tools_json.py --source-dir examples/cpp --output config/tools.json --check
lang-graph-json --config-dir config --port 8000           # run the server
pytest tests/test_qmake.py                                # qmake integration (needs qmake + make on PATH, or LGJ_QMAKE / LGJ_MAKE)
```

The parser is wired into builds via `qmake/ai_metadata.pri` (qmake) and `examples/cpp/CMakeLists.txt`
(CMake). In the `.pri`, keep paths as `D:/...` (`absolute_path`, no `shell_path`): native Python must
accept them from cmd, PowerShell and MSYS shells alike.

## Architecture (where things live)

- `langgraph_json/agent.py`: `GraphBuilder` builds the `StateGraph`: a `supervisor` node, one node per
  agent, plus `<agent>__tools`. A `sub_graph` agent is a nested compiled graph wrapped in a node.
  `make_graph()` is the LangGraph CLI/Studio entry point (`langgraph.json`).
- `langgraph_json/utils/nodes.py`: node functions.
  - The supervisor routes with a `route` tool. With a single member, routing is deterministic.
  - Agents see other agents' work as text (`render_for_agent`), so the history is always valid for any
    provider.
  - The tools node enforces the whitelist, the per-turn limit and the confirmations.
- `langgraph_json/utils/state.py`: `AgentState`. `turn_input()` resets the per-turn fields.
  `tokens_used` is an additive reducer, accumulated per session.
- `langgraph_json/gateway/runtime.py`: `Runtime` owns:
  - the config and the graph lifecycle (rebuilt on reload);
  - the single checkpointer (memory, SQLite or injected; history survives reloads), and the sessions
    restored from it at startup;
  - the confirmations (`interrupt` / `Command(resume=...)`);
  - turns run as cancellable tasks (`_execute`, `cancel`, `_close_turn`);
  - the maintenance task (session TTL, confirmation timeouts);
  - the dry-run `validate`.
- `langgraph_json/gateway/app.py`: FastAPI routes and the `WS /stream` endpoint.
- `langgraph_json/utils/tools.py`: `ToolRegistry`, REST tools, Python tools (`PythonToolLoader`: module
  whitelist + `plugins_dir`, reloaded on hot-reload), `ctx` injection, schemas.
- `langgraph_json/utils/config_loader.py`: pydantic schemas + cross-file validation. `load_config` (files)
  and `config_from_data` (objects) share `_finalize`.
- `langgraph_json/utils/llm_factory.py`: providers kept in memory only, Azure AD tokens, proxy,
  `StaticLLMFactory` (models injected by library users).
- `langgraph_json/utils/events.py`: LangGraph `astream_events(v2)` → unified events + `EventBus`.
- `langgraph_json/tools_parser.py`: the generic `AI_METADATA` → `tools.json` parser. **Standard library
  only** (it is copied as a standalone script into the bundle).
- `langgraph_json/utils/fake_llm.py`: `FakeChatModel`, scripted, with tool calls and streaming. Provider
  type `fake`.

## Invariants: do not break

- **No secrets on disk.** Provider and proxy settings only come from `POST /config/provider` /
  `POST /config/proxy` (or `llm=` in library mode). Redact them in every response or log.
- **The whitelist is the permission system.** Agents only get `allowed_tools`, and the tools node
  re-checks it at execution time. Do not add permission flags.
- **Confirmations ask everything before executing anything.** `interrupt()` re-runs the node on
  resume, so a tool must never run before every decision of the step is known.
- **Python tools from config are restricted** to `server.json` → `python_tool_modules`. Functions passed in
  code (`Runtime(functions=...)`) are trusted.
- **Invalid config never replaces a valid one**: reload/update raise `ConfigError` (with every error) and
  keep the previous state.
- **No persistence must keep working**: `persistence: "memory"` is the default and needs no extra
  dependency. `langgraph-checkpoint-sqlite` is an optional extra (`[sqlite]`), imported only when
  `persistence` is `"sqlite"`.
- **Cancelling never leaves an invalid history**: `_close_turn` answers every dangling tool call, so the
  next LLM call is valid. Timeouts and cancellations end in the same `_close_turn`.
- **Validation is a dry run**: `Runtime.validate` builds everything but never assigns runtime state or
  writes files.
- **Localhost only** unless `--allow-remote`. The backend client never uses a proxy (`trust_env=False`).
- The parser output is **deterministic** (sorted, no timestamp). `config/tools.json` must equal the output
  for `examples/cpp` (a test checks it).
- `import langgraph_json` stays light (lazy exports in `__init__.py`).

## Conventions

- Code, comments and docs in English. Python 3.11+, type hints, `from __future__ import annotations`.
- Tests:
  - use `FakeChatModel` scripts (`route(...)`, `call(...)` helpers in `tests/conftest.py`) and the `Backend`
    mock (`httpx.MockTransport`); never real network in the offline suite;
  - `make_runtime` (fixture) builds a runtime from a temp config;
  - keep coverage at 100 %, and use `# pragma: no cover` only for purely defensive branches.
- Live tests (`tests/test_live.py`) assert behaviour (tools called, arguments, routing), not exact wording.
- When adding a server setting: add it to `ServerSettings`, `config/server.json`, the README table and a test.
- When adding an HTTP endpoint: add it to `gateway/app.py`, the README API table and `tests/test_gateway.py`.
- When adding a language profile to the parser: add it to `PROFILES`. The parametrised test covers the basics.
- Windows: write files with `encoding="utf-8"` (the default locale encoding is cp1252).

## Packaging notes

- `lang-graph-json.spec`:
  - bundles the LangChain providers, uvicorn, websockets and a list of stdlib modules for plugins;
  - `LGJ_BUNDLE_MODULES` / `LGJ_BUNDLE_PATHS` / `LGJ_HIDDEN_IMPORTS` embed extra code.
- `scripts/build.py` lays out `dist/lang-graph-json/` (`config/`, `plugins/`, standalone parser) and
  smoke-tests the bundle and the wheel. The wheel is tested in a fresh virtual environment.
- PyInstaller does not cross-compile: build on each target OS (the CI does Linux + Windows).
