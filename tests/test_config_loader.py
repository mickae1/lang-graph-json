import json

import pytest
from conftest import ROOT, TOOLS, agent, write_config

from langgraph_json.utils.config_loader import (
    AgentsConfig,
    ConfigError,
    ConfigWatcher,
    load_config,
    validate_semantics,
)
from langgraph_json.utils.tools import BUILTIN_TOOL_NAMES


def test_shipped_config_is_valid():
    bundle = load_config(ROOT / "config", BUILTIN_TOOL_NAMES)
    assert {"hmi", "analysis", "jobs"} <= set(bundle.agents.enabled_agents())
    assert bundle.warnings == []
    assert bundle.settings.max_tool_calls_per_turn == 20
    assert bundle.settings.token_budget_per_session == 100_000


def test_defaults_applied(tmp_path):
    cfg = write_config(tmp_path, {"a": agent("A", ["open_item"])})
    (cfg / "server.json").unlink()
    bundle = load_config(cfg)
    a = bundle.agents.agents["a"]
    assert a.name == "a" and a.enabled and a.type == "react"
    assert a.requires_confirmation == [] and a.can_handoff_to == [] and a.max_history_messages is None
    assert bundle.settings.host == "127.0.0.1" and bundle.settings.port == 8000


def test_agents_as_list_are_accepted():
    cfg = AgentsConfig.model_validate({"agents": [{"name": "x", **agent("X", ["t"])}]})
    assert list(cfg.agents) == ["x"]


def test_missing_agents_file(tmp_path):
    with pytest.raises(ConfigError, match="agents.json: file not found"):
        load_config(tmp_path)


def test_invalid_json_reports_position(tmp_path):
    cfg = write_config(tmp_path)
    (cfg / "agents.json").write_text('{"agents": {,}}')
    with pytest.raises(ConfigError, match="invalid JSON at line 1"):
        load_config(cfg)


def test_all_errors_are_collected(tmp_path):
    cfg = write_config(
        tmp_path,
        {
            "bad name": agent("x", ["open_item"]),
            "supervisor": agent("x", ["open_item"]),
            "empty": {"description": " ", "system_prompt": "", "allowed_tools": []},
        },
    )
    with pytest.raises(ConfigError) as exc:
        load_config(cfg)
    text = "\n".join(exc.value.errors)
    assert "invalid name" in text
    assert "reserved" in text
    assert "'description' must not be empty" in text
    assert "'system_prompt' must not be empty" in text
    assert "at least one tool" in text


def test_missing_required_field(tmp_path):
    cfg = write_config(tmp_path, {"a": {"description": "x", "allowed_tools": ["t"]}})
    with pytest.raises(ConfigError, match="agents.a.system_prompt"):
        load_config(cfg)


def test_unknown_tools_and_handoffs_are_warnings(tmp_path):
    cfg = write_config(
        tmp_path,
        {"a": agent("A", ["open_item", "ghost_tool"], requires_confirmation=["run_job"], can_handoff_to=["nobody"])},
    )
    bundle = load_config(cfg)
    w = "\n".join(bundle.warnings)
    assert "unknown tool 'ghost_tool'" in w
    assert "not in allowed_tools" in w
    assert "unknown handoff target 'nobody'" in w


def test_disabled_agents_are_not_semantically_checked(tmp_path):
    cfg = write_config(tmp_path, {"a": agent("A", ["open_item"]), "off": {"enabled": False, "description": "", "system_prompt": "", "allowed_tools": []}})
    assert "off" in load_config(cfg).agents.agents


def test_sub_graph_validation():
    cfg = AgentsConfig.model_validate(
        {
            "agents": {
                "a": {"type": "sub_graph", "description": "A", "system_prompt": "p", "sub_agents": ["b"]},
                "b": {"type": "sub_graph", "description": "B", "system_prompt": "p", "sub_agents": ["a"]},
                "c": {"type": "sub_graph", "description": "C", "system_prompt": "p", "sub_agents": []},
                "d": {"type": "sub_graph", "description": "D", "system_prompt": "p", "sub_agents": ["zzz"]},
            }
        }
    )
    errors, _ = validate_semantics(cfg, set())
    text = "\n".join(errors)
    assert "cycle" in text
    assert "needs 'sub_agents'" in text
    assert "unknown sub-agent 'zzz'" in text


def test_tools_file_errors(tmp_path):
    cfg = write_config(tmp_path, tools=[TOOLS[0], TOOLS[0]])
    with pytest.raises(ConfigError, match="duplicate tool names"):
        load_config(cfg)
    bad = dict(TOOLS[0], method="FETCH", parameters={"x": {"type": "date"}})
    write_config(tmp_path, tools=[bad])
    with pytest.raises(ConfigError) as exc:
        load_config(cfg)
    assert any("invalid HTTP method" in e for e in exc.value.errors)
    assert any("unsupported parameter type" in e for e in exc.value.errors)


def test_missing_tools_file_is_a_warning(tmp_path):
    cfg = write_config(tmp_path, {"a": agent("A", ["wait_for_event"])})
    (cfg / "tools.json").unlink()
    bundle = load_config(cfg, BUILTIN_TOOL_NAMES)
    assert bundle.tools == [] and any("tools.json not found" in w for w in bundle.warnings)


def test_server_settings_validation(tmp_path):
    cfg = write_config(tmp_path, server={"port": 0})
    with pytest.raises(ConfigError, match="server.json: port"):
        load_config(cfg)


def test_utf8_bom_is_accepted(tmp_path):
    cfg = write_config(tmp_path)
    content = (cfg / "agents.json").read_text()
    (cfg / "agents.json").write_bytes(b"\xef\xbb\xbf" + content.encode())
    assert load_config(cfg).agents.agents


def test_watcher_detects_changes(tmp_path):
    cfg = write_config(tmp_path)
    w = ConfigWatcher(cfg)
    assert not w.changed()
    data = json.loads((cfg / "agents.json").read_text())
    data["agents"]["hmi"]["description"] = "changed description, longer"
    (cfg / "agents.json").write_text(json.dumps(data))
    assert w.changed()
    assert not w.changed()
