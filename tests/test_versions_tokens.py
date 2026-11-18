"""Schema versions of the JSON files, and renewal of host-pushed tokens."""

import json

import pytest
from conftest import ROOT, TOOLS, agent, write_config
from fastapi.testclient import TestClient

from langgraph_json import FakeChatModel, Runtime, create_app, tools_parser
from langgraph_json.utils.config_loader import SCHEMA_VERSIONS, AgentsConfig, ConfigError, ToolsFile, load_config
from langgraph_json.utils.llm_factory import LLMFactory, ProviderConfig, StaticLLMFactory


# --------------------------------------------------------------------------- #
# Schema versions
# --------------------------------------------------------------------------- #
def test_current_versions_are_accepted():
    assert SCHEMA_VERSIONS == {"agents.json": 1, "tools.json": 1}
    assert AgentsConfig.model_validate({"version": 1, "agents": {}}).version == 1
    assert ToolsFile.model_validate({"version": 1, "tools": []}).version == 1
    assert AgentsConfig.model_validate({"agents": {}}).version == 1  # missing = 1


def test_newer_version_is_refused_with_an_explicit_message(tmp_path):
    cfg = write_config(tmp_path, version=2)
    with pytest.raises(ConfigError) as exc:
        load_config(cfg)
    msg = "\n".join(exc.value.errors)
    assert "agents.json schema version 2 is not supported by lang-graph-json" in msg
    assert "upgrade lang-graph-json" in msg

    write_config(tmp_path)
    (tmp_path / "tools.json").write_text(json.dumps({"version": 3, "tools": TOOLS}))
    with pytest.raises(ConfigError, match="tools.json schema version 3"):
        load_config(tmp_path)


def test_invalid_versions(tmp_path):
    with pytest.raises(ValueError, match="versions start at 1"):
        AgentsConfig.model_validate({"version": 0, "agents": {}})
    with pytest.raises(ValueError):
        ToolsFile.model_validate({"version": "latest", "tools": []})


def test_parser_writes_the_supported_version(tmp_path):
    assert tools_parser.SCHEMA_VERSION == SCHEMA_VERSIONS["tools.json"]
    doc = json.loads((ROOT / "config" / "tools.json").read_text())
    assert doc["version"] == SCHEMA_VERSIONS["tools.json"]


def test_status_and_validate_expose_versions(tmp_path, backend):
    cfg = write_config(tmp_path / "c")
    rt = Runtime(cfg, backend_transport=backend.transport)
    with TestClient(create_app(rt)) as c:
        assert c.get("/status").json()["schema_versions"] == SCHEMA_VERSIONS
        r = c.post("/agents/validate", json={"version": 9, "agents": {}}).json()
        assert r["valid"] is False and "schema version 9" in r["errors"][0]


# --------------------------------------------------------------------------- #
# Token renewal (user-delegated sign-in)
# --------------------------------------------------------------------------- #
GATEWAY = {
    "default": "gateway",
    "providers": {
        "gateway": {"type": "anthropic", "auth": "bearer", "api_key": "user-token-1",
                    "endpoint": "https://gateway.example.com", "default_model": "m", "token_expires_in": 3600},
        "aoai": {"type": "azure_openai", "auth": "bearer", "api_key": "aad-user-token",
                 "endpoint": "https://res.openai.azure.com", "default_model": "d"},
        "secret": {"type": "anthropic", "auth": "azure_ad", "default_model": "m",
                   "azure_ad": {"tenant_id": "t", "client_id": "c", "client_secret": "s", "scope": "x"}},
    },
}


def test_update_token_rebuilds_only_the_client():
    f = LLMFactory(ProviderConfig.model_validate(GATEWAY))
    assert 3590 < f.token_expiry["gateway"] - __import__("time").time() <= 3600
    first = f.get()
    assert first.default_headers["Authorization"] == "Bearer user-token-1"
    assert f.update_token(None, "user-token-2", expires_in=60) == "gateway"
    second = f.get()
    assert second is not first and second.default_headers["Authorization"] == "Bearer user-token-2"
    f.update_token("gateway", "user-token-3")  # no lifetime given: expiry unknown
    assert "gateway" not in f.token_expiry
    with pytest.raises(ValueError, match="uses auth 'azure_ad'"):
        f.update_token("secret", "x")
    with pytest.raises(ValueError, match="empty token"):
        f.update_token(None, "")


def test_azure_openai_with_a_user_token():
    f = LLMFactory(ProviderConfig.model_validate(GATEWAY))
    llm = f.get("aoai")
    assert llm.azure_ad_token.get_secret_value() == "aad-user-token"
    f.update_token("aoai", "renewed")
    assert f.get("aoai").azure_ad_token.get_secret_value() == "renewed"


def test_static_factory_refuses_token_updates():
    with pytest.raises(ValueError, match="created in code"):
        StaticLLMFactory(FakeChatModel()).update_token(None, "t")


def test_token_endpoint_and_redaction(tmp_path, backend):
    cfg = write_config(tmp_path / "c")
    rt = Runtime(cfg, backend_transport=backend.transport)
    with TestClient(create_app(rt)) as c:
        assert c.post("/config/provider/token", json={"token": "t"}).status_code == 503  # no provider yet
        assert c.post("/config/provider", json=GATEWAY).status_code == 200
        r = c.post("/config/provider/token", json={"token": "fresh-user-token", "expires_in": 1800})
        assert r.status_code == 200 and r.json()["provider"] == "gateway" and r.json()["expires_at"]
        got = c.get("/config/provider").json()
        assert "fresh-user-token" not in json.dumps(got) and "user-token-1" not in json.dumps(got)
        assert 1790 < got["token_expires_in_s"]["gateway"] <= 1800
        assert c.get("/status").json()["token_expires_in_s"]["gateway"] > 0
        assert c.post("/config/provider/token", json={"token": "x", "provider": "secret"}).status_code == 400
        assert c.post("/config/provider/token", json={"token": "x", "provider": "ghost"}).status_code == 503
        assert c.post("/config/provider/token", json={"token": ""}).status_code == 422
        graph_before = rt.graph
        c.post("/config/provider/token", json={"token": "again"})
        assert rt.graph is graph_before  # no rebuild


async def test_expiry_warnings_are_sent_once(make_runtime):
    rt = await make_runtime(agents={"solo": agent("solo", ["open_item"])})
    rt.set_provider(ProviderConfig.model_validate({**GATEWAY}))
    q = rt.events.subscribe(None)
    expiry = rt.llms.token_expiry["gateway"]
    await rt.maintenance(now=expiry - 3000)
    assert q.empty()
    await rt.maintenance(now=expiry - 200)
    ev = q.get_nowait()
    assert ev["event_type"] == "provider.token_expiring" and ev["data"]["provider"] == "gateway"
    await rt.maintenance(now=expiry - 100)
    assert q.empty()  # warned once per token
    rt.set_token("renewed", expires_in=10)
    await rt.maintenance(now=rt.llms.token_expiry["gateway"] + 1)
    ev = q.get_nowait()
    assert ev["event_type"] == "provider.token_expired" and ev["data"]["expires_in_s"] == 0
