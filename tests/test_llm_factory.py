import json
import time

import httpx
import pytest
from pydantic import ValidationError

from langgraph_json.utils.fake_llm import FakeChatModel
from langgraph_json.utils.llm_factory import (
    AzureAdConfig,
    AzureAdTokenProvider,
    LLMFactory,
    ProviderConfig,
    ProviderNotConfigured,
    ProxyConfig,
)

MULTI = {
    "default": "gateway",
    "providers": {
        "gateway": {
            "type": "anthropic",
            "auth": "api_key",
            "api_key": "sk-secret",
            "endpoint": "https://llm.example.com",
            "models": ["model-large", "model-small"],
            "default_model": "model-large",
        },
        "local": {"type": "ollama", "endpoint": "http://127.0.0.1:11434", "models": ["llama3"]},
        "compat": {"type": "openai_compatible", "auth": "bearer", "api_key": "tok", "endpoint": "https://compat.example.com/v1", "default_model": "m"},
        "azure": {
            "type": "azure_openai",
            "auth": "azure_ad",
            "endpoint": "https://res.openai.azure.com",
            "default_model": "deployment-1",
            "azure_ad": {"tenant_id": "t", "client_id": "c", "client_secret": "s3cr3t", "scope": "api://x/.default"},
        },
    },
}


def test_single_provider_shortcut():
    cfg = ProviderConfig.model_validate({"type": "fake"})
    assert cfg.default == "default" and cfg.providers["default"].type == "fake"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"default": "x", "providers": {}}, "not defined"),
        ({"type": "anthropic"}, "requires 'api_key'"),
        ({"type": "anthropic", "auth": "azure_ad"}, "requires an 'azure_ad'"),
        ({"type": "azure_openai", "api_key": "k"}, "requires 'endpoint'"),
        ({"type": "unknown"}, "type"),
    ],
)
def test_invalid_provider_configs(payload, message):
    with pytest.raises(ValidationError, match=message):
        ProviderConfig.model_validate(payload)


def test_redaction_hides_secrets():
    red = json.dumps(ProviderConfig.model_validate(MULTI).redacted())
    assert "sk-secret" not in red and "s3cr3t" not in red and "tok" not in red.replace("token", "")
    assert '"api_key": "***"' in red


def test_builders_create_expected_clients():
    f = LLMFactory(ProviderConfig.model_validate(MULTI))
    a = f.get()
    assert type(a).__name__ == "ChatAnthropic" and a.model == "model-large"
    assert f.get(model="model-small").model == "model-small"
    assert f.get() is a  # cached
    assert type(f.get("local")).__name__ == "ChatOllama"
    compat = f.get("compat")
    assert type(compat).__name__ == "ChatOpenAI" and compat.model_name == "m"
    azure = f.get("azure")
    assert type(azure).__name__ == "AzureChatOpenAI" and azure.azure_ad_token_provider is not None
    assert f.available_models()["gateway"] == ["model-large", "model-small"]


def test_unconfigured_and_unknown_provider():
    with pytest.raises(ProviderNotConfigured, match="POST /config/provider"):
        LLMFactory().get()
    f = LLMFactory(ProviderConfig.model_validate(MULTI))
    with pytest.raises(ProviderNotConfigured, match="unknown provider"):
        f.get("nope")
    with pytest.raises(ProviderNotConfigured, match="no model"):
        LLMFactory(ProviderConfig.model_validate({"type": "ollama"})).get()


def test_proxy_rules():
    p = ProxyConfig.model_validate({"http_proxy": "http://p:3128", "https_proxy": "http://ps:3128", "no_proxy": "internal.corp, .local"})
    assert p.no_proxy == ["internal.corp", ".local"]
    assert p.proxy_for("https://api.example.com") == "http://ps:3128"
    assert p.proxy_for("http://api.example.com") == "http://p:3128"
    assert p.proxy_for("https://llm.internal.corp/v1") is None
    assert p.proxy_for("http://box.local") is None
    assert p.proxy_for("http://127.0.0.1:11434") is None
    assert "https://" in p.httpx_mounts() and p.httpx_mounts()["all://internal.corp"] is None
    assert ProxyConfig().httpx_mounts() == {}


def test_proxy_is_applied_to_clients():
    proxy = ProxyConfig(https_proxy="http://proxy:8080")
    f = LLMFactory(ProviderConfig.model_validate(MULTI), proxy)
    assert f.get().anthropic_proxy == "http://proxy:8080"
    assert f.get("compat").http_async_client is not None
    assert "proxy" not in f.get("local").client_kwargs  # localhost bypass


def _token_transport(counter):
    def handler(request: httpx.Request) -> httpx.Response:
        counter.append(dict(httpx.QueryParams(request.content.decode())))
        return httpx.Response(200, json={"access_token": f"tok-{len(counter)}", "expires_in": 3600})

    return httpx.MockTransport(handler)


def test_azure_ad_token_is_cached_and_refreshed():
    calls = []
    cfg = AzureAdConfig(tenant_id="tenant", client_id="cid", client_secret="sec", scope="api://x/.default")
    tp = AzureAdTokenProvider(cfg, transport=_token_transport(calls))
    assert tp.token_url == "https://login.microsoftonline.com/tenant/oauth2/v2.0/token"
    assert tp() == "tok-1" and tp() == "tok-1"
    assert calls[0]["grant_type"] == "client_credentials" and calls[0]["client_secret"] == "sec"
    tp._expires_at = time.time() + 10  # inside the refresh margin
    assert tp() == "tok-2"


def test_azure_ad_token_error():
    tp = AzureAdTokenProvider(
        AzureAdConfig(tenant_id="t", client_id="c", client_secret="s", scope="x"),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, text="invalid_client")),
    )
    with pytest.raises(RuntimeError, match="401.*invalid_client"):
        tp.get_token()


def test_anthropic_behind_azure_ad_rebuilds_on_new_token():
    cfg = ProviderConfig.model_validate(
        {
            "type": "anthropic",
            "auth": "azure_ad",
            "endpoint": "https://gateway.example.com",
            "default_model": "m",
            "azure_ad": {"tenant_id": "t", "client_id": "c", "client_secret": "s", "scope": "x"},
        }
    )
    f = LLMFactory(cfg)
    calls = []
    f._token_providers["default"] = AzureAdTokenProvider(cfg.providers["default"].azure_ad, transport=_token_transport(calls))
    first = f.get()
    assert first.default_headers["Authorization"] == "Bearer tok-1"
    assert f.get() is first
    f._token_providers["default"]._expires_at = 0
    second = f.get()
    assert second is not first and second.default_headers["Authorization"] == "Bearer tok-2"


async def test_fake_model_script_and_tools():
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    llm = FakeChatModel(responses=[{"content": "calc", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]}, "done"])
    r1 = await llm.bind_tools([add]).ainvoke([HumanMessage(content="hi")])
    assert r1.tool_calls[0]["name"] == "add" and r1.usage_metadata["total_tokens"] > 0
    assert llm.calls[0]["tools"] == ["add"]
    assert (await llm.ainvoke([HumanMessage(content="x")])).content == "done"
    assert (await llm.ainvoke([HumanMessage(content="ping")])).content == "echo: ping"
    chunks = [c async for c in llm.astream([HumanMessage(content="a b c")])]
    assert "".join(c.content for c in chunks) == "echo: a b c"
