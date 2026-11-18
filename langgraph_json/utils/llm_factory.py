"""LLM factory.

Provider configuration (credentials, endpoints, models) is pushed at runtime
via ``POST /config/provider`` and only ever kept in memory. Proxy settings are
pushed via ``POST /config/proxy``.

Supported provider types:

=================== ===================== ==========================
type                LangChain client      auth
=================== ===================== ==========================
``anthropic``       ``ChatAnthropic``     api_key | azure_ad | bearer
``azure_openai``    ``AzureChatOpenAI``   api_key | azure_ad
``openai_compatible`` ``ChatOpenAI``      bearer | api_key | none
``ollama``          ``ChatOllama``        none
``fake``            scripted fake model   none (development / tests only)
=================== ===================== ==========================
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import httpx
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

log = logging.getLogger(__name__)

ProviderType = Literal["anthropic", "azure_openai", "openai_compatible", "ollama", "fake"]
AuthType = Literal["api_key", "azure_ad", "bearer", "none"]


class ProviderNotConfigured(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Configuration models (in-memory only)
# --------------------------------------------------------------------------- #
class AzureAdConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tenant_id: str
    client_id: str
    client_secret: SecretStr
    scope: str
    authority_host: str = "https://login.microsoftonline.com"


class ProviderEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: ProviderType
    auth: AuthType = "api_key"
    endpoint: str | None = None
    api_key: SecretStr | None = None
    api_version: str | None = None  # azure_openai
    azure_ad: AzureAdConfig | None = None
    models: list[str] = Field(default_factory=list)
    default_model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_s: float = 120.0
    headers: dict[str, str] = Field(default_factory=dict)
    # Only for type == "fake": scripted responses (see fake_llm.FakeChatModel).
    fake_responses: list[Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> "ProviderEntry":
        if self.auth == "azure_ad" and self.azure_ad is None:
            raise ValueError("auth 'azure_ad' requires an 'azure_ad' block")
        if self.auth in ("api_key", "bearer") and self.type in ("anthropic", "azure_openai") and self.api_key is None:
            raise ValueError(f"auth '{self.auth}' requires 'api_key'")
        if self.type == "azure_openai" and not self.endpoint:
            raise ValueError("azure_openai requires 'endpoint'")
        return self

    def resolve_model(self, requested: str | None) -> str:
        model = requested or self.default_model or (self.models[0] if self.models else None)
        if not model:
            if self.type == "fake":
                return "fake-model"
            raise ProviderNotConfigured("no model given and the provider has no default_model/models")
        return model


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default: str
    providers: dict[str, ProviderEntry]

    @model_validator(mode="before")
    @classmethod
    def _single_provider_shortcut(cls, data: Any) -> Any:
        # Accept a single provider object: {"type": "anthropic", ...}
        if isinstance(data, dict) and "type" in data and "providers" not in data:
            return {"default": "default", "providers": {"default": data}}
        return data

    @model_validator(mode="after")
    def _check_default(self) -> "ProviderConfig":
        if self.default not in self.providers:
            raise ValueError(f"default provider '{self.default}' is not defined in 'providers'")
        return self

    def redacted(self) -> dict:
        out = {"default": self.default, "providers": {}}
        for name, p in self.providers.items():
            d = p.model_dump(exclude={"api_key", "azure_ad", "fake_responses", "headers"})
            d["api_key"] = "***" if p.api_key else None
            if p.azure_ad:
                d["azure_ad"] = {
                    "tenant_id": p.azure_ad.tenant_id,
                    "client_id": p.azure_ad.client_id,
                    "scope": p.azure_ad.scope,
                    "client_secret": "***",
                }
            d["headers"] = sorted(p.headers)
            out["providers"][name] = d
        return out


class ProxyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _split_no_proxy(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("no_proxy"), str):
            data = {**data, "no_proxy": [s.strip() for s in data["no_proxy"].split(",") if s.strip()]}
        return data

    def bypass(self, url: str | None) -> bool:
        host = (urlparse(url).hostname or "") if url else ""
        if host in ("localhost", "127.0.0.1", "::1"):
            return True
        for entry in self.no_proxy:
            e = entry.lstrip(".").lower()
            if e == "*" or host.lower() == e or host.lower().endswith("." + e):
                return True
        return False

    def proxy_for(self, url: str | None) -> str | None:
        """Proxy URL to use for ``url`` (None = direct connection)."""
        if url is not None and self.bypass(url):
            return None
        scheme = urlparse(url).scheme if url else "https"
        if scheme == "http":
            return self.http_proxy or self.https_proxy
        return self.https_proxy or self.http_proxy

    def httpx_mounts(self) -> dict[str, httpx.AsyncHTTPTransport | None]:
        return self._mounts(httpx.AsyncHTTPTransport)

    def httpx_sync_mounts(self) -> dict[str, httpx.HTTPTransport | None]:
        return self._mounts(httpx.HTTPTransport)

    def _mounts(self, transport_cls) -> dict:
        mounts: dict[str, Any] = {}
        for entry in ["localhost", "127.0.0.1", *self.no_proxy]:
            e = entry.lstrip(".")
            if e == "*":
                return {}
            mounts[f"all://{e}"] = None
            mounts[f"all://*.{e}"] = None
        if self.http_proxy:
            mounts["http://"] = transport_cls(proxy=self.http_proxy)
        if self.https_proxy or self.http_proxy:
            mounts["https://"] = transport_cls(proxy=self.https_proxy or self.http_proxy)
        return mounts if (self.http_proxy or self.https_proxy) else {}


# --------------------------------------------------------------------------- #
# Azure AD (OAuth2 client-credentials) token provider
# --------------------------------------------------------------------------- #
class AzureAdTokenProvider:
    """Fetches and caches an OAuth2 access token (client-credentials flow)."""

    REFRESH_MARGIN_S = 300

    def __init__(self, cfg: AzureAdConfig, proxy: ProxyConfig | None = None, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.proxy = proxy or ProxyConfig()
        self._transport = transport
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    @property
    def token_url(self) -> str:
        return f"{self.cfg.authority_host.rstrip('/')}/{self.cfg.tenant_id}/oauth2/v2.0/token"

    def __call__(self) -> str:
        return self.get_token()

    def get_token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at - self.REFRESH_MARGIN_S:
                return self._token
            kwargs: dict[str, Any] = {"timeout": 30}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            elif self.proxy.proxy_for(self.token_url):
                kwargs["proxy"] = self.proxy.proxy_for(self.token_url)
            with httpx.Client(**kwargs) as client:
                resp = client.post(
                    self.token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.cfg.client_id,
                        "client_secret": self.cfg.client_secret.get_secret_value(),
                        "scope": self.cfg.scope,
                    },
                )
            if resp.status_code != 200:
                raise RuntimeError(f"Azure AD token request failed ({resp.status_code}): {resp.text[:300]}")
            payload = resp.json()
            self._token = payload["access_token"]
            self._expires_at = time.time() + float(payload.get("expires_in", 3600))
            return self._token

    async def aget_token(self) -> str:
        import asyncio

        return await asyncio.to_thread(self.get_token)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
Builder = Callable[[ProviderEntry, str, ProxyConfig, "LLMFactory"], BaseChatModel]


class LLMFactory:
    """Creates (and caches) chat models from the in-memory provider config."""

    _builders: dict[str, Builder] = {}

    def __init__(self, providers: ProviderConfig | None = None, proxy: ProxyConfig | None = None):
        self.providers = providers
        self.proxy = proxy or ProxyConfig()
        self._cache: dict[tuple[str, str], tuple[BaseChatModel, str | None]] = {}
        self._token_providers: dict[str, AzureAdTokenProvider] = {}
        self._lock = threading.Lock()

    @classmethod
    def register(cls, provider_type: str, builder: Builder) -> None:
        cls._builders[provider_type] = builder

    @property
    def configured(self) -> bool:
        return self.providers is not None

    def provider_entry(self, name: str | None = None) -> tuple[str, ProviderEntry]:
        if self.providers is None:
            raise ProviderNotConfigured("no LLM provider configured — call POST /config/provider first")
        name = name or self.providers.default
        if name not in self.providers.providers:
            raise ProviderNotConfigured(f"unknown provider '{name}'")
        return name, self.providers.providers[name]

    def token_provider(self, name: str, entry: ProviderEntry) -> AzureAdTokenProvider:
        tp = self._token_providers.get(name)
        if tp is None:
            tp = AzureAdTokenProvider(entry.azure_ad, self.proxy)
            self._token_providers[name] = tp
        return tp

    def get(self, provider: str | None = None, model: str | None = None) -> BaseChatModel:
        name, entry = self.provider_entry(provider)
        model_name = entry.resolve_model(model)
        key = (name, model_name)
        # Anthropic behind an Azure AD gateway receives the token as a static
        # header: rebuild the client whenever the token changes.
        token = None
        if entry.type == "anthropic" and entry.auth == "azure_ad":
            token = self.token_provider(name, entry).get_token()
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[1] == token:
                return cached[0]
            builder = self._builders.get(entry.type)
            if builder is None:
                raise ProviderNotConfigured(f"unsupported provider type '{entry.type}'")
            llm = builder(entry, model_name, self.proxy, self, provider_name=name, token=token)
            self._cache[key] = (llm, token)
            return llm

    def provider_names(self) -> set[str] | None:
        """Known provider names (None = not configured yet)."""
        return set(self.providers.providers) if self.providers is not None else None

    def available_models(self) -> dict[str, list[str]]:
        if self.providers is None:
            return {}
        return {
            n: list(dict.fromkeys(([p.default_model] if p.default_model else []) + p.models))
            for n, p in self.providers.providers.items()
        }


def _common_kwargs(entry: ProviderEntry) -> dict:
    kw: dict[str, Any] = {}
    if entry.temperature is not None:
        kw["temperature"] = entry.temperature
    if entry.max_tokens is not None:
        kw["max_tokens"] = entry.max_tokens
    return kw


def _build_anthropic(entry, model, proxy, factory, provider_name=None, token=None):
    from langchain_anthropic import ChatAnthropic

    headers = dict(entry.headers)
    api_key = entry.api_key.get_secret_value() if entry.api_key else "unused"
    if entry.auth == "azure_ad":
        headers["Authorization"] = f"Bearer {token}"
    elif entry.auth == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    kw = _common_kwargs(entry)
    if entry.endpoint:
        kw["base_url"] = entry.endpoint
    p = proxy.proxy_for(entry.endpoint or "https://api.anthropic.com")
    if p:
        kw["anthropic_proxy"] = p
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        default_headers=headers or None,
        default_request_timeout=entry.timeout_s,
        max_retries=2,
        **kw,
    )


def _openai_http_clients(proxy: ProxyConfig, timeout: float) -> dict:
    mounts = proxy.httpx_mounts()
    if not mounts:
        return {}
    return {
        "http_client": httpx.Client(mounts=proxy.httpx_sync_mounts(), timeout=timeout),
        "http_async_client": httpx.AsyncClient(mounts=mounts, timeout=timeout),
    }


def _build_azure_openai(entry, model, proxy, factory, provider_name=None, token=None):
    from langchain_openai import AzureChatOpenAI

    kw = _common_kwargs(entry)
    if entry.auth == "azure_ad":
        tp = factory.token_provider(provider_name, entry)
        kw["azure_ad_token_provider"] = tp.get_token
        kw["azure_ad_async_token_provider"] = tp.aget_token
    else:
        kw["api_key"] = entry.api_key.get_secret_value()
    return AzureChatOpenAI(
        azure_deployment=model,
        azure_endpoint=entry.endpoint,
        api_version=entry.api_version or "2024-10-21",
        default_headers=entry.headers or None,
        timeout=entry.timeout_s,
        **_openai_http_clients(proxy, entry.timeout_s),
        **kw,
    )


def _build_openai_compatible(entry, model, proxy, factory, provider_name=None, token=None):
    from langchain_openai import ChatOpenAI

    api_key = entry.api_key.get_secret_value() if entry.api_key else "not-needed"
    return ChatOpenAI(
        model=model,
        base_url=entry.endpoint,
        api_key=api_key,
        default_headers=entry.headers or None,
        timeout=entry.timeout_s,
        **_openai_http_clients(proxy, entry.timeout_s),
        **_common_kwargs(entry),
    )


def _build_ollama(entry, model, proxy, factory, provider_name=None, token=None):
    from langchain_ollama import ChatOllama

    base_url = entry.endpoint or "http://127.0.0.1:11434"
    client_kwargs: dict[str, Any] = {"timeout": entry.timeout_s}
    if entry.headers:
        client_kwargs["headers"] = entry.headers
    p = proxy.proxy_for(base_url)
    if p:
        client_kwargs["proxy"] = p
    kw: dict[str, Any] = {}
    if entry.temperature is not None:
        kw["temperature"] = entry.temperature
    if entry.max_tokens is not None:
        kw["num_predict"] = entry.max_tokens
    return ChatOllama(model=model, base_url=base_url, client_kwargs=client_kwargs, **kw)


def _build_fake(entry, model, proxy, factory, provider_name=None, token=None):
    from .fake_llm import FakeChatModel

    return FakeChatModel(responses=list(entry.fake_responses or []), model_name=model)


LLMFactory.register("anthropic", _build_anthropic)
LLMFactory.register("azure_openai", _build_azure_openai)
LLMFactory.register("openai_compatible", _build_openai_compatible)
LLMFactory.register("ollama", _build_ollama)
LLMFactory.register("fake", _build_fake)


class StaticLLMFactory(LLMFactory):
    """Factory serving chat models created by the caller (library use).

    ``models`` maps a provider name to a model instance; the first entry (or
    ``default``) is the default provider. Agents selecting a ``provider`` get
    that instance; the ``model`` field of agents is ignored.
    """

    def __init__(self, models: "BaseChatModel | dict[str, BaseChatModel]", default: str | None = None, proxy: ProxyConfig | None = None):
        super().__init__(None, proxy)
        self.models = {"default": models} if isinstance(models, BaseChatModel) else dict(models)
        if not self.models:
            raise ValueError("StaticLLMFactory needs at least one model")
        self.default = default or next(iter(self.models))
        if self.default not in self.models:
            raise ValueError(f"default model '{self.default}' not in {sorted(self.models)}")

    @property
    def configured(self) -> bool:
        return True

    def get(self, provider: str | None = None, model: str | None = None) -> BaseChatModel:
        name = provider or self.default
        if name not in self.models:
            raise ProviderNotConfigured(f"unknown provider '{name}' (available: {sorted(self.models)})")
        return self.models[name]

    def provider_names(self) -> set[str]:
        return set(self.models)

    def available_models(self) -> dict[str, list[str]]:
        return {n: [getattr(m, "model", None) or getattr(m, "model_name", None) or type(m).__name__] for n, m in self.models.items()}
