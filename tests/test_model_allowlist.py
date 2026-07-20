"""Security T1-A — the caller-supplied `model` field must be allowlisted.

Without this, any caller (including an anonymous portal user on the public
key) can pass `model: "anthropic/claude-sonnet-4-5"` or `"openai/gpt-4o"`
and route the request to a premium cloud model on our server-side keys. The
daily caps bound request *count*, not *which model*, so they don't stop it.

The control: callers may select only the configured default_model (or a tag
explicitly added to settings.allowed_models). Cloud/fallback models are
server-internal only and never caller-selectable. See
workspace docs/security-review-2026-07-19.md (T1-A).

Integration tests (HTTP behaviour across all five LLM endpoints) come first,
then focused unit tests for the resolver's edge cases.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from pydantic import SecretStr

from koleslaw_api.config import Settings
from koleslaw_api.services.llm import LLMService, resolve_allowed_model

PUBLIC_KEY = "pk_live_fNuYW07Bf16zLFLbghxSPERYV4QS1SWF4-OuLJ5JcCE"

# Models a caller must never be able to select.
CLOUD_MODELS = [
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]

# A local-looking tag that is NOT served. Naming it would 404 at Ollama and,
# because NotFoundError is an availability error for ollama_chat/* models,
# fall back to the cloud model — the exact bypass a prefix-only rule misses.
BOGUS_LOCAL_MODEL = "ollama_chat/prompt-enhance-99b-ultra"


# ── Integration: rejection (the security-critical path) ──────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CLOUD_MODELS)
async def test_enhance_sync_rejects_cloud_model(client: AsyncClient, model: str) -> None:
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem", "model": model},
    )
    assert response.status_code == 400
    assert "not available" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_enhance_sync_rejects_bogus_local_model(client: AsyncClient) -> None:
    """Exact-match, not prefix: an unserved ollama_chat/* tag must 400, not
    404-into-the-cloud-fallback."""
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem", "model": BOGUS_LOCAL_MODEL},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_enhance_stream_rejects_cloud_model(client: AsyncClient) -> None:
    """The streaming path must reject before the StreamingResponse starts."""
    response = await client.post(
        "/v1/enhance",
        json={"prompt": "Write a poem", "model": "openai/gpt-4o"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_rejects_cloud_model(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "model": "anthropic/claude-sonnet-4-5"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_sync_rejects_cloud_model(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/sync",
        json={"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-4o"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_upload_rejects_cloud_model(client: AsyncClient) -> None:
    """Multipart path parses `model` from a Form field — must reject too."""
    response = await client.post(
        "/v1/chat/upload",
        data={
            "messages": json.dumps([{"role": "user", "content": "hi"}]),
            "model": "anthropic/claude-sonnet-4-5",
        },
    )
    assert response.status_code == 400


# ── Integration: the allowed model still works ───────────────────────────


@pytest.mark.asyncio
async def test_enhance_sync_allows_default_model(client: AsyncClient, settings: Settings, mock_redis) -> None:
    """The default model is accepted (proven via a cache hit, no LLM call)."""
    mock_redis.get.return_value = "cached result"
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem", "model": settings.default_model},
    )
    assert response.status_code == 200
    assert response.json()["cached"] is True


@pytest.mark.asyncio
async def test_enhance_sync_allows_omitted_model(client: AsyncClient, mock_redis) -> None:
    """Omitting `model` (the portal's behaviour) resolves to the default."""
    mock_redis.get.return_value = "cached result"
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem"},
    )
    assert response.status_code == 200
    assert response.json()["cached"] is True


@pytest.mark.asyncio
async def test_chat_sync_allows_and_forwards_resolved_model(client: AsyncClient, settings: Settings) -> None:
    """Chat has no service/cache layer, so this proves the route forwards the
    resolved (default) model to acompletion and returns 200."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "hello there"
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 7
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = resp
        response = await client.post(
            "/v1/chat/sync",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json()["model_used"] == settings.default_model
    assert mock_llm.call_args.kwargs["model"] == settings.default_model


# ── Integration: the real anon path (cap admits, then model is rejected) ──


@pytest.mark.asyncio
async def test_anon_public_key_caller_still_rejected(client: AsyncClient, mock_redis) -> None:
    """The actual threat: an anonymous caller on the public key. The per-IP cap
    (a dependency) runs first and admits the request; the route body then still
    rejects the disallowed model with 400. Proves validation isn't shadowed by
    the auth/cap dependency for a real capped caller."""
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, True])  # count=1, under the cap
    mock_redis.pipeline = MagicMock(return_value=pipe)

    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem", "model": "openai/gpt-4o"},
        headers={"Authorization": f"Bearer {PUBLIC_KEY}", "CF-Connecting-IP": "203.0.113.7"},
    )
    assert response.status_code == 400
    assert "not available" in response.json()["detail"].lower()


# ── /v1/models must advertise only what a caller can actually select ─────


def _mock_ollama_tags(*names: str):
    """Patch the Ollama /api/tags call in get_available_models to return `names`."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"models": [{"name": n} for n in names]})
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return patch("koleslaw_api.services.llm.httpx.AsyncClient", return_value=client)


@pytest.mark.asyncio
async def test_models_endpoint_excludes_cloud_and_unserved() -> None:
    """/v1/models must not advertise models the allowlist would 400 — otherwise
    the discovery endpoint contradicts the enhance endpoint (T1-A tail)."""
    settings = Settings(
        default_model="ollama_chat/prompt-enhance-14b-think",
        anthropic_api_key=SecretStr("sk-test"),  # would add cloud entries
        openai_api_key=SecretStr("sk-test"),
    )
    service = LLMService(settings)
    with _mock_ollama_tags("prompt-enhance-14b-think", "some-other-served-tag"):
        models = await service.get_available_models()
    ids = [m.id for m in models]
    # Only the allowlisted default survives — no cloud, no unserved local tag.
    assert ids == ["ollama_chat/prompt-enhance-14b-think"]


@pytest.mark.asyncio
async def test_models_endpoint_includes_explicitly_allowed() -> None:
    """A tag added to allowed_models is advertised alongside the default."""
    settings = Settings(
        default_model="ollama_chat/base",
        allowed_models=["ollama_chat/koleslaw-nemotron"],
    )
    service = LLMService(settings)
    with _mock_ollama_tags("base", "koleslaw-nemotron", "not-allowed"):
        models = await service.get_available_models()
    ids = {m.id for m in models}
    assert ids == {"ollama_chat/base", "ollama_chat/koleslaw-nemotron"}


# ── Unit: the resolver's edge cases ──────────────────────────────────────


def test_resolver_none_returns_default(settings: Settings) -> None:
    assert resolve_allowed_model(None, settings) == settings.default_model


def test_resolver_blank_returns_default(settings: Settings) -> None:
    assert resolve_allowed_model("", settings) == settings.default_model
    assert resolve_allowed_model("   ", settings) == settings.default_model


def test_resolver_default_model_allowed(settings: Settings) -> None:
    assert resolve_allowed_model(settings.default_model, settings) == settings.default_model


def test_resolver_trims_whitespace(settings: Settings) -> None:
    assert resolve_allowed_model(f"  {settings.default_model}  ", settings) == settings.default_model


def test_resolver_allowed_models_extends(settings: Settings) -> None:
    extended = settings.model_copy(update={"allowed_models": ["ollama_chat/koleslaw-nemotron"]})
    assert resolve_allowed_model("ollama_chat/koleslaw-nemotron", extended) == "ollama_chat/koleslaw-nemotron"
    # ...but the default is still allowed alongside the extras
    assert resolve_allowed_model(extended.default_model, extended) == extended.default_model


@pytest.mark.parametrize("model", CLOUD_MODELS + [BOGUS_LOCAL_MODEL])
def test_resolver_rejects_non_allowlisted(settings: Settings, model: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_allowed_model(model, settings)
    assert exc_info.value.status_code == 400


def test_resolver_echoes_truncated_model_name(settings: Settings) -> None:
    """A hostile over-long model string must not be echoed back in full."""
    huge = "openai/" + "x" * 500
    with pytest.raises(HTTPException) as exc_info:
        resolve_allowed_model(huge, settings)
    assert len(exc_info.value.detail) < 200


def test_resolver_cloud_not_selectable_even_if_configured() -> None:
    """A configured fallback/cloud model is NOT caller-selectable unless it is
    explicitly in allowed_models."""
    s = Settings(
        default_model="ollama_chat/base",
        fallback_model="anthropic/claude-haiku-4-5",
        anthropic_api_key=SecretStr("sk-test"),
    )
    with pytest.raises(HTTPException):
        resolve_allowed_model("anthropic/claude-haiku-4-5", s)
