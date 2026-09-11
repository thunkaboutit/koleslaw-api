"""Binary attachments on /v1/chat/upload.

An attached image becomes an image_url content block, which only a
vision-capable model accepts. The self-hosted default is a text-only
fine-tune, so before this the playground's attach button produced a raw
provider 400 for every image ("model does not support multimodal requests").
These cover the routing that replaces it, the anonymous refusal that keeps
unauthenticated traffic off metered inference, and the sanitised error text.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from fastapi import HTTPException, Request
from httpx import AsyncClient
from pydantic import SecretStr

from koleslaw_api.config import Settings
from koleslaw_api.services.auth import get_current_api_key
from koleslaw_api.services.llm import (
    model_supports_vision,
    resolve_attachment_model,
    user_facing_llm_error,
    wrapped_upstream_status,
)

LOCAL = "ollama_chat/prompt-enhance-14b-think"
FALLBACK = "anthropic/claude-haiku-4-5"

# Magic bytes are all `filetype` reads; the rest of the file is never decoded.
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PDF = b"%PDF-1.4\n" + b"\x00" * 64

AUTH = {"Authorization": "Bearer pk_live_some-users-key"}

# The exact text LiteLLM produced for the reported failure: an Ollama 400
# wrapped as a transport error, with the provider's JSON body inside it.
OLLAMA_400 = (
    "litellm.APIConnectionError: Ollama_chatException - "
    '{"error":"{\\"error\\":{\\"code\\":400,\\"message\\":\\"Multimodal data provided, '
    'but model does not support multimodal requests.\\",\\"type\\":\\"invalid_request_error\\"}}"}'
)


@pytest.fixture
def settings() -> Settings:
    # Overrides the conftest fixture for this module: fallback enabled, so the
    # vision tier exists to route to.
    return Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model=LOCAL,
        fallback_model=FALLBACK,
    )


@pytest.fixture(autouse=True)
def _under_daily_cap(mock_redis: AsyncMock) -> None:
    """Keep the per-plan daily cap out of the way; it has its own tests."""
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, True])
    mock_redis.pipeline = MagicMock(return_value=pipe)


def _plan(app, plan: str) -> None:
    """Simulate get_current_api_key resolving a key whose owner has `plan`."""

    def fake_key(request: Request) -> uuid.UUID:
        request.state.api_key_plan = plan
        return uuid.uuid4()

    app.dependency_overrides[get_current_api_key] = fake_key


async def _fake_stream():
    yield MagicMock(choices=[MagicMock(delta=MagicMock(content="ok", reasoning_content=None), finish_reason="stop")])


def _upload(files: list[tuple], prompt: str = "Enhance this prompt:\n\nmake the kitchen wider"):
    return {
        "data": {"messages": json.dumps([{"role": "user", "content": prompt}])},
        "files": files,
    }


def _last_user_content(mock_llm) -> list[dict]:
    return mock_llm.call_args.kwargs["messages"][-1]["content"]


# ── Model policy ─────────────────────────────────────────────────────


def test_local_finetune_is_not_vision_capable() -> None:
    """The premise of the bug: our default model cannot take an image."""
    assert model_supports_vision("ollama_chat/prompt-enhance-nemotron-q4:latest") is False
    assert model_supports_vision(LOCAL) is False


def test_cloud_fallback_is_vision_capable() -> None:
    assert model_supports_vision(FALLBACK) is True


def test_vision_capable_model_is_left_alone(settings: Settings) -> None:
    assert resolve_attachment_model(FALLBACK, settings, "free") == FALLBACK


def test_text_only_model_routes_to_the_fallback(settings: Settings) -> None:
    assert resolve_attachment_model(LOCAL, settings, "free") == FALLBACK


def test_anonymous_plan_is_refused(settings: Settings) -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_attachment_model(LOCAL, settings, "anonymous")
    assert exc_info.value.status_code == 403
    assert "Sign in" in exc_info.value.detail


def test_anonymous_plan_is_allowed_when_no_cloud_spend_is_involved(settings: Settings) -> None:
    """The refusal is about metered spend, not about images as such."""
    assert resolve_attachment_model(FALLBACK, settings, "anonymous") == FALLBACK


def test_no_fallback_configured_is_a_clear_422() -> None:
    no_fallback = Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        default_model=LOCAL,
        fallback_model=None,
    )
    with pytest.raises(HTTPException) as exc_info:
        resolve_attachment_model(LOCAL, no_fallback, "free")
    assert exc_info.value.status_code == 422


# ── Upload endpoint ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_image_attachment_routes_to_the_vision_model(app, client: AsyncClient) -> None:
    """The regression test for the reported failure."""
    _plan(app, "free")
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream()
        response = await client.post(
            "/v1/chat/upload",
            headers=AUTH,
            **_upload([("files", ("kitchen.jpeg", JPEG, "image/jpeg"))]),
        )

    assert response.status_code == 200
    assert mock_llm.call_args.kwargs["model"] == FALLBACK
    blocks = _last_user_content(mock_llm)
    assert [b["type"] for b in blocks] == ["text", "image_url"]
    assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_two_images_both_reach_the_model(app, client: AsyncClient) -> None:
    """cp's report attached two files; both must survive the routing."""
    _plan(app, "free")
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream()
        response = await client.post(
            "/v1/chat/upload",
            headers=AUTH,
            **_upload(
                [
                    ("files", ("kitchen-1.jpeg", JPEG, "image/jpeg")),
                    ("files", ("kitchen-2.png", PNG, "image/png")),
                ]
            ),
        )

    assert response.status_code == 200
    assert mock_llm.call_args.kwargs["model"] == FALLBACK
    assert [b["type"] for b in _last_user_content(mock_llm)] == ["text", "image_url", "image_url"]


@pytest.mark.asyncio
async def test_text_attachment_stays_on_the_local_model(app, client: AsyncClient) -> None:
    """No image, no cloud spend: a text file still runs on the self-hosted tier."""
    _plan(app, "free")
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream()
        response = await client.post(
            "/v1/chat/upload",
            headers=AUTH,
            **_upload([("files", ("notes.txt", b"some notes", "text/plain"))]),
        )

    assert response.status_code == 200
    assert mock_llm.call_args.kwargs["model"] == LOCAL
    assert [b["type"] for b in _last_user_content(mock_llm)] == ["text", "text"]


@pytest.mark.asyncio
async def test_anonymous_image_upload_is_refused_before_any_llm_call(app, client: AsyncClient) -> None:
    _plan(app, "anonymous")
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        response = await client.post(
            "/v1/chat/upload",
            headers=AUTH,
            **_upload([("files", ("kitchen.jpeg", JPEG, "image/jpeg"))]),
        )

    assert response.status_code == 403
    assert "Sign in" in response.json()["detail"]
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_anonymous_text_upload_still_works(app, client: AsyncClient) -> None:
    """The refusal is scoped to images; the anonymous playground keeps working."""
    _plan(app, "anonymous")
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream()
        response = await client.post(
            "/v1/chat/upload",
            headers=AUTH,
            **_upload([("files", ("notes.txt", b"some notes", "text/plain"))]),
        )

    assert response.status_code == 200
    assert mock_llm.call_args.kwargs["model"] == LOCAL


@pytest.mark.asyncio
async def test_pdf_is_rejected_with_a_readable_error(app, client: AsyncClient) -> None:
    """It was base64-encoded and shipped before, then failed at the provider."""
    _plan(app, "free")
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        response = await client.post(
            "/v1/chat/upload",
            headers=AUTH,
            **_upload([("files", ("spec.pdf", PDF, "application/pdf"))]),
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "spec.pdf" in detail
    assert "application/pdf" in detail
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_usage_is_logged_against_the_model_that_served(app, client: AsyncClient) -> None:
    _plan(app, "free")
    with (
        patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm,
        patch("koleslaw_api.routes.chat.log_usage", new_callable=AsyncMock) as mock_log,
    ):
        mock_llm.return_value = _fake_stream()
        response = await client.post(
            "/v1/chat/upload",
            headers=AUTH,
            **_upload([("files", ("kitchen.jpeg", JPEG, "image/jpeg"))]),
        )

    assert response.status_code == 200
    mock_log.assert_awaited_once()
    assert mock_log.await_args.args[2] == FALLBACK


# ── Error text ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_failure_does_not_leak_provider_text(app, client: AsyncClient) -> None:
    _plan(app, "free")
    boom = litellm.APIConnectionError(message=OLLAMA_400, llm_provider="ollama", model=LOCAL)
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = boom
        response = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}, headers=AUTH)

    body = response.text
    assert "temporarily unavailable" in body
    for leak in ("Ollama_chatException", "APIConnectionError", LOCAL, "invalid_request_error"):
        assert leak not in body


def test_user_facing_error_never_returns_provider_text() -> None:
    exc = litellm.APIConnectionError(message=OLLAMA_400, llm_provider="ollama", model=LOCAL)
    message = user_facing_llm_error(exc)
    assert "Ollama" not in message
    assert "400" not in message


# ── Wrapped provider status (cost leak) ──────────────────────────────


def test_wrapped_400_is_parsed_from_the_real_message() -> None:
    assert wrapped_upstream_status(Exception(OLLAMA_400)) == 400


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("connection refused", None),
        ("Error code: 429 - rate limited", 429),
        ('{"status_code": 503}', 503),
        ('{"code":404}', 404),
    ],
)
def test_wrapped_status_parsing(message: str, expected: int | None) -> None:
    assert wrapped_upstream_status(Exception(message)) == expected
