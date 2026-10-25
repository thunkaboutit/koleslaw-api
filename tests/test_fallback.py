"""Availability fallback: when the local Ollama tier is unreachable (a GPU
instance replacement is a ~4 min serving gap), requests degrade to the
configured cloud model instead of erroring. Cloud-primary failures and
non-availability errors must NOT fall back."""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from httpx import AsyncClient
from pydantic import SecretStr

from koleslaw_api.config import Settings
from koleslaw_api.schemas.enhance import EnhanceRequest
from koleslaw_api.services.llm import LLMService, LLMServiceError

FALLBACK = "anthropic/claude-haiku-4-5"
LOCAL = "ollama_chat/prompt-enhance-14b-think"


@pytest.fixture
def settings() -> Settings:
    # Overrides the conftest fixture for this module: fallback enabled.
    return Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model=LOCAL,
        fallback_model=FALLBACK,
    )


def _fake_completion():
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "Enhanced prompt result"
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    return resp


async def _fake_stream():
    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Enhanced "), finish_reason=None)]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="prompt"), finish_reason="stop")]),
    ]
    for chunk in chunks:
        yield chunk


def _conn_error(model: str = LOCAL) -> litellm.APIConnectionError:
    return litellm.APIConnectionError(message="connection refused", llm_provider="ollama", model=model)


@pytest.mark.asyncio
async def test_sync_falls_back_on_connection_error(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [_conn_error(), _fake_completion()]
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert response.json()["model_used"] == FALLBACK
    assert mock_llm.call_count == 2
    # The retry must be rebuilt for the cloud model: our per-type system
    # prompt, not the Modelfile-implied bare user message.
    retry_kwargs = mock_llm.call_args_list[1].kwargs
    assert retry_kwargs["model"] == FALLBACK
    assert [m["role"] for m in retry_kwargs["messages"]] == ["system", "user"]


@pytest.mark.asyncio
async def test_sync_falls_back_on_model_not_found(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            litellm.NotFoundError(message="model not found", llm_provider="ollama", model=LOCAL),
            _fake_completion(),
        ]
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert response.json()["model_used"] == FALLBACK


@pytest.mark.asyncio
async def test_sync_bad_request_does_not_fall_back(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = litellm.BadRequestError(message="bad request", llm_provider="ollama", model=LOCAL)
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 400
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_cloud_primary_does_not_fall_back() -> None:
    """A cloud *primary* does not fall back — fallback is only for ollama_chat/*
    primaries. Driven at the service level: post-T1-A a caller can't select a
    cloud model, so this only arises when default_model itself is a cloud model.
    (connection_error maps to HTTP 502.)"""
    settings = Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model="anthropic/claude-sonnet-4-5",
        fallback_model=FALLBACK,
    )
    service = LLMService(settings)
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = _conn_error(model="anthropic/claude-sonnet-4-5")
        with pytest.raises(LLMServiceError) as exc_info:
            await service.enhance(EnhanceRequest(prompt="Write a poem", enhancement_type="refine"))
    assert exc_info.value.code == "connection_error"
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_stream_falls_back_before_first_byte(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [_conn_error(), _fake_stream()]
        response = await client.post(
            "/v1/enhance",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert mock_llm.call_count == 2
    retry_kwargs = mock_llm.call_args_list[1].kwargs
    assert retry_kwargs["model"] == FALLBACK
    assert retry_kwargs["stream"] is True
    lines = response.text.strip().split("\n\n")
    assert any("Enhanced " in line for line in lines)
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_fallback_failure_surfaces_mapped_error() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model=LOCAL,
        fallback_model=FALLBACK,
    )
    service = LLMService(settings)
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [_conn_error(), _conn_error(model=FALLBACK)]
        with pytest.raises(LLMServiceError) as exc_info:
            await service.enhance(EnhanceRequest(prompt="Write a poem"))
    assert exc_info.value.code == "connection_error"
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_no_fallback_configured_surfaces_error() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model=LOCAL,
        fallback_model=None,
    )
    service = LLMService(settings)
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = _conn_error()
        with pytest.raises(LLMServiceError) as exc_info:
            await service.enhance(EnhanceRequest(prompt="Write a poem"))
    assert exc_info.value.code == "connection_error"
    assert mock_llm.call_count == 1


# ── Wrapped provider client errors ───────────────────────────────────
#
# LiteLLM reports an Ollama 400 as APIConnectionError, wrapping the provider's
# JSON body. APIConnectionError is also the availability signal, so without a
# guard a plain bad request gets retried on the metered cloud tier and fails
# there too, at cost. 404 must keep falling back: an instance that lost its
# model registration answers 404, and that IS an outage.

OLLAMA_400 = (
    "litellm.APIConnectionError: Ollama_chatException - "
    '{"error":"{\\"error\\":{\\"code\\":400,\\"message\\":\\"Multimodal data provided, '
    'but model does not support multimodal requests.\\",\\"type\\":\\"invalid_request_error\\"}}"}'
)


@pytest.mark.asyncio
async def test_wrapped_provider_400_does_not_fall_back(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = litellm.APIConnectionError(message=OLLAMA_400, llm_provider="ollama", model=LOCAL)
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    # One call: the cloud tier was never billed for a request that is simply wrong.
    assert mock_llm.call_count == 1
    assert response.status_code == 400
    assert response.json()["code"] == "bad_request"


@pytest.mark.asyncio
async def test_wrapped_provider_404_still_falls_back(client: AsyncClient) -> None:
    """A 404 means the instance lost its model registration, which is an outage."""
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            litellm.APIConnectionError(
                message='Ollama_chatException - {"error":{"code":404,"message":"model not found"}}',
                llm_provider="ollama",
                model=LOCAL,
            ),
            _fake_completion(),
        ]
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert response.json()["model_used"] == FALLBACK
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_error_body_carries_no_provider_text(client: AsyncClient) -> None:
    """The mapped error reaches the caller sanitised; detail stays in the log."""
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = _conn_error()
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    body = response.text
    assert "connection refused" not in body
    assert LOCAL not in body


# ── Retry budget on the local tier ───────────────────────────────────
#
# A reclaimed GPU fails every retry the same way, and each retry can burn a
# full timeout before the fallback answers (4 x 90s on the sync path).  With
# a fallback configured the local call gets no retries; the cloud call keeps
# its retries, and a local model with no fallback keeps them too.


@pytest.mark.asyncio
async def test_sync_local_primary_does_not_retry_when_fallback_exists(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [_conn_error(), _fake_completion()]
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert mock_llm.call_args_list[0].kwargs["num_retries"] == 0
    assert mock_llm.call_args_list[1].kwargs["num_retries"] == 3


@pytest.mark.asyncio
async def test_stream_local_primary_does_not_retry_when_fallback_exists(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [_conn_error(), _fake_stream()]
        response = await client.post(
            "/v1/enhance",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert mock_llm.call_args_list[0].kwargs["num_retries"] == 0
    assert mock_llm.call_args_list[1].kwargs["num_retries"] == 3


@pytest.mark.asyncio
async def test_local_primary_keeps_retries_without_fallback() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model=LOCAL,
        fallback_model=None,
    )
    service = LLMService(settings)
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        await service.enhance(EnhanceRequest(prompt="Write a poem"))
    assert mock_llm.call_args.kwargs["num_retries"] == 3


@pytest.mark.asyncio
async def test_cloud_primary_keeps_retries() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model=FALLBACK,
        fallback_model=FALLBACK,
    )
    service = LLMService(settings)
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        await service.enhance(EnhanceRequest(prompt="Write a poem"))
    assert mock_llm.call_args.kwargs["num_retries"] == 3
