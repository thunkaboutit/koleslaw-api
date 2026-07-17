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
async def test_sync_cloud_primary_does_not_fall_back(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = _conn_error(model="anthropic/claude-sonnet-4-5")
        response = await client.post(
            "/v1/enhance/sync",
            json={
                "prompt": "Write a poem",
                "enhancement_type": "refine",
                "model": "anthropic/claude-sonnet-4-5",
            },
        )
    assert response.status_code == 502
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
