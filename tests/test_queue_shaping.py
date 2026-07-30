"""Queue shaping: every LLM call carries an explicit timeout so a request
queued behind a busy GPU spills to the cloud fallback instead of hanging
(LiteLLM's default request timeout is 6000s). Streams get the short
first-token ceiling; the sync path gets the longer whole-generation one.
A local-tier timeout is an availability failure and must fall back."""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from httpx import AsyncClient
from pydantic import SecretStr

from koleslaw_api.config import Settings
from koleslaw_api.schemas.enhance import EnhanceRequest
from koleslaw_api.services.llm import LLMService

FALLBACK = "anthropic/claude-haiku-4-5"
LOCAL = "ollama_chat/prompt-enhance-14b-think"


@pytest.fixture
def settings() -> Settings:
    # Overrides the conftest fixture for this module: fallback enabled,
    # distinctive timeout values so a swapped constant can't pass.
    return Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model=LOCAL,
        fallback_model=FALLBACK,
        llm_stream_timeout_seconds=31.5,
        llm_sync_timeout_seconds=91.5,
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


def _timeout_error(model: str = LOCAL) -> litellm.Timeout:
    return litellm.Timeout(message="Request timed out", llm_provider="ollama", model=model)


@pytest.mark.asyncio
async def test_sync_calls_carry_the_sync_timeout(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert mock_llm.call_args.kwargs["timeout"] == 91.5


@pytest.mark.asyncio
async def test_stream_calls_carry_the_stream_timeout(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream()
        response = await client.post(
            "/v1/enhance",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert mock_llm.call_args.kwargs["timeout"] == 31.5


@pytest.mark.asyncio
async def test_sync_timeout_spills_to_fallback_with_same_ceiling(client: AsyncClient) -> None:
    """A local-tier timeout is the designed spill: the retry goes to the
    cloud model and stays bounded by the same explicit ceiling."""
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [_timeout_error(), _fake_completion()]
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert response.json()["model_used"] == FALLBACK
    assert mock_llm.call_count == 2
    retry_kwargs = mock_llm.call_args_list[1].kwargs
    assert retry_kwargs["model"] == FALLBACK
    assert retry_kwargs["timeout"] == 91.5


@pytest.mark.asyncio
async def test_stream_timeout_spills_to_fallback_with_same_ceiling(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [_timeout_error(), _fake_stream()]
        response = await client.post(
            "/v1/enhance",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert mock_llm.call_count == 2
    retry_kwargs = mock_llm.call_args_list[1].kwargs
    assert retry_kwargs["model"] == FALLBACK
    assert retry_kwargs["stream"] is True
    assert retry_kwargs["timeout"] == 31.5
    lines = response.text.strip().split("\n\n")
    assert any("Enhanced " in line for line in lines)
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_queue_overflow_503_spills_to_fallback() -> None:
    """Ollama answers a full queue (OLLAMA_MAX_QUEUE) with an instant 503;
    LiteLLM surfaces it wrapped as APIConnectionError, which must qualify
    as an availability failure — the burst path of the queue-shaping design."""
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
    busy = litellm.APIConnectionError(
        message='Ollama_chatException - {"error": "server busy, please try again."}',
        llm_provider="ollama",
        model=LOCAL,
    )
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [busy, _fake_completion()]
        result = await service.enhance(EnhanceRequest(prompt="Write a poem", enhancement_type="refine"))
    assert result.model_used == FALLBACK
    assert mock_llm.call_count == 2
