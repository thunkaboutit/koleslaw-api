from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


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
        MagicMock(choices=[MagicMock(delta=MagicMock(content="prompt"), finish_reason=None)]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=""), finish_reason="stop")]),
    ]
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_enhance_sync_returns_response(client: AsyncClient) -> None:
    with patch("prompt_enhance.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["enhanced_prompt"] == "Enhanced prompt result"
    assert data["model_used"] == "claude-haiku-4-5-20250929"
    assert data["usage"]["input_tokens"] == 10
    assert data["usage"]["output_tokens"] == 20


@pytest.mark.asyncio
async def test_enhance_stream_returns_sse(client: AsyncClient) -> None:
    with patch("prompt_enhance.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream()
        response = await client.post(
            "/v1/enhance",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    lines = response.text.strip().split("\n\n")
    assert any("Enhanced " in line for line in lines)
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_enhance_sync_invalid_request(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "", "enhancement_type": "refine"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_enhance_sync_missing_prompt(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/enhance/sync",
        json={"enhancement_type": "refine"},
    )
    assert response.status_code == 422
