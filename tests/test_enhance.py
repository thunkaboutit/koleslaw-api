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
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["enhanced_prompt"] == "Enhanced prompt result"
    assert data["model_used"] == "ollama_chat/prompt-enhance-14b-think"
    assert data["usage"]["input_tokens"] == 10
    assert data["usage"]["output_tokens"] == 20


@pytest.mark.asyncio
async def test_enhance_stream_returns_sse(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
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
async def test_enhance_sync_auto_type(client: AsyncClient) -> None:
    """When enhancement_type is auto (the default), the request should succeed."""
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "write a function"},
        )
    assert response.status_code == 200
    assert response.json()["enhanced_prompt"] == "Enhanced prompt result"


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


@pytest.mark.asyncio
async def test_enhance_ollama_model_gets_no_system_prompt(client: AsyncClient) -> None:
    """Ollama models carry their SYSTEM prompt in the Modelfile — the API
    must not send one (a request-level system message would override it)."""
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    messages = mock_llm.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == "Write a poem"


@pytest.mark.asyncio
async def test_enhance_ollama_model_context_folds_into_user_message(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "context": "for a wedding toast"},
        )
    assert response.status_code == 200
    messages = mock_llm.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"].startswith("Write a poem")
    assert "for a wedding toast" in messages[0]["content"]


@pytest.mark.asyncio
async def test_enhance_cloud_model_keeps_system_prompt(client: AsyncClient) -> None:
    """Cloud fallback models have no Modelfile — they still need ours."""
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_completion()
        response = await client.post(
            "/v1/enhance/sync",
            json={
                "prompt": "Write a poem",
                "enhancement_type": "refine",
                "model": "anthropic/claude-haiku-4-5",
            },
        )
    assert response.status_code == 200
    messages = mock_llm.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]


@pytest.mark.asyncio
async def test_enhance_stream_emits_keepalive_during_stall(client: AsyncClient, monkeypatch) -> None:
    """A silent thinking phase must not leave the SSE connection byteless —
    Cloudflare kills proxied connections after 100s of silence."""
    import asyncio as aio

    from koleslaw_api.routes import enhance as enhance_route

    monkeypatch.setattr(enhance_route, "KEEPALIVE_SECONDS", 0.05)

    async def _slow_stream():
        await aio.sleep(0.2)  # stall longer than the keepalive window
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Enhanced"), finish_reason=None)])
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=""), finish_reason="stop")])

    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _slow_stream()
        response = await client.post(
            "/v1/enhance",
            json={"prompt": "Write a poem", "enhancement_type": "refine"},
        )
    assert response.status_code == 200
    assert response.text.startswith(": connected")
    assert ": keepalive" in response.text
    lines = response.text.strip().split("\n\n")
    assert any("Enhanced" in line for line in lines)
    assert lines[-1] == "data: [DONE]"
