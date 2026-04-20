import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


async def _fake_stream_with_thinking():
    """Simulates a model that emits reasoning_content then content."""
    thinking_chunks = [
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="", reasoning_content="Let me think"),
                    finish_reason=None,
                )
            ]
        ),
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="", reasoning_content=" about this..."),
                    finish_reason=None,
                )
            ]
        ),
    ]
    content_chunks = [
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="Enhanced ", reasoning_content=None),
                    finish_reason=None,
                )
            ]
        ),
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="prompt", reasoning_content=None),
                    finish_reason=None,
                )
            ]
        ),
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="", reasoning_content=None),
                    finish_reason="stop",
                )
            ]
        ),
    ]
    for chunk in thinking_chunks + content_chunks:
        yield chunk


async def _fake_stream_no_thinking():
    """Simulates a model that has no reasoning_content attribute."""
    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "), finish_reason=None)]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="world"), finish_reason=None)]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=""), finish_reason="stop")]),
    ]
    # Remove reasoning_content attribute to simulate models that don't have it
    for chunk in chunks:
        del chunk.choices[0].delta.reasoning_content
    for chunk in chunks:
        yield chunk


def _parse_sse_chunks(text: str) -> list[dict]:
    """Parse SSE response text into a list of JSON chunk dicts."""
    chunks = []
    for line in text.strip().split("\n\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        chunks.append(json.loads(data))
    return chunks


@pytest.mark.asyncio
async def test_chat_stream_with_thinking_chunks(client: AsyncClient) -> None:
    """Thinking chunks should have type='thinking', content chunks type='text'."""
    with patch("prompt_enhance.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream_with_thinking()
        response = await client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    chunks = _parse_sse_chunks(response.text)

    thinking_chunks = [c for c in chunks if c.get("type") == "thinking"]
    text_chunks = [c for c in chunks if c.get("type", "text") == "text"]

    assert len(thinking_chunks) == 2
    assert thinking_chunks[0]["delta"] == "Let me think"
    assert thinking_chunks[1]["delta"] == " about this..."

    # Text chunks include content deltas and the finish_reason chunk
    assert any(c["delta"] == "Enhanced " for c in text_chunks)
    assert any(c["delta"] == "prompt" for c in text_chunks)
    assert any(c.get("finish_reason") == "stop" for c in text_chunks)


@pytest.mark.asyncio
async def test_chat_stream_without_thinking(client: AsyncClient) -> None:
    """Models without reasoning_content should emit only type='text' chunks."""
    with patch("prompt_enhance.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream_no_thinking()
        response = await client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    chunks = _parse_sse_chunks(response.text)

    # No thinking chunks should be present
    thinking_chunks = [c for c in chunks if c.get("type") == "thinking"]
    assert len(thinking_chunks) == 0

    # All chunks should be text type (default)
    assert all(c.get("type", "text") == "text" for c in chunks)
    assert any(c["delta"] == "Hello " for c in chunks)
    assert any(c["delta"] == "world" for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_thinking_not_in_collected(client: AsyncClient) -> None:
    """Reasoning content should not be included in the collected output used for usage logging."""
    # We verify via SSE output: text-type deltas should only contain content, not reasoning.
    with patch("prompt_enhance.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _fake_stream_with_thinking()
        response = await client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    chunks = _parse_sse_chunks(response.text)

    # The text content chunks should only contain "Enhanced " and "prompt"
    text_deltas = [c["delta"] for c in chunks if c.get("type", "text") == "text" and c["delta"]]
    assert text_deltas == ["Enhanced ", "prompt"]

    # Thinking content should NOT appear in text chunks
    thinking_deltas = [c["delta"] for c in chunks if c.get("type") == "thinking"]
    for td in thinking_deltas:
        assert td not in text_deltas
