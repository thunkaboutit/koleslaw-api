"""Model reasoning never reaches answer text on any endpoint (PRD scenario 1)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from koleslaw_api.config import Settings


def _completion(content: str | None) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    return resp


def _chunk(content: str = "", reasoning: str | None = None, finish: str | None = None) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(delta=MagicMock(content=content, reasoning_content=reasoning), finish_reason=finish)]
    )


async def _stream(*chunks: MagicMock):
    for chunk in chunks:
        yield chunk


def _events(body: str) -> list[dict]:
    events = []
    for frame in body.strip().split("\n\n"):
        if frame.startswith("data: ") and frame != "data: [DONE]":
            events.append(json.loads(frame[6:]))
    return events


def _split(events: list[dict]) -> tuple[str, str]:
    thinking = "".join(e["delta"] for e in events if e.get("type") == "thinking")
    text = "".join(e["delta"] for e in events if e.get("type", "text") == "text")
    return thinking, text


# --- /v1/chat (old UI) ---------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_splits_inline_think_across_chunks(client: AsyncClient) -> None:
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _stream(
            _chunk("<think>"),
            _chunk("Need more info"),
            _chunk("</think>"),
            _chunk("Which framework?"),
            _chunk(finish="stop"),
        )
        response = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    events = _events(response.text)
    assert _split(events) == ("Need more info", "Which framework?")
    assert events[-1] == {"delta": "", "finish_reason": "stop", "type": "text"}


@pytest.mark.asyncio
async def test_chat_stream_close_only_tag_is_not_in_text(client: AsyncClient) -> None:
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _stream(_chunk("vague prompt, ask</think>Which framework?"), _chunk(finish="stop"))
        response = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert _split(_events(response.text)) == ("vague prompt, ask", "Which framework?")


@pytest.mark.asyncio
async def test_chat_stream_finish_rides_on_the_last_text_delta(client: AsyncClient) -> None:
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _stream(_chunk("<think>a</think>b", finish="stop"))
        response = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert _events(response.text) == [
        {"delta": "a", "finish_reason": None, "type": "thinking"},
        {"delta": "b", "finish_reason": "stop", "type": "text"},
    ]


@pytest.mark.asyncio
async def test_chat_sync_drops_inline_reasoning(client: AsyncClient) -> None:
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _completion("<think>plan</think>\n\nAnswer")
        response = await client.post("/v1/chat/sync", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response.json()["message"]["content"] == "Answer"


# --- /v1/enhance and /v1/enhance/sync (Chrome, hook, IDEs) ------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["<think>plan</think>\n\nEnhanced prompt", "plan</think>Enhanced prompt"],
)
async def test_enhance_sync_drops_reasoning(client: AsyncClient, mock_redis: AsyncMock, content: str) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _completion(content)
        response = await client.post("/v1/enhance/sync", json={"prompt": "Write a poem", "enhancement_type": "refine"})

    assert response.json()["enhanced_prompt"] == "Enhanced prompt"
    cached_value = mock_redis.set.call_args.args[1]
    assert cached_value == "Enhanced prompt"


@pytest.mark.asyncio
async def test_enhance_stream_drops_reasoning(client: AsyncClient, mock_redis: AsyncMock) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _stream(
            _chunk(reasoning="native "),
            _chunk("<think>inline"),
            _chunk("</think>Enhanced "),
            _chunk("prompt"),
            _chunk(finish="stop"),
        )
        response = await client.post("/v1/enhance", json={"prompt": "Write a poem", "enhancement_type": "refine"})

    events = _events(response.text)
    assert "".join(e["delta"] for e in events) == "Enhanced prompt"
    assert all("type" not in e for e in events)
    assert events[-1] == {"delta": "", "finish_reason": "stop"}
    assert mock_redis.set.call_args.args[1] == "Enhanced prompt"


# --- MCP enhance_prompt ----------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_enhance_prompt_drops_reasoning(app: FastAPI, settings: Settings) -> None:
    from tests.test_mcp_endpoint import _call_tool, _harness

    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as mcp_client:
        with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _completion("<think>plan</think>Enhanced prompt")
            result = await _call_tool(mcp_client, {"prompt": "Write a poem"})

    assert result["structuredContent"]["enhanced_prompt"] == "Enhanced prompt"


# --- Empty answers are never cached --------------------------------------------


@pytest.mark.asyncio
async def test_enhance_sync_never_caches_a_reasoning_only_answer(client: AsyncClient, mock_redis: AsyncMock) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _completion("<think>only reasoning</think>")
        response = await client.post("/v1/enhance/sync", json={"prompt": "Write a poem", "enhancement_type": "refine"})

    assert response.json()["enhanced_prompt"] == ""
    mock_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_enhance_stream_never_caches_a_reasoning_only_answer(client: AsyncClient, mock_redis: AsyncMock) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _stream(_chunk(reasoning="only reasoning"), _chunk(finish="stop"))
        await client.post("/v1/enhance", json={"prompt": "Write a poem", "enhancement_type": "refine"})

    mock_redis.set.assert_not_called()
