"""Response shapes the shipped clients parse (PRD scenario 16).

Chrome and the auto-enhance hook call /v1/enhance/sync, the MCP plugin calls
the enhance_prompt tool, and the old UI bundle reads /v1/chat SSE.  Each
response body is pinned byte for byte in tests/fixtures/golden.  Regenerate
only on purpose: GOLDEN_UPDATE=1 uv run pytest tests/test_old_client_shapes.py
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from koleslaw_api.config import Settings

GOLDEN = Path(__file__).parent / "fixtures" / "golden"


def _assert_golden(name: str, actual: str) -> None:
    path = GOLDEN / name
    if os.environ.get("GOLDEN_UPDATE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual)
    assert actual == path.read_text()


def _completion(content: str) -> MagicMock:
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


@pytest.mark.asyncio
async def test_enhance_sync_shape_for_chrome_and_hook(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _completion("Enhanced prompt result")
        response = await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem", "enhancement_type": "refine", "stream": False},
        )
    assert response.status_code == 200
    _assert_golden("enhance_sync.json", response.text)


@pytest.mark.asyncio
async def test_enhance_stream_shape(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _stream(_chunk("Enhanced "), _chunk("prompt"), _chunk(finish="stop"))
        response = await client.post("/v1/enhance", json={"prompt": "Write a poem", "enhancement_type": "refine"})
    assert response.status_code == 200
    _assert_golden("enhance_stream.sse", response.text)


@pytest.mark.asyncio
async def test_chat_stream_shape_for_old_ui(client: AsyncClient) -> None:
    with patch("koleslaw_api.routes.chat.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _stream(
            _chunk(reasoning="Let me think"),
            _chunk(reasoning=" about this..."),
            _chunk("Enhanced "),
            _chunk("prompt"),
            _chunk(finish="stop"),
        )
        response = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "Hello"}]})
    assert response.status_code == 200
    _assert_golden("chat_stream.sse", response.text)


@pytest.mark.asyncio
async def test_mcp_enhance_prompt_shape(app: FastAPI, settings: Settings) -> None:
    from tests.test_mcp_endpoint import _call_tool, _harness

    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as mcp_client:
        with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _completion("Enhanced prompt result")
            result = await _call_tool(mcp_client, {"prompt": "Write a poem", "enhancement_type": "refine"})
    _assert_golden("mcp_enhance_prompt.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
