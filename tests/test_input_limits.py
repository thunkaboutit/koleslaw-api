"""Security T2-E — the `context` field must be length-bounded like `prompt`.

`prompt` / chat `content` are capped at 50 000 chars, but `context` was
unbounded and is appended into the model input — a cost/token-amplification
lever that sidesteps the prompt cap. See docs/security-review-2026-07-19.md.
"""

import pytest
from httpx import AsyncClient

CAP = 50_000


@pytest.mark.asyncio
async def test_enhance_rejects_oversized_context(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "hi", "context": "x" * (CAP + 1)},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_rejects_oversized_context(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/sync",
        json={"messages": [{"role": "user", "content": "hi"}], "context": "x" * (CAP + 1)},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_enhance_allows_context_at_cap(client: AsyncClient, mock_redis) -> None:
    """A context right at the cap is accepted (proven via a cache hit)."""
    mock_redis.get.return_value = "cached result"
    resp = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "hi", "context": "x" * CAP},
    )
    assert resp.status_code == 200
