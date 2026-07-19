"""Anonymous per-IP daily cap (get_current_api_key_anon_capped)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient
from starlette.requests import Request

from koleslaw_api.services.auth import get_client_ip

# The portal playground key baked into the web bundle. Its sha256 is the
# default entry in Settings.anon_key_hashes, so using the real string here
# also cross-checks that default against the kowpen seed.
PUBLIC_KEY = "pk_live_fNuYW07Bf16zLFLbghxSPERYV4QS1SWF4-OuLJ5JcCE"


def _request_with(headers: dict[str, str], client_host: str = "10.0.0.1") -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (client_host, 12345),
    }
    return Request(scope)


def test_get_client_ip_prefers_cloudflare_header() -> None:
    request = _request_with({"cf-connecting-ip": "203.0.113.7", "x-forwarded-for": "1.1.1.1, 2.2.2.2"})
    assert get_client_ip(request) == "203.0.113.7"


def test_get_client_ip_uses_last_xff_hop() -> None:
    """Only the ALB-appended (last) hop is trustworthy; earlier entries are client-supplied."""
    request = _request_with({"x-forwarded-for": "6.6.6.6, 198.51.100.9"})
    assert get_client_ip(request) == "198.51.100.9"


def test_get_client_ip_falls_back_to_peer() -> None:
    request = _request_with({}, client_host="127.0.0.1")
    assert get_client_ip(request) == "127.0.0.1"


def _arm_pipeline(mock_redis: AsyncMock, count: int) -> MagicMock:
    """Make redis.pipeline() a sync call whose execute() reports `count` for INCR."""
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[count, True])
    mock_redis.pipeline = MagicMock(return_value=pipe)
    return pipe


@pytest.mark.asyncio
async def test_anon_over_limit_gets_429(client: AsyncClient, mock_redis: AsyncMock) -> None:
    _arm_pipeline(mock_redis, count=11)
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem"},
        headers={"Authorization": f"Bearer {PUBLIC_KEY}", "CF-Connecting-IP": "203.0.113.7"},
    )
    assert response.status_code == 429
    assert "Daily free limit" in response.json()["detail"]
    assert "Retry-After" in response.headers


@pytest.mark.asyncio
async def test_anon_under_limit_passes(client: AsyncClient, mock_redis: AsyncMock) -> None:
    _arm_pipeline(mock_redis, count=3)
    mock_redis.get = AsyncMock(return_value="cached result")  # cache hit → no LLM call
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem"},
        headers={"Authorization": f"Bearer {PUBLIC_KEY}", "CF-Connecting-IP": "203.0.113.7"},
    )
    assert response.status_code == 200
    assert response.json()["cached"] is True


@pytest.mark.asyncio
async def test_counter_is_per_ip_and_per_day(client: AsyncClient, mock_redis: AsyncMock) -> None:
    pipe = _arm_pipeline(mock_redis, count=1)
    mock_redis.get = AsyncMock(return_value="cached result")
    for ip in ("203.0.113.7", "198.51.100.9"):
        await client.post(
            "/v1/enhance/sync",
            json={"prompt": "Write a poem"},
            headers={"Authorization": f"Bearer {PUBLIC_KEY}", "CF-Connecting-IP": ip},
        )
    keys = [call.args[0] for call in pipe.incr.call_args_list]
    assert len(keys) == 2
    assert keys[0] != keys[1]
    assert keys[0].startswith("anon_daily:")
    assert keys[0].endswith(":203.0.113.7")


@pytest.mark.asyncio
async def test_non_anon_key_uses_keyed_counter_not_ip(client: AsyncClient, mock_redis: AsyncMock) -> None:
    """Non-anonymous keys are governed by the per-key plan cap (see test_plan_caps), not the IP counter."""
    pipe = _arm_pipeline(mock_redis, count=1)
    mock_redis.get = AsyncMock(return_value="cached result")
    response = await client.post(
        "/v1/enhance/sync",
        json={"prompt": "Write a poem"},
        headers={"Authorization": "Bearer pk_live_some-other-key", "CF-Connecting-IP": "203.0.113.7"},
    )
    assert response.status_code == 200
    (counter_key,) = pipe.incr.call_args.args
    assert counter_key.startswith("key_daily:")
    assert "203.0.113.7" not in counter_key


@pytest.mark.asyncio
async def test_no_auth_header_skips_cap(client: AsyncClient, mock_redis: AsyncMock) -> None:
    """Development-mode auth bypass (no credentials) must not hit the counter."""
    _arm_pipeline(mock_redis, count=999)
    mock_redis.get = AsyncMock(return_value="cached result")
    response = await client.post("/v1/enhance/sync", json={"prompt": "Write a poem"})
    assert response.status_code == 200
    mock_redis.pipeline.assert_not_called()
