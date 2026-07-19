"""Per-key daily caps by plan (free/pro) in get_llm_api_key."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from httpx import AsyncClient

from koleslaw_api.services.auth import get_current_api_key

# Any key whose hash is NOT in settings.anon_key_hashes
KEYED = {"Authorization": "Bearer pk_live_some-users-key", "Content-Type": "application/json"}
BODY = {"prompt": "Write a poem"}


def _plan(app, plan: str) -> None:
    """Simulate get_current_api_key resolving a key whose owner has `plan`."""

    def fake_key(request: Request) -> uuid.UUID:
        request.state.api_key_plan = plan
        return uuid.uuid4()

    app.dependency_overrides[get_current_api_key] = fake_key


def _arm_pipeline(mock_redis: AsyncMock, count: int) -> MagicMock:
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[count, True])
    mock_redis.pipeline = MagicMock(return_value=pipe)
    return pipe


@pytest.mark.asyncio
async def test_free_under_limit_passes(app, client: AsyncClient, mock_redis: AsyncMock) -> None:
    _plan(app, "free")
    _arm_pipeline(mock_redis, count=50)  # count == limit is still allowed
    mock_redis.get = AsyncMock(return_value="cached result")
    response = await client.post("/v1/enhance/sync", json=BODY, headers=KEYED)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_free_over_limit_gets_upgrade_429(app, client: AsyncClient, mock_redis: AsyncMock) -> None:
    _plan(app, "free")
    _arm_pipeline(mock_redis, count=51)
    response = await client.post("/v1/enhance/sync", json=BODY, headers=KEYED)
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert "free plan" in detail
    assert "pricing" in detail
    assert "Retry-After" in response.headers


@pytest.mark.asyncio
async def test_pro_passes_where_free_would_be_capped(app, client: AsyncClient, mock_redis: AsyncMock) -> None:
    _plan(app, "pro")
    _arm_pipeline(mock_redis, count=51)
    mock_redis.get = AsyncMock(return_value="cached result")
    response = await client.post("/v1/enhance/sync", json=BODY, headers=KEYED)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_pro_over_limit_gets_contact_429(app, client: AsyncClient, mock_redis: AsyncMock) -> None:
    _plan(app, "pro")
    _arm_pipeline(mock_redis, count=501)
    response = await client.post("/v1/enhance/sync", json=BODY, headers=KEYED)
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert "pro plan" in detail
    assert "contact" in detail


@pytest.mark.asyncio
async def test_anonymous_plan_excluded_from_keyed_cap(app, client: AsyncClient, mock_redis: AsyncMock) -> None:
    """A key owned by the 'anonymous' user (plan mismatch vs hash list) is never key-capped."""
    _plan(app, "anonymous")
    _arm_pipeline(mock_redis, count=999)
    mock_redis.get = AsyncMock(return_value="cached result")
    response = await client.post("/v1/enhance/sync", json=BODY, headers=KEYED)
    assert response.status_code == 200
    mock_redis.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_plan_treated_as_free(app, client: AsyncClient, mock_redis: AsyncMock) -> None:
    _plan(app, "enterprise-legacy")
    _arm_pipeline(mock_redis, count=51)
    response = await client.post("/v1/enhance/sync", json=BODY, headers=KEYED)
    assert response.status_code == 429
    assert "free plan" in response.json()["detail"]
