from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_ok(client: AsyncClient) -> None:
    response = await client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["database"] is True
    assert data["checks"]["redis"] is True
    assert "version" in data


@pytest.mark.asyncio
async def test_health_degraded_redis(client: AsyncClient, mock_redis: AsyncMock) -> None:
    mock_redis.ping.side_effect = ConnectionError("Redis down")
    response = await client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["redis"] is False
