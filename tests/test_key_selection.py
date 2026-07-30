"""Session-selected API keys (X-Api-Key-Id): the signed-in playground acting with a user's own key."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from koleslaw_api.db.models import ApiKey, UsageLog
from koleslaw_api.services.auth import get_current_api_key, hash_api_key
from koleslaw_api.services.session_auth import SESSION_COOKIE, create_session_token

# The portal playground bearer rides along on every request (anonymous
# fallback); with the header present the API must ignore it entirely.
PUBLIC_KEY = "pk_live_fNuYW07Bf16zLFLbghxSPERYV4QS1SWF4-OuLJ5JcCE"
BODY = {"prompt": "Write a poem"}

USER_ID = uuid.uuid4()
KEY_ID = uuid.uuid4()
KEY_HASH = hash_api_key("pk_live_the-selected-key")


def _headers(key_id: str | None = None) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {PUBLIC_KEY}",
        "CF-Connecting-IP": "203.0.113.7",
        "X-Api-Key-Id": key_id if key_id is not None else str(KEY_ID),
    }


def _key_row(*, is_active: bool = True, rate_limit: int = 60) -> ApiKey:
    return ApiKey(
        id=KEY_ID,
        user_id=USER_ID,
        key_hash=KEY_HASH,
        name="laptop",
        key_prefix="pk_live_sel",
        is_active=is_active,
        rate_limit=rate_limit,
    )


def _arm_db(app, row: ApiKey | None, plan: str = "free") -> AsyncMock:
    """Point the app at a session whose key lookup resolves to (row, plan)."""
    result = MagicMock()
    result.one_or_none = MagicMock(return_value=None if row is None else (row, plan))
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def factory():
        yield session

    app.state.db_session_factory = factory
    return session


def _arm_pipes(mock_redis: AsyncMock, *, rate_count: int = 1, daily_count: int = 1) -> tuple[MagicMock, MagicMock]:
    """First pipeline call = sliding rate limit (zcard at [2]); second = daily cap (incr at [0])."""
    rate_pipe = MagicMock()
    rate_pipe.execute = AsyncMock(return_value=[0, 1, rate_count, True])
    daily_pipe = MagicMock()
    daily_pipe.execute = AsyncMock(return_value=[daily_count, True])
    mock_redis.pipeline = MagicMock(side_effect=[rate_pipe, daily_pipe])
    return rate_pipe, daily_pipe


@pytest.fixture
async def session_client(app, settings) -> AsyncIterator[AsyncClient]:
    """The conftest app with the REAL get_current_api_key, plus a valid session cookie."""
    del app.dependency_overrides[get_current_api_key]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ac.cookies.set(SESSION_COOKIE, create_session_token(USER_ID, settings))
        yield ac


@pytest.fixture
async def anonymous_client(app) -> AsyncIterator[AsyncClient]:
    """Same app, no session cookie."""
    del app.dependency_overrides[get_current_api_key]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_selected_key_works_when_anon_ip_cap_is_exhausted(
    app, session_client: AsyncClient, mock_redis: AsyncMock
) -> None:
    """The headline regression: 11 > the anon 10/day, but the selected key is plan-capped, not IP-capped."""
    _arm_db(app, _key_row())
    _, daily_pipe = _arm_pipes(mock_redis, daily_count=11)
    mock_redis.get = AsyncMock(return_value="cached result")

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 200
    (counter_key,) = daily_pipe.incr.call_args.args
    assert counter_key.startswith("key_daily:")
    assert counter_key.endswith(f":{KEY_HASH}")
    assert "203.0.113.7" not in counter_key


@pytest.mark.asyncio
async def test_usage_is_logged_against_the_selected_key(
    app, session_client: AsyncClient, mock_redis: AsyncMock
) -> None:
    session = _arm_db(app, _key_row())
    _arm_pipes(mock_redis)
    mock_redis.get = AsyncMock(return_value="cached result")

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 200
    logged = [a.args[0] for a in session.add.call_args_list if isinstance(a.args[0], UsageLog)]
    assert len(logged) == 1
    assert logged[0].api_key_id == KEY_ID


@pytest.mark.asyncio
async def test_selected_key_over_free_cap_gets_upgrade_429(
    app, session_client: AsyncClient, mock_redis: AsyncMock
) -> None:
    _arm_db(app, _key_row())
    _arm_pipes(mock_redis, daily_count=51)

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 429
    assert "free plan" in response.json()["detail"]
    assert "Retry-After" in response.headers


@pytest.mark.asyncio
async def test_selected_key_pro_plan_gets_pro_limit(app, session_client: AsyncClient, mock_redis: AsyncMock) -> None:
    _arm_db(app, _key_row(), plan="pro")
    _arm_pipes(mock_redis, daily_count=51)
    mock_redis.get = AsyncMock(return_value="cached result")

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_selected_key_sliding_rate_limit_applies(app, session_client: AsyncClient, mock_redis: AsyncMock) -> None:
    _arm_db(app, _key_row(rate_limit=60))
    _arm_pipes(mock_redis, rate_count=61)

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded"


@pytest.mark.asyncio
async def test_header_without_session_cookie_is_401(app, anonymous_client: AsyncClient, mock_redis: AsyncMock) -> None:
    _arm_db(app, _key_row())
    _arm_pipes(mock_redis)

    response = await anonymous_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


@pytest.mark.asyncio
async def test_key_not_owned_by_session_user_is_401(app, session_client: AsyncClient, mock_redis: AsyncMock) -> None:
    """The ownership filter is in the query, so someone else's key id resolves to no row."""
    _arm_db(app, None)
    _arm_pipes(mock_redis)

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 401
    assert response.json()["detail"] == "API key not found"


@pytest.mark.asyncio
async def test_revoked_selected_key_is_401(app, session_client: AsyncClient, mock_redis: AsyncMock) -> None:
    _arm_db(app, _key_row(is_active=False))
    _arm_pipes(mock_redis)

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 401
    assert response.json()["detail"] == "API key has been revoked"


@pytest.mark.asyncio
async def test_malformed_key_id_is_401(app, session_client: AsyncClient, mock_redis: AsyncMock) -> None:
    _arm_db(app, _key_row())
    _arm_pipes(mock_redis)

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers("not-a-uuid"))

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key selection"


@pytest.mark.asyncio
async def test_anon_daily_counter_is_never_touched_on_the_selected_path(
    app, session_client: AsyncClient, mock_redis: AsyncMock
) -> None:
    """The bearer (the public playground key) must be ignored when the header is present."""
    _arm_db(app, _key_row())
    _, daily_pipe = _arm_pipes(mock_redis)
    mock_redis.get = AsyncMock(return_value="cached result")

    response = await session_client.post("/v1/enhance/sync", json=BODY, headers=_headers())

    assert response.status_code == 200
    for call in daily_pipe.incr.call_args_list:
        assert not call.args[0].startswith("anon_daily:")
