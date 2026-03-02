from unittest.mock import AsyncMock

import pytest

from prompt_enhance.services.cache import PromptCache


@pytest.fixture
def cache(mock_redis: AsyncMock) -> PromptCache:
    return PromptCache(mock_redis)


@pytest.fixture
def broken_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.ping = AsyncMock(side_effect=ConnectionError("down"))
    return redis


@pytest.mark.asyncio
async def test_cache_miss_returns_none(cache: PromptCache) -> None:
    result = await cache.get("hello", "refine", "model-a")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_then_get(cache: PromptCache, mock_redis: AsyncMock) -> None:
    await cache.set("hello", "refine", "model-a", "enhanced hello")
    mock_redis.set.assert_called_once()

    mock_redis.get.return_value = b"enhanced hello"
    result = await cache.get("hello", "refine", "model-a")
    assert result == "enhanced hello"


@pytest.mark.asyncio
async def test_cache_set_with_custom_ttl(cache: PromptCache, mock_redis: AsyncMock) -> None:
    await cache.set("hello", "refine", "model-a", "enhanced", ttl=3600)
    call_kwargs = mock_redis.set.call_args
    assert call_kwargs.kwargs["ex"] == 3600


@pytest.mark.asyncio
async def test_cache_invalidate(cache: PromptCache, mock_redis: AsyncMock) -> None:
    await cache.invalidate("hello", "refine", "model-a")
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_memory_on_redis_failure(broken_redis: AsyncMock) -> None:
    cache = PromptCache(broken_redis)

    result = await cache.get("hello", "refine", "model-a")
    assert result is None

    await cache.set("hello", "refine", "model-a", "cached value")
    result = await cache.get("hello", "refine", "model-a")
    assert result == "cached value"


@pytest.mark.asyncio
async def test_fallback_memory_lru_eviction(broken_redis: AsyncMock) -> None:
    cache = PromptCache(broken_redis)

    for i in range(1001):
        await cache.set(f"prompt-{i}", "refine", "model-a", f"response-{i}")

    result = await cache.get("prompt-0", "refine", "model-a")
    assert result is None

    result = await cache.get("prompt-1000", "refine", "model-a")
    assert result == "response-1000"


@pytest.mark.asyncio
async def test_fallback_invalidate(broken_redis: AsyncMock) -> None:
    cache = PromptCache(broken_redis)

    await cache.set("hello", "refine", "model-a", "cached")
    await cache.invalidate("hello", "refine", "model-a")
    result = await cache.get("hello", "refine", "model-a")
    assert result is None
