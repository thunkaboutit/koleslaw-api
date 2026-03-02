import hashlib
import logging
from collections import OrderedDict

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

MAX_MEMORY_ENTRIES = 1000


class PromptCache:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._memory: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def _cache_key(prompt: str, enhancement_type: str, model: str) -> str:
        normalized = prompt.strip().lower()
        raw = f"{model}:{enhancement_type}:{normalized}"
        return f"prompt_cache:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def _redis_available(self) -> bool:
        try:
            await self._redis.ping()
        except Exception:
            return False
        return True

    async def get(self, prompt: str, enhancement_type: str, model: str) -> str | None:
        key = self._cache_key(prompt, enhancement_type, model)

        if await self._redis_available():
            value = await self._redis.get(key)
            return value.decode() if isinstance(value, bytes) else value

        logger.warning("Redis unavailable, falling back to in-memory cache")
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        return None

    async def set(
        self,
        prompt: str,
        enhancement_type: str,
        model: str,
        response: str,
        ttl: int = 86400,
    ) -> None:
        key = self._cache_key(prompt, enhancement_type, model)

        if await self._redis_available():
            await self._redis.set(key, response, ex=ttl)
            return

        logger.warning("Redis unavailable, falling back to in-memory cache")
        if key in self._memory:
            self._memory.move_to_end(key)
        self._memory[key] = response
        while len(self._memory) > MAX_MEMORY_ENTRIES:
            self._memory.popitem(last=False)

    async def invalidate(self, prompt: str, enhancement_type: str, model: str) -> None:
        key = self._cache_key(prompt, enhancement_type, model)

        if await self._redis_available():
            await self._redis.delete(key)
            return

        logger.warning("Redis unavailable, falling back to in-memory cache")
        self._memory.pop(key, None)
