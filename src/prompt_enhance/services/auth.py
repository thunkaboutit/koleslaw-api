import hashlib
import logging
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy import select

from prompt_enhance.config import Settings, get_settings
from prompt_enhance.db.models import ApiKey

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    return f"pk_live_{secrets.token_urlsafe(32)}"


async def check_rate_limit(
    key_hash: str,
    redis: Redis,
    max_requests: int = 60,
    window_seconds: int = 60,
) -> tuple[bool, int]:
    import time

    now = time.time()
    window_key = f"rate_limit:{key_hash}"

    pipe = redis.pipeline()
    pipe.zremrangebyscore(window_key, 0, now - window_seconds)
    pipe.zadd(window_key, {f"{now}:{secrets.token_hex(4)}": now})
    pipe.zcard(window_key)
    pipe.expire(window_key, window_seconds)
    results = await pipe.execute()

    current_count = results[2]
    remaining = max(0, max_requests - current_count)
    is_allowed = current_count <= max_requests

    return is_allowed, remaining


async def get_current_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> str | None:
    redis: Redis = request.app.state.redis

    if credentials is None:
        if settings.environment == "development":
            logger.warning("Auth bypass active — no API key provided in development mode")
            return None
        raise HTTPException(status_code=401, detail="Missing API key")

    key = credentials.credentials
    key_hash = hash_api_key(key)

    # Validate key against the database and check is_active
    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
        api_key_row = result.scalar_one_or_none()

    if api_key_row is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not api_key_row.is_active:
        raise HTTPException(status_code=401, detail="API key has been revoked")

    # Use the per-key rate limit from the database
    is_allowed, remaining = await check_rate_limit(
        key_hash,
        redis,
        max_requests=api_key_row.rate_limit,
    )

    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )

    request.state.rate_limit_remaining = remaining
    return key_hash
