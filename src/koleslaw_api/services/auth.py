import hashlib
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy import select

from koleslaw_api.config import Settings, get_settings
from koleslaw_api.db.models import ApiKey

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


def get_client_ip(request: Request) -> str:
    """Best-effort client IP behind Cloudflare → ALB.

    Cloudflare sets CF-Connecting-IP to the true client. Without it, only the
    LAST X-Forwarded-For entry is trustworthy (the ALB appends its immediate
    peer; everything before that is client-supplied). Direct-to-ALB traffic
    that forges CF-Connecting-IP can dodge a per-IP cap, but the shared
    per-key sliding window still bounds aggregate cost.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _seconds_to_utc_midnight() -> int:
    now = datetime.now(UTC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))


async def check_anon_daily_cap(ip: str, redis: Redis, limit: int) -> tuple[bool, int]:
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    counter_key = f"anon_daily:{day}:{ip}"

    pipe = redis.pipeline()
    pipe.incr(counter_key)
    # The key is date-scoped, so the TTL only garbage-collects; 25h outlives
    # every counter without risking a premature reset.
    pipe.expire(counter_key, 90_000)
    results = await pipe.execute()

    count = results[0]
    return count <= limit, max(0, limit - count)


async def get_current_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> uuid.UUID | None:
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
    return api_key_row.id


async def get_current_api_key_anon_capped(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    api_key_id: uuid.UUID | None = Depends(get_current_api_key),  # noqa: B008
) -> uuid.UUID | None:
    """get_current_api_key plus a per-client-IP daily cap for anonymous keys.

    Apply to LLM-invoking endpoints only — cheap endpoints (/v1/models, key
    management) stay uncapped so the portal keeps loading once a visitor's
    quota is spent.
    """
    if credentials is None or settings.anon_daily_limit <= 0:
        return api_key_id
    if hash_api_key(credentials.credentials) not in settings.anon_key_hashes:
        return api_key_id

    redis: Redis = request.app.state.redis
    ip = get_client_ip(request)
    allowed, remaining = await check_anon_daily_cap(ip, redis, settings.anon_daily_limit)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily free limit reached ({settings.anon_daily_limit} requests/day per visitor). "
                "Sign in at https://koleslaw.ai to create a free API key with higher limits."
            ),
            headers={"Retry-After": str(_seconds_to_utc_midnight())},
        )

    request.state.anon_daily_remaining = remaining
    return api_key_id
