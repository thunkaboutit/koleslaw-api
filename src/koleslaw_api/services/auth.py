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
from koleslaw_api.db.models import ApiKey, User
from koleslaw_api.services.session_auth import SESSION_COOKIE, decode_session_token

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

# The signed-in portal playground names one of the user's own keys by id
# (the plaintext key is shown once at creation and never stored client-side).
# The id alone grants nothing: the request must also carry a session cookie
# for the key's owner.
SELECTED_KEY_HEADER = "x-api-key-id"


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


async def check_daily_cap(kind: str, ident: str, redis: Redis, limit: int) -> tuple[bool, int]:
    """Increment-and-check a UTC-day-scoped counter (`{kind}:{date}:{ident}`)."""
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    counter_key = f"{kind}:{day}:{ident}"

    pipe = redis.pipeline()
    pipe.incr(counter_key)
    # The key is date-scoped, so the TTL only garbage-collects; 25h outlives
    # every counter without risking a premature reset.
    pipe.expire(counter_key, 90_000)
    results = await pipe.execute()

    count = results[0]
    return count <= limit, max(0, limit - count)


async def _get_session_selected_key(
    request: Request,
    raw_key_id: str,
    settings: Settings,
    redis: Redis,
) -> uuid.UUID:
    """Authenticate a portal request acting as one of the session user's keys.

    This path is cookie-authenticated, so CSRF matters: browsers preflight any
    cross-origin request carrying the custom header, and the CORS allowlist
    (T1-C) rejects unknown origins, so a hostile page never reaches this code.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = decode_session_token(token, settings)

    try:
        key_id = uuid.UUID(raw_key_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid API key selection") from exc

    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(
            select(ApiKey, User.plan)
            .join(User, ApiKey.user_id == User.id)
            .where(ApiKey.id == key_id, ApiKey.user_id == user_id)
        )
        row = result.one_or_none()

    if row is None:
        raise HTTPException(status_code=401, detail="API key not found")

    api_key_row, plan = row

    if not api_key_row.is_active:
        raise HTTPException(status_code=401, detail="API key has been revoked")

    is_allowed, remaining = await check_rate_limit(
        api_key_row.key_hash,
        redis,
        max_requests=api_key_row.rate_limit,
    )

    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )

    request.state.api_key_plan = plan
    request.state.selected_key_hash = api_key_row.key_hash
    request.state.rate_limit_remaining = remaining
    return api_key_row.id


async def get_current_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> uuid.UUID | None:
    redis: Redis = request.app.state.redis

    selected_key_id = request.headers.get(SELECTED_KEY_HEADER)
    if selected_key_id is not None:
        return await _get_session_selected_key(request, selected_key_id, settings, redis)

    if credentials is None:
        if settings.environment == "development":
            logger.warning("Auth bypass active — no API key provided in development mode")
            return None
        raise HTTPException(status_code=401, detail="Missing API key")

    key = credentials.credentials
    key_hash = hash_api_key(key)

    # Validate key against the database and check is_active; the owner's
    # plan rides along for the per-tier caps in get_llm_api_key
    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(
            select(ApiKey, User.plan).join(User, ApiKey.user_id == User.id).where(ApiKey.key_hash == key_hash)
        )
        row = result.one_or_none()

    if row is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    api_key_row, plan = row
    request.state.api_key_plan = plan

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


async def get_llm_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    api_key_id: uuid.UUID | None = Depends(get_current_api_key),  # noqa: B008
) -> uuid.UUID | None:
    """get_current_api_key plus per-tier daily caps, for LLM-invoking endpoints only.

    Anonymous (portal playground) keys get a per-client-IP cap; free/pro keys
    get a per-key cap sized by the owner's plan. A session-selected key (the
    signed-in playground, SELECTED_KEY_HEADER) is plan-capped exactly as if the
    key itself had been presented — never IP-capped. Cheap endpoints
    (/v1/models, key management) stay on plain auth so the portal keeps
    loading once a visitor's quota is spent.
    """
    redis: Redis = request.app.state.redis

    selected_key_hash: str | None = getattr(request.state, "selected_key_hash", None)
    if selected_key_hash is not None:
        await _check_plan_daily_cap(request, selected_key_hash, redis, settings)
        return api_key_id

    if credentials is None:
        return api_key_id

    key_hash = hash_api_key(credentials.credentials)

    if key_hash in settings.anon_key_hashes:
        if settings.anon_daily_limit <= 0:
            return api_key_id
        ip = get_client_ip(request)
        allowed, remaining = await check_daily_cap("anon_daily", ip, redis, settings.anon_daily_limit)
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

    # Keyed tiers: cap by the key owner's plan. 'anonymous' is the seeded
    # public user — governed by the IP cap above, never the keyed one.
    plan = getattr(request.state, "api_key_plan", "free")
    if plan == "anonymous":
        return api_key_id
    await _check_plan_daily_cap(request, key_hash, redis, settings)
    return api_key_id


async def _check_plan_daily_cap(request: Request, key_hash: str, redis: Redis, settings: Settings) -> None:
    """Per-key daily cap sized by the key owner's plan (free unless pro)."""
    plan = getattr(request.state, "api_key_plan", "free")
    limit = settings.pro_daily_limit if plan == "pro" else settings.free_daily_limit
    if limit <= 0:
        return

    allowed, remaining = await check_daily_cap("key_daily", key_hash, redis, limit)
    if not allowed:
        if plan == "pro":
            detail = (
                f"Daily limit reached ({limit} requests/day on the pro plan). "
                "Contact us at https://koleslaw.ai/contact to raise limits."
            )
        else:
            detail = (
                f"Daily limit reached ({limit} requests/day on the free plan). "
                "Upgrade at https://koleslaw.ai/pricing for higher limits."
            )
        raise HTTPException(
            status_code=429,
            detail=detail,
            headers={"Retry-After": str(_seconds_to_utc_midnight())},
        )

    request.state.daily_remaining = remaining
