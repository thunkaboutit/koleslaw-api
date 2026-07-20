import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from koleslaw_api.config import Settings, get_settings
from koleslaw_api.db.engine import async_session_factory, create_engine
from koleslaw_api.routes import account, auth, billing, chat, enhance, health, keys, models, usage
from koleslaw_api.schemas.enhance import ErrorResponse
from koleslaw_api.services.cache import PromptCache
from koleslaw_api.services.llm import LLMService, LLMServiceError

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s:     %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Hostnames that indicate a local/dev database, so the dev auth bypass is safe.
_LOCAL_DB_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def _redact_dsn(url: str) -> str:
    """Mask the password in a connection string before logging it (T3-H)."""
    try:
        parts = urlparse(url)
        if not parts.password:
            return url
        netloc = f"{parts.username}:***@{parts.hostname or ''}"
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunparse(parts._replace(netloc=netloc))
    except Exception:
        return "<redacted>"


def _assert_safe_environment(settings: Settings) -> None:
    """Refuse to start the dev auth-bypass against a production-shaped database.

    ENVIRONMENT defaults to 'development', which disables API-key auth. If that
    default is ever shipped against a real (cloud) database the API would be
    wide open, so fail fast instead of serving unauthenticated (T3-K).
    """
    if settings.environment != "development":
        return
    host = (urlparse(settings.database_url).hostname or "").lower()
    is_local = host in _LOCAL_DB_HOSTS or "." not in host
    if not is_local:
        raise RuntimeError(
            f"Refusing to start: ENVIRONMENT=development (API-key auth is bypassed) but the database "
            f"host {host!r} is not local. Set ENVIRONMENT=production for non-local databases."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    _assert_safe_environment(settings)

    logger.info("Starting Prompt Enhance API")
    logger.info("  environment    = %s", settings.environment)
    logger.info("  database_url   = %s", _redact_dsn(settings.database_url))
    logger.info("  redis_url      = %s", _redact_dsn(settings.redis_url))
    logger.info("  ollama_base    = %s", settings.ollama_base_url)
    logger.info("  default_model  = %s", settings.default_model)
    logger.info("  frontend_url   = %s", settings.frontend_url)
    logger.info("  cors_origins   = %s", settings.cors_origins)
    logger.info("  github_oauth   = %s", "configured" if settings.github_client_id else "not configured")
    logger.info("  google_oauth   = %s", "configured" if settings.google_client_id else "not configured")
    logger.info("  facebook_oauth = %s", "configured" if settings.facebook_client_id else "not configured")
    logger.info("  instagram_oauth= %s", "configured" if settings.instagram_client_id else "not configured")

    engine = create_engine(settings.database_url)
    app.state.db_engine = engine
    app.state.db_session_factory = async_session_factory(engine)

    redis = Redis.from_url(settings.redis_url)
    app.state.redis = redis

    app.state.llm_service = LLMService(settings)
    app.state.cache = PromptCache(redis)

    yield

    await redis.aclose()
    await engine.dispose()


app = FastAPI(title="Prompt Enhance API", version="0.1.0", lifespan=lifespan)


def configure_cors(app: FastAPI, settings: Settings) -> None:
    """Apply the explicit-allowlist CORS policy (T1-C).

    An allowlist (never a wildcard) paired with credentials, so only known
    origins can make credentialed cross-origin requests. The portal is
    same-origin and needs no CORS; browser extensions declare host permissions
    and bypass CORS, so this list is the small set of cross-origin browser
    clients we actually trust.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


configure_cors(app, get_settings())

app.include_router(health.router)
app.include_router(models.router)
app.include_router(enhance.router)
app.include_router(chat.router)
app.include_router(keys.router)
app.include_router(usage.router)
app.include_router(auth.router)
app.include_router(account.router)
app.include_router(billing.router)


def _cors_headers(request: Request) -> dict[str, str]:
    # Error responses bypass the CORS middleware, so mirror its allowlist here —
    # only reflect an origin we actually trust (T1-C).
    origin = request.headers.get("origin")
    if origin and origin in request.app.state.settings.cors_origins:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
        }
    return {}


@app.exception_handler(LLMServiceError)
async def llm_error_handler(request: Request, exc: LLMServiceError) -> JSONResponse:
    status_map = {
        "authentication_error": 502,
        "rate_limit_error": 502,
        "bad_request": 400,
        "model_not_found": 404,
        "connection_error": 502,
        "service_unavailable": 503,
    }
    status_code = status_map.get(exc.code, 500)
    body = ErrorResponse(error=str(exc), code=exc.code, detail=exc.detail)
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=_cors_headers(request))


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Full type + traceback go to the logs; the client body stays generic so an
    # unhandled error never discloses internal exception shapes (T3-I).
    logger.exception("Unhandled exception")
    body = ErrorResponse(
        error="Internal server error",
        code="internal_error",
        detail=None,
    )
    return JSONResponse(status_code=500, content=body.model_dump(), headers=_cors_headers(request))
