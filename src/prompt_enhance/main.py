import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from prompt_enhance.config import get_settings
from prompt_enhance.db.engine import async_session_factory, create_engine
from prompt_enhance.routes import enhance, health, keys, models, usage
from prompt_enhance.schemas.enhance import ErrorResponse
from prompt_enhance.services.cache import PromptCache
from prompt_enhance.services.llm import LLMService, LLMServiceError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings

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

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(enhance.router)
app.include_router(keys.router)
app.include_router(usage.router)


@app.exception_handler(LLMServiceError)
async def llm_error_handler(_request: Request, exc: LLMServiceError) -> JSONResponse:
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
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(Exception)
async def global_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    body = ErrorResponse(
        error="Internal server error",
        code="internal_error",
        detail=type(exc).__name__,
    )
    return JSONResponse(status_code=500, content=body.model_dump())
