import logging
from importlib.metadata import version

from fastapi import APIRouter, Request
from sqlalchemy import text

from koleslaw_api.schemas.enhance import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["health"])

APP_VERSION = version("koleslaw-api")


@router.get("/health")
async def health_check(request: Request) -> HealthResponse:
    checks: dict[str, bool] = {}

    try:
        async with request.app.state.db_session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        logger.exception("Database health check failed")
        checks["database"] = False

    try:
        await request.app.state.redis.ping()
        checks["redis"] = True
    except Exception:
        logger.exception("Redis health check failed")
        checks["redis"] = False

    status = "ok" if all(checks.values()) else "degraded"
    return HealthResponse(status=status, version=APP_VERSION, checks=checks)
