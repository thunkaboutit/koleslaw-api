import logging
import uuid

from prompt_enhance.db.models import UsageLog

logger = logging.getLogger(__name__)


async def log_usage(
    db_session_factory,
    api_key_id: uuid.UUID | None,
    model: str,
    enhancement_type: str,
    input_tokens: int,
    output_tokens: int,
    cached: bool,
    latency_ms: int,
) -> None:
    if api_key_id is None:
        return

    try:
        async with db_session_factory() as session:
            session.add(
                UsageLog(
                    api_key_id=api_key_id,
                    model=model,
                    enhancement_type=enhancement_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=cached,
                    latency_ms=latency_ms,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("Failed to log usage")
