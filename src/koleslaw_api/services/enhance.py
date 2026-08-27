"""One enhancement, cache-or-generate, usage logged.

Extracted from the /v1/enhance/sync route so the REST endpoint and the MCP
endpoint share a single implementation. Two front doors onto the same path
cannot drift; two copies of this would.
"""

import time
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from koleslaw_api.config import Settings
from koleslaw_api.schemas.enhance import EnhanceRequest, EnhanceResponse
from koleslaw_api.services.cache import PromptCache
from koleslaw_api.services.llm import LLMService, resolve_allowed_model
from koleslaw_api.services.usage import log_usage

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


async def enhance_once(
    body: EnhanceRequest,
    api_key_id: uuid.UUID | None,
    *,
    llm: LLMService,
    cache: PromptCache,
    settings: Settings,
    db_session_factory: SessionFactory,
) -> EnhanceResponse:
    """Enhance one prompt, serving a cache hit when there is one.

    The caller is responsible for authentication and caps — by the time a
    request reaches here it has already been allowed.
    """
    model = resolve_allowed_model(body.model, settings)
    resolved_type = LLMService.resolve_enhancement_type(body)

    cached = await cache.get(body.prompt, resolved_type, model)
    if cached is not None:
        await log_usage(
            db_session_factory,
            api_key_id,
            model,
            resolved_type,
            input_tokens=0,
            output_tokens=0,
            cached=True,
            latency_ms=0,
        )
        return EnhanceResponse(
            enhanced_prompt=cached,
            model_used=model,
            usage={"input_tokens": 0, "output_tokens": 0},
            cached=True,
        )

    start = time.monotonic()
    result = await llm.enhance(body, resolved_type)
    latency_ms = int((time.monotonic() - start) * 1000)

    await cache.set(body.prompt, resolved_type, model, result.enhanced_prompt, ttl=settings.cache_ttl_seconds)

    await log_usage(
        db_session_factory,
        api_key_id,
        model,
        resolved_type,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cached=False,
        latency_ms=latency_ms,
    )

    return result
