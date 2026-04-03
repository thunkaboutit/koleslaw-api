import asyncio
import logging
import time
import uuid

import litellm
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from prompt_enhance.schemas.enhance import EnhanceRequest, EnhanceResponse
from prompt_enhance.services.auth import get_current_api_key
from prompt_enhance.services.cache import PromptCache
from prompt_enhance.services.llm import LLMService
from prompt_enhance.services.usage import log_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["enhance"])


def _sse_event(data: str) -> str:
    return f"data: {data}\n\n"


@router.post("/enhance")
async def enhance_stream(
    request: Request,
    body: EnhanceRequest,
    api_key_id: uuid.UUID | None = Depends(get_current_api_key),  # noqa: B008
) -> StreamingResponse:
    llm: LLMService = request.app.state.llm_service
    cache: PromptCache = request.app.state.cache
    model = body.model or request.app.state.settings.default_model
    resolved_type = LLMService.resolve_enhancement_type(body)

    cached = await cache.get(body.prompt, resolved_type, model)
    if cached is not None:
        hit = EnhanceResponse(
            enhanced_prompt=cached,
            model_used=model,
            usage={"input_tokens": 0, "output_tokens": 0},
            cached=True,
        )

        async def cached_stream():
            yield _sse_event(hit.model_dump_json())
            yield _sse_event("[DONE]")

        response = StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

        async def log_cached_usage():
            await log_usage(
                request.app.state.db_session_factory,
                api_key_id,
                model,
                resolved_type,
                input_tokens=0,
                output_tokens=0,
                cached=True,
                latency_ms=0,
            )

        response.background = log_cached_usage  # type: ignore[assignment]
        return response

    collected: list[str] = []
    start = time.monotonic()

    async def generate():
        async for chunk in llm.enhance_stream(body, resolved_type):
            collected.append(chunk.delta)
            yield _sse_event(chunk.model_dump_json())
        yield _sse_event("[DONE]")

    async def after_stream():
        await asyncio.sleep(0)
        full_response = "".join(collected)
        latency_ms = int((time.monotonic() - start) * 1000)

        if full_response:
            try:
                ttl = request.app.state.settings.cache_ttl_seconds
                await cache.set(body.prompt, resolved_type, model, full_response, ttl=ttl)
            except Exception:
                logger.exception("Failed to cache response")

        messages = [{"role": "user", "content": body.prompt}]
        try:
            input_tokens = litellm.token_counter(model=model, messages=messages)
            output_tokens = litellm.token_counter(model=model, text=full_response)
        except Exception:
            input_tokens = 0
            output_tokens = 0

        await log_usage(
            request.app.state.db_session_factory,
            api_key_id,
            model,
            resolved_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
            latency_ms=latency_ms,
        )

    response = StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    response.background = after_stream  # type: ignore[assignment]
    return response


@router.post("/enhance/sync")
async def enhance_sync(
    request: Request,
    body: EnhanceRequest,
    api_key_id: uuid.UUID | None = Depends(get_current_api_key),  # noqa: B008
) -> EnhanceResponse:
    llm: LLMService = request.app.state.llm_service
    cache: PromptCache = request.app.state.cache
    model = body.model or request.app.state.settings.default_model
    resolved_type = LLMService.resolve_enhancement_type(body)

    cached = await cache.get(body.prompt, resolved_type, model)
    if cached is not None:
        await log_usage(
            request.app.state.db_session_factory,
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

    ttl = request.app.state.settings.cache_ttl_seconds
    await cache.set(body.prompt, resolved_type, model, result.enhanced_prompt, ttl=ttl)

    await log_usage(
        request.app.state.db_session_factory,
        api_key_id,
        model,
        resolved_type,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cached=False,
        latency_ms=latency_ms,
    )

    return result
