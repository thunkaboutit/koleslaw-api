import asyncio
import logging
import time
import uuid

import litellm
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from koleslaw_api.schemas.enhance import EnhanceRequest, EnhanceResponse
from koleslaw_api.services.auth import get_current_api_key
from koleslaw_api.services.cache import PromptCache
from koleslaw_api.services.llm import LLMService
from koleslaw_api.services.usage import log_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["enhance"])


def _sse_event(data: str) -> str:
    return f"data: {data}\n\n"


# Cloudflare drops proxied connections that stay silent for 100s (not
# configurable below Enterprise). The thinking phase of a reasoning model
# produces no enhance deltas, so long generations 524'd. SSE comment lines
# every KEEPALIVE_SECONDS keep bytes flowing; spec-compliant and line-based
# ("data: "-prefixed) parsers both ignore them.
KEEPALIVE_SECONDS = 20.0
_KEEPALIVE = ": keepalive\n\n"


async def _with_keepalive(agen):
    """Yield from an async generator, emitting SSE comments during stalls."""
    next_task = None
    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(anext(agen))
            done, _ = await asyncio.wait({next_task}, timeout=KEEPALIVE_SECONDS)
            if not done:
                yield _KEEPALIVE
                continue
            next_task = None
            try:
                yield done.pop().result()
            except StopAsyncIteration:
                return
    finally:
        if next_task is not None:
            next_task.cancel()


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

    async def stream_chunks():
        async for chunk in llm.enhance_stream(body, resolved_type):
            collected.append(chunk.delta)
            yield _sse_event(chunk.model_dump_json())

    async def generate():
        # First byte goes out immediately so the proxy sees a response even
        # while the request is queued behind a model load.
        yield ": connected\n\n"
        async for event in _with_keepalive(aiter(stream_chunks())):
            yield event
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
