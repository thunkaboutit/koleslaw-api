import asyncio
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from prompt_enhance.schemas.enhance import EnhanceRequest, EnhanceResponse
from prompt_enhance.services.auth import get_current_api_key
from prompt_enhance.services.cache import PromptCache
from prompt_enhance.services.llm import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["enhance"])


def _sse_event(data: str) -> str:
    return f"data: {data}\n\n"


@router.post("/enhance")
async def enhance_stream(
    request: Request,
    body: EnhanceRequest,
    _api_key: str | None = Depends(get_current_api_key),  # noqa: B008
) -> StreamingResponse:
    llm: LLMService = request.app.state.llm_service
    cache: PromptCache = request.app.state.cache
    model = body.model or request.app.state.settings.default_model

    cached = await cache.get(body.prompt, body.enhancement_type, model)
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

        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    collected: list[str] = []

    async def generate():
        async for chunk in llm.enhance_stream(body):
            collected.append(chunk.delta)
            yield _sse_event(chunk.model_dump_json())
        yield _sse_event("[DONE]")

    async def cache_after_stream():
        await asyncio.sleep(0)
        full_response = "".join(collected)
        if full_response:
            try:
                ttl = request.app.state.settings.cache_ttl_seconds
                await cache.set(body.prompt, body.enhancement_type, model, full_response, ttl=ttl)
            except Exception:
                logger.exception("Failed to cache response")

    response = StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    response.background = cache_after_stream  # type: ignore[assignment]
    return response


@router.post("/enhance/sync")
async def enhance_sync(
    request: Request,
    body: EnhanceRequest,
    _api_key: str | None = Depends(get_current_api_key),  # noqa: B008
) -> EnhanceResponse:
    llm: LLMService = request.app.state.llm_service
    cache: PromptCache = request.app.state.cache
    model = body.model or request.app.state.settings.default_model

    cached = await cache.get(body.prompt, body.enhancement_type, model)
    if cached is not None:
        return EnhanceResponse(
            enhanced_prompt=cached,
            model_used=model,
            usage={"input_tokens": 0, "output_tokens": 0},
            cached=True,
        )

    try:
        result = await llm.enhance(body)
    except LLMServiceError:
        raise
    except Exception:
        raise

    ttl = request.app.state.settings.cache_ttl_seconds
    await cache.set(body.prompt, body.enhancement_type, model, result.enhanced_prompt, ttl=ttl)
    return result
