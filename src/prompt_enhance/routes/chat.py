import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from litellm import acompletion

from prompt_enhance.prompts.system import get_chat_system_prompt
from prompt_enhance.schemas.chat import ChatChunk, ChatMessage, ChatRequest, ChatResponse
from prompt_enhance.services.auth import get_current_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


def _sse_event(data: str) -> str:
    return f"data: {data}\n\n"


def _build_messages(body: ChatRequest) -> list[dict[str, str]]:
    system_prompt = get_chat_system_prompt(body.context)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in body.messages:
        messages.append({"role": msg.role, "content": msg.content})
    return messages


@router.post("/chat")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    _api_key: str | None = Depends(get_current_api_key),  # noqa: B008
) -> StreamingResponse:
    model = body.model or request.app.state.settings.default_model
    messages = _build_messages(body)

    async def generate():
        try:
            response = await acompletion(
                model=model,
                messages=messages,
                stream=True,
                num_retries=3,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                finish_reason = chunk.choices[0].finish_reason
                if delta or finish_reason:
                    chat_chunk = ChatChunk(delta=delta, finish_reason=finish_reason)
                    yield _sse_event(chat_chunk.model_dump_json())
            yield _sse_event("[DONE]")
        except Exception as exc:
            logger.exception("Chat stream failed")
            error_chunk = ChatChunk(delta=f"\n\n[Error: {exc}]", finish_reason="stop")
            yield _sse_event(error_chunk.model_dump_json())
            yield _sse_event("[DONE]")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/sync")
async def chat_sync(
    request: Request,
    body: ChatRequest,
    _api_key: str | None = Depends(get_current_api_key),  # noqa: B008
) -> ChatResponse:
    model = body.model or request.app.state.settings.default_model
    messages = _build_messages(body)

    response = await acompletion(
        model=model,
        messages=messages,
        num_retries=3,
    )

    content = response.choices[0].message.content
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content),
        model_used=model,
        usage={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )
