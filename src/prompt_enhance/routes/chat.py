import base64
import json
import logging

import filetype
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from litellm import acompletion
from pydantic import ValidationError

from prompt_enhance.prompts.system import get_chat_system_prompt
from prompt_enhance.schemas.chat import ChatChunk, ChatMessage, ChatRequest, ChatResponse
from prompt_enhance.services.auth import get_current_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])

# ── File upload constraints ──────────────────────────────────────────
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_FILE_COUNT = 5

# MIME type prefixes that are blocked (archives / compressed formats)
BLOCKED_MIME_PREFIXES: tuple[str, ...] = (
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/x-bzip2",
    "application/x-xz",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/vnd.rar",
    "application/zstd",
    "application/x-lz4",
    "application/x-iso9660-image",
    "application/x-apple-diskimage",
    "application/java-archive",
)


def _sse_event(data: str) -> str:
    return f"data: {data}\n\n"


def _build_messages(body: ChatRequest) -> list[dict[str, str]]:
    system_prompt = get_chat_system_prompt(body.context)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in body.messages:
        messages.append({"role": msg.role, "content": msg.content})
    return messages


# ── File helpers ─────────────────────────────────────────────────────

async def _read_and_validate(upload: UploadFile) -> tuple[bytes, str]:
    """Read file bytes, enforce size limit, detect MIME via magic bytes, and block archives.

    Returns (data, detected_mime).
    """
    data = await upload.read()

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File '{upload.filename}' exceeds the {MAX_FILE_SIZE // (1024 * 1024)} MB limit.",
        )

    # Detect real type from file content, not from the extension or Content-Type header
    kind = filetype.guess(data)
    detected_mime = kind.mime if kind else None

    if detected_mime and detected_mime in BLOCKED_MIME_PREFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Compressed/archive files are not supported (detected: {detected_mime}).",
        )

    # Use the detected MIME when available, fall back to the browser-supplied header
    mime = detected_mime or upload.content_type or "application/octet-stream"
    return data, mime


def _file_to_content_block(data: bytes, mime: str, filename: str) -> dict:
    """Convert file bytes to a litellm-compatible content block.

    If filetype couldn't identify the file (kind is None), it's almost certainly
    text-based (source code, markdown, config, etc.) since filetype only recognises
    binary formats.  We also check the MIME for an explicit text/* prefix.
    """
    is_text = mime.startswith("text/") or mime == "application/octet-stream"

    if is_text:
        text = data.decode("utf-8", errors="replace")
        return {"type": "text", "text": f"[Attached file: {filename}]\n\n{text}"}

    # Binary files (images, PDFs, etc.): base64-encode as an image_url block
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _build_messages_with_files(
    body: ChatRequest,
    file_blocks: list[dict],
) -> list[dict]:
    """Build the LLM message list, attaching file content blocks to the last user message."""
    system_prompt = get_chat_system_prompt(body.context)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for i, msg in enumerate(body.messages):
        is_last_user = i == len(body.messages) - 1 and msg.role == "user"
        if is_last_user and file_blocks:
            # Convert to multimodal content array
            content: list[dict] = [{"type": "text", "text": msg.content}] + file_blocks
            messages.append({"role": msg.role, "content": content})
        else:
            messages.append({"role": msg.role, "content": msg.content})

    return messages


# ── Streaming helper ─────────────────────────────────────────────────

async def _stream_chat(model: str, messages: list[dict]):
    """Shared SSE generator used by both endpoints."""
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


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/chat")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    _api_key: str | None = Depends(get_current_api_key),  # noqa: B008
) -> StreamingResponse:
    model = body.model or request.app.state.settings.default_model
    messages = _build_messages(body)

    return StreamingResponse(
        _stream_chat(model, messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/upload")
async def chat_stream_with_files(
    request: Request,
    messages: str = Form(...),
    files: list[UploadFile] = File(default=[]),  # noqa: B008
    model: str | None = Form(default=None),
    context: str | None = Form(default=None),
    _api_key: str | None = Depends(get_current_api_key),  # noqa: B008
) -> StreamingResponse:
    """Chat endpoint that accepts file attachments via multipart/form-data.

    The `messages` field is a JSON-encoded array of ChatMessage objects.
    Files are validated for type and size, then forwarded to the LLM as
    content blocks alongside the last user message.
    """
    # Parse the JSON-encoded messages field
    try:
        raw = json.loads(messages)
        parsed_messages = [ChatMessage(**m) for m in raw]
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid messages JSON: {exc}") from exc

    if len(files) > MAX_FILE_COUNT:
        raise HTTPException(status_code=422, detail=f"Too many files. Maximum is {MAX_FILE_COUNT}.")

    # Read, validate (magic-byte type detection + size), and convert to content blocks
    file_blocks: list[dict] = []
    for f in files:
        data, mime = await _read_and_validate(f)
        block = _file_to_content_block(data, mime, f.filename or "attachment")
        file_blocks.append(block)

    body = ChatRequest(messages=parsed_messages, model=model, context=context)
    resolved_model = body.model or request.app.state.settings.default_model
    llm_messages = _build_messages_with_files(body, file_blocks)

    return StreamingResponse(
        _stream_chat(resolved_model, llm_messages),
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
