"""Model sessions over HTTP: start, turn, delete.  Framing and status only.

The session id travels in the ``X-Session-Id`` header, never the path, and is
never logged (uvicorn's access log records paths).  Quota (D4): the daily cap
is counted once, on ``POST /v1/sessions`` (``get_llm_api_key``).  Turns and
delete use ``get_current_api_key``, so every one still passes the per-key
rate limit.  Sessions never read or write the prompt cache.

A turn goes get, lock, get again, cap checks, stream, store.  Anything that
fails before the stream is an HTTP error with an ``ErrorResponse`` body.
Anything that fails once the stream is open is an ``error`` event; the turn
is not stored and the lock is released, so the client may resend.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from koleslaw_api.config import Settings
from koleslaw_api.prompts.system import classify_enhancement_type
from koleslaw_api.schemas.enhance import ErrorResponse
from koleslaw_api.schemas.session import (
    ErrorEvent,
    SessionEvent,
    StartSessionRequest,
    TurnEvent,
    TurnRequest,
)
from koleslaw_api.services.auth import get_current_api_key, get_llm_api_key
from koleslaw_api.services.llm import LLMServiceError
from koleslaw_api.services.session_store import (
    Session,
    SessionBusyError,
    SessionError,
    SessionFullError,
    SessionNotFoundError,
    SessionStore,
)
from koleslaw_api.services.session_turn import TurnAnswer, TurnInput, estimate_input_tokens, stream_turn
from koleslaw_api.services.sse import sse_event, with_keepalive
from koleslaw_api.services.turn_protocol import TurnMode
from koleslaw_api.services.usage import log_usage

logger = logging.getLogger(__name__)

SESSION_ID_HEADER = "X-Session-Id"
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class SessionsDisabledError(SessionError):
    def __init__(self) -> None:
        super().__init__("sessions disabled")


_ERROR_STATUS: dict[type[SessionError], tuple[int, str]] = {
    SessionNotFoundError: (404, "session_not_found"),
    SessionBusyError: (409, "session_busy"),
    SessionFullError: (409, "session_full"),
    SessionsDisabledError: (503, "sessions_disabled"),
}


def session_error_response(exc: SessionError) -> JSONResponse:
    status, code = _ERROR_STATUS.get(type(exc), (500, "internal_error"))
    body = ErrorResponse(error=str(exc), code=code)
    return JSONResponse(status_code=status, content=body.model_dump())


def _require_sessions_enabled(request: Request) -> None:
    # A router dependency, so it runs before auth: a disabled flag never
    # spends a visitor's daily quota.
    if not request.app.state.settings.sessions_enabled:
        raise SessionsDisabledError()


router = APIRouter(prefix="/v1/sessions", tags=["sessions"], dependencies=[Depends(_require_sessions_enabled)])

SessionId = Annotated[str, Header(alias=SESSION_ID_HEADER, min_length=1, max_length=128)]


def _owner(api_key_id: uuid.UUID | None) -> str:
    # None only on the development auth bypass.
    return str(api_key_id)


def _mode(session: Session, requested: TurnMode, settings: Settings) -> TurnMode:
    """The last allowed user turn is always a finalize."""
    return "finalize" if session.user_turns + 1 >= settings.session_max_turns else requested


def _check_tokens(session: Session, user: TurnInput, model: str, settings: Settings) -> None:
    if estimate_input_tokens(session, user, model) > settings.session_max_input_tokens:
        raise SessionFullError()


async def _lock(store: SessionStore, session_id: str) -> AsyncExitStack:
    """Take the turn lock now; the stream releases it when the turn ends."""
    stack = AsyncExitStack()
    await stack.enter_async_context(store.lock(session_id))
    return stack


@router.post("")
async def start_session(
    request: Request,
    body: StartSessionRequest,
    api_key_id: uuid.UUID | None = Depends(get_llm_api_key),  # noqa: B008
) -> StreamingResponse:
    settings: Settings = request.app.state.settings
    store: SessionStore = request.app.state.session_store
    model = settings.default_model
    resolved_type = body.enhancement_type
    if resolved_type == "auto":
        resolved_type = classify_enhancement_type(body.prompt)

    owner = _owner(api_key_id)
    draft = Session(owner=owner, context=body.context, enhancement_type=resolved_type)
    user = TurnInput(body.prompt, _mode(draft, "reply", settings))
    _check_tokens(draft, user, model, settings)

    session_id, session = await store.create(owner, context=body.context, enhancement_type=resolved_type)
    lock = await _lock(store, session_id)
    opening = SessionEvent(
        session_id=session_id,
        expires_in=settings.session_ttl_seconds,
        turns_left=settings.session_max_turns - 1,
    )
    return _turn_response(request, session_id, session, user, lock, api_key_id, opening)


@router.post("/turns")
async def send_turn(
    request: Request,
    body: TurnRequest,
    session_id: SessionId,
    api_key_id: uuid.UUID | None = Depends(get_current_api_key),  # noqa: B008
) -> StreamingResponse:
    settings: Settings = request.app.state.settings
    store: SessionStore = request.app.state.session_store
    owner = _owner(api_key_id)

    await store.get(session_id, owner)
    lock = await _lock(store, session_id)
    try:
        # Read again under the lock: a turn that just finished has changed it.
        session = await store.get(session_id, owner)
        if session.user_turns >= settings.session_max_turns:
            raise SessionFullError()
        if body.message and len(body.message) > settings.session_max_reply_chars:
            raise HTTPException(status_code=422, detail="Reply is too long")
        if not body.message and not session.turns:
            raise HTTPException(status_code=422, detail="Nothing to finalize yet. Send the prompt as a reply.")
        user = TurnInput(body.message or "", _mode(session, body.action, settings))
        _check_tokens(session, user, settings.default_model, settings)
    except BaseException:
        await lock.aclose()
        raise
    return _turn_response(request, session_id, session, user, lock, api_key_id)


@router.delete("", status_code=204)
async def delete_session(
    request: Request,
    session_id: SessionId,
    api_key_id: uuid.UUID | None = Depends(get_current_api_key),  # noqa: B008
) -> Response:
    store: SessionStore = request.app.state.session_store
    await store.delete(session_id, _owner(api_key_id))
    return Response(status_code=204)


def _turn_response(
    request: Request,
    session_id: str,
    session: Session,
    user: TurnInput,
    lock: AsyncExitStack,
    api_key_id: uuid.UUID | None,
    opening: SessionEvent | None = None,
) -> StreamingResponse:
    settings: Settings = request.app.state.settings
    store: SessionStore = request.app.state.session_store
    done: list[TurnAnswer] = []
    start = time.monotonic()

    async def events() -> AsyncIterator[str]:
        try:
            answer: TurnAnswer | None = None
            async for item in stream_turn(request.app.state.llm_service, session, user, settings.default_model):
                if isinstance(item, TurnAnswer):
                    answer = item
                else:
                    yield sse_event(item.model_dump_json())
            if answer is None:
                raise RuntimeError("stream ended without an answer")
            stored = answer.turns(user)
            user_turns = session.user_turns
            if stored:
                user_turns = (
                    await store.append(session_id, session.owner, stored, input_tokens=answer.input_tokens)
                ).user_turns
            done.append(answer)
            await lock.aclose()
            turn = TurnEvent(
                kind=answer.kind,
                turn=session.user_turns + 1,
                turns_left=settings.session_max_turns - user_turns,
                model_used=answer.model_used,
            )
            yield sse_event(turn.model_dump_json())
        except LLMServiceError as exc:
            yield sse_event(ErrorEvent(code=exc.code, message=str(exc)).model_dump_json())
        except SessionNotFoundError as exc:
            # Deleted or expired while the turn streamed.
            yield sse_event(ErrorEvent(code="session_not_found", message=str(exc)).model_dump_json())
        except Exception:
            logger.exception("Session turn failed")
            message = "Something went wrong generating a response. Please try again."
            yield sse_event(ErrorEvent(code="internal_error", message=message).model_dump_json())
        finally:
            await lock.aclose()

    async def generate() -> AsyncIterator[str]:
        try:
            # First byte goes out at once so the proxy sees a response even
            # while the request is queued behind a model load.
            yield ": connected\n\n"
            if opening is not None:
                yield sse_event(opening.model_dump_json())
            async for event in with_keepalive(aiter(events())):
                yield event
            yield sse_event("[DONE]")
        finally:
            await lock.aclose()

    async def record_usage() -> None:
        if not done:
            return
        answer = done[0]
        await log_usage(
            request.app.state.db_session_factory,
            api_key_id,
            answer.model_used,
            session.enhancement_type,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            cached=False,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    response = StreamingResponse(generate(), media_type="text/event-stream", headers=_SSE_HEADERS)
    response.background = record_usage  # type: ignore[assignment]
    return response
