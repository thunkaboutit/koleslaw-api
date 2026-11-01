"""Wire contract for model sessions.  The UI mirrors this file.

Requests:

- ``POST /v1/sessions`` takes ``StartSessionRequest`` and starts a session.
- ``POST /v1/sessions/turns`` takes ``TurnRequest``.  The session is named by
  the ``X-Session-Id`` header, never the path.
- ``DELETE /v1/sessions`` takes no body.  Always 204.

Both POSTs answer with server-sent events: a ``: connected`` comment, then
one JSON object per ``data:`` line, discriminated by ``type``, then
``data: [DONE]``.  Comment lines (``: keepalive``) may arrive at any time.

- ``session``: first event of ``POST /v1/sessions`` only.
- ``thinking`` / ``text``: reasoning and answer deltas, any number.
- ``turn``: last event of a turn that completed.
- ``error``: last event of a turn that failed after the stream opened.  The
  turn is not stored; the session is as it was before the request.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from koleslaw_api.services.turn_protocol import TurnKind, TurnMode

EnhancementType = Literal["auto", "expand", "refine", "structure"]


class StartSessionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=50000)
    context: str | None = Field(default=None, max_length=50000)
    enhancement_type: EnhancementType = "auto"


class TurnRequest(BaseModel):
    action: TurnMode = "reply"
    message: str | None = Field(default=None, min_length=1, max_length=10000)

    @model_validator(mode="after")
    def _reply_needs_a_message(self) -> "TurnRequest":
        if self.action == "reply" and self.message is None:
            raise ValueError("message is required for a reply")
        return self


class SessionEvent(BaseModel):
    """``session_id`` goes in ``X-Session-Id`` on every later request.  Hold
    it in memory only.  ``expires_in`` is the sliding TTL in seconds, reset
    by each stored turn.  ``turns_left`` is as in ``TurnEvent``."""

    type: Literal["session"] = "session"
    session_id: str
    expires_in: int
    turns_left: int


class ThinkingEvent(BaseModel):
    type: Literal["thinking"] = "thinking"
    delta: str


class TextEvent(BaseModel):
    type: Literal["text"] = "text"
    delta: str


class TurnEvent(BaseModel):
    """``kind`` is ``question``, ``prompt`` or ``empty``.  ``turn`` is the
    1-based number of this user turn.  ``turns_left`` is how many more user
    turns the session accepts; the last of them is always a finalize, and
    at 0 the next one gets 409 ``session_full``.  An ``empty`` turn is not
    stored or counted, so its ``turn`` number is used again."""

    type: Literal["turn"] = "turn"
    kind: TurnKind
    turn: int
    turns_left: int
    model_used: str


class ErrorEvent(BaseModel):
    """``code`` is an ``LLMServiceError`` code (``connection_error``,
    ``service_unavailable``, ``rate_limit_error``, ``authentication_error``,
    ``bad_request``, ``model_not_found``, ``llm_error``), ``session_not_found``
    (deleted mid-turn) or ``internal_error``.  ``message`` is safe to show."""

    type: Literal["error"] = "error"
    code: str
    message: str


SessionStreamEvent = Annotated[
    SessionEvent | ThinkingEvent | TextEvent | TurnEvent | ErrorEvent,
    Field(discriminator="type"),
]
