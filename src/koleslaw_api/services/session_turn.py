"""One session turn: the model input, the stream, and the classified answer.

The transcript is sent whole on every turn.  The ask-or-finalize trailer
rides on the latest user turn only, and a stored question goes back to the
model with its ``[[ASK]]`` marker, so the model sees its own protocol.

A local (Ollama) model gets no system message, so its Modelfile applies, and
the session context rides on the first user turn.  A cloud fallback model
gets the per-type cloud system prompt with the context instead.

No I/O here except the model call.  The routes own Redis and HTTP.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import litellm

from koleslaw_api.prompts.system import get_system_prompt
from koleslaw_api.schemas.session import TextEvent, ThinkingEvent
from koleslaw_api.services.llm import LLMService, Messages, is_local_model, with_context
from koleslaw_api.services.session_store import Session, Turn
from koleslaw_api.services.turn_protocol import ASK_MARKER, MarkerFilter, TurnKind, TurnMode, trailer, with_trailer


@dataclass(frozen=True, slots=True)
class TurnInput:
    """What the user sent.  ``message`` is empty for a bare finalize."""

    message: str
    mode: TurnMode

    @property
    def stored(self) -> str:
        """The user turn as kept in the transcript."""
        return self.message or trailer(self.mode)


@dataclass(frozen=True, slots=True)
class TurnAnswer:
    """The end of a streamed turn.  ``text`` has the marker stripped."""

    kind: TurnKind
    text: str
    model_used: str
    input_tokens: int
    output_tokens: int

    def turns(self, user: TurnInput) -> list[Turn]:
        """The turns to store.  An empty answer stores nothing."""
        if self.kind == "empty":
            return []
        return [Turn("user", user.stored), Turn("assistant", self.text, self.kind)]


def build_messages(session: Session, user: TurnInput, model: str) -> Messages:
    local = is_local_model(model)
    context = session.context if local else None
    messages: Messages = []
    for turn in session.turns:
        content = f"{ASK_MARKER}\n{turn.content}" if turn.kind == "question" else turn.content
        if not messages:
            content = with_context(content, context)
        messages.append({"role": turn.role, "content": content})
    latest = user.message if messages else with_context(user.message, context)
    messages.append({"role": "user", "content": with_trailer(latest, user.mode) if latest else trailer(user.mode)})
    if local:
        return messages
    system = get_system_prompt(session.enhancement_type, session.context)
    return [{"role": "system", "content": system}, *messages]


def estimate_tokens(model: str, *, messages: Messages | None = None, text: str | None = None) -> int:
    """LiteLLM's estimate, or about four characters a token if it fails."""
    try:
        if messages is not None:
            return litellm.token_counter(model=model, messages=messages)
        return litellm.token_counter(model=model, text=text or "")
    except Exception:
        chars = sum(len(m["content"]) for m in messages) if messages is not None else len(text or "")
        return chars // 4


def estimate_input_tokens(session: Session, user: TurnInput, model: str) -> int:
    return estimate_tokens(model, messages=build_messages(session, user, model))


async def stream_turn(
    llm: LLMService, session: Session, user: TurnInput, model: str
) -> AsyncGenerator[ThinkingEvent | TextEvent | TurnAnswer, None]:
    """Yield thinking and text events, then one ``TurnAnswer``.

    Raises ``LLMServiceError`` if the model fails before or during the
    stream.  Fallback happens only before the first byte (in ``LLMService``).
    """
    stream = await llm.stream_messages(lambda m: build_messages(session, user, m), model)
    marker = MarkerFilter(user.mode)
    raw: list[str] = []
    async for piece in stream.deltas:
        if piece.kind == "thinking":
            yield ThinkingEvent(delta=piece.text)
            continue
        raw.append(piece.text)
        if text := marker.feed(piece.text):
            yield TextEvent(delta=text)
    if text := marker.flush():
        yield TextEvent(delta=text)
    result = marker.result()
    yield TurnAnswer(
        kind=result.kind,
        text=result.text,
        model_used=stream.model,
        input_tokens=estimate_input_tokens(session, user, stream.model),
        output_tokens=estimate_tokens(stream.model, text="".join(raw)),
    )
