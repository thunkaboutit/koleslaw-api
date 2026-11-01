"""The ask-or-finalize protocol for session turns.

A session turn tells the model, through a short trailer on the latest user
turn, that it may ask instead of answering.  The model marks a question by
starting its reply with ``[[ASK]]``.  Everything else is an enhanced prompt.
There is no "ends with ?" heuristic: enhanced prompts often end in one.

``classify`` is the pure rule over a whole answer.  ``MarkerFilter`` applies
the same rule to a stream of answer text (after ``ReasoningSplitter``): it
holds back leading text only while it could still be the marker, so the
marker never reaches the client, then passes text straight through.

Rules:

- Leading whitespace before the marker is ignored.  Whitespace right after it
  is dropped.
- In finalize mode the marker is still stripped, but the kind is ``prompt``.
- Text that is blank once the marker is removed is ``empty``.
"""

from dataclasses import dataclass
from typing import Literal

TurnMode = Literal["reply", "finalize"]
TurnKind = Literal["question", "prompt", "empty"]

ASK_MARKER = "[[ASK]]"

_TRAILERS: dict[TurnMode, str] = {
    "reply": (
        "If the prompt is too ambiguous to enhance well, start your reply with the line "
        f"{ASK_MARKER} and ask up to 3 short questions.  Otherwise return only the enhanced prompt."
    ),
    "finalize": "Do not ask questions.  Make reasonable assumptions and return only the enhanced prompt.",
}


@dataclass(frozen=True, slots=True)
class Classified:
    kind: TurnKind
    text: str


def trailer(mode: TurnMode) -> str:
    return _TRAILERS[mode]


def with_trailer(content: str, mode: TurnMode) -> str:
    """The latest user turn as sent to the model."""
    return f"{content}\n\n{trailer(mode)}"


def _kind(text: str, asked: bool, mode: TurnMode) -> TurnKind:
    if not text.strip():
        return "empty"
    return "question" if asked and mode == "reply" else "prompt"


def classify(text: str, mode: TurnMode = "reply") -> Classified:
    stripped = text.lstrip()
    asked = stripped.startswith(ASK_MARKER)
    if asked:
        text = stripped[len(ASK_MARKER) :].lstrip()
    kind = _kind(text, asked, mode)
    return Classified(kind, "" if kind == "empty" else text)


class MarkerFilter:
    """Streaming ``classify``: feed answer deltas, ``flush``, then ``result``."""

    def __init__(self, mode: TurnMode = "reply") -> None:
        self._mode = mode
        self._held = ""
        self._decided = False
        self._asked = False
        self._strip_leading = False
        self._emitted: list[str] = []

    def feed(self, delta: str) -> str:
        if self._decided:
            return self._emit(delta)
        self._held += delta
        head = self._held.lstrip()
        if ASK_MARKER.startswith(head):
            return ""
        self._decided = True
        self._asked = head.startswith(ASK_MARKER)
        if self._asked:
            self._strip_leading = True
            return self._emit(head[len(ASK_MARKER) :])
        return self._emit(self._held)

    def flush(self) -> str:
        if self._decided:
            return ""
        self._decided = True
        head = self._held.lstrip()
        self._asked = head == ASK_MARKER
        if not head or self._asked:
            return ""
        return self._emit(self._held)

    def result(self) -> Classified:
        text = "".join(self._emitted)
        kind = _kind(text, self._asked, self._mode)
        return Classified(kind, "" if kind == "empty" else text)

    def _emit(self, text: str) -> str:
        if self._strip_leading:
            text = text.lstrip()
            if text:
                self._strip_leading = False
        if text:
            self._emitted.append(text)
        return text
