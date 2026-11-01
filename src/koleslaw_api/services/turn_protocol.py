"""The ask-or-finalize protocol for session turns.

A session turn tells the model, through a short trailer on the latest user
turn, that it may ask instead of answering.  The model marks a question by
starting its reply with ``[[ASK]]``.  Everything else is an enhanced prompt.
There is no "ends with ?" heuristic: enhanced prompts often end in one.

``classify`` is the pure rule over a whole answer.  ``MarkerFilter`` applies
the same rule to a stream of answer text (after ``ReasoningSplitter``): it
holds back only a trailing piece that could still be the start of a marker
and releases it as soon as it cannot be, so no marker ever reaches the client.

Rules:

- Every marker is stripped, wherever the model puts it, together with one
  newline right after it if present (so no blank line is left behind).
- The kind is decided only by position: a marker at the start of the answer
  (after optional whitespace) makes it a ``question`` in reply mode.  A marker
  anywhere else is just removed and the answer stays a ``prompt``.
- After a leading marker, whitespace before the first real character is
  dropped, as is the whitespace before the marker.
- In finalize mode the markers are still stripped, but the kind is ``prompt``.
- Text that is blank once the markers are removed is ``empty``.
"""

import re
from dataclasses import dataclass
from typing import Literal

TurnMode = Literal["reply", "finalize"]
TurnKind = Literal["question", "prompt", "empty"]

ASK_MARKER = "[[ASK]]"
_MARKER_RE = re.compile(re.escape(ASK_MARKER) + r"\n?")

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


def _strip_markers(text: str) -> str:
    return _MARKER_RE.sub("", text)


def classify(text: str, mode: TurnMode = "reply") -> Classified:
    asked = text.lstrip().startswith(ASK_MARKER)
    text = _strip_markers(text)
    if asked:
        text = text.lstrip()
    kind = _kind(text, asked, mode)
    return Classified(kind, "" if kind == "empty" else text)


class MarkerFilter:
    """Streaming ``classify``: feed answer deltas, ``flush``, then ``result``."""

    def __init__(self, mode: TurnMode = "reply") -> None:
        self._mode = mode
        self._buf = ""  # raw text not yet processed (may end in a marker prefix)
        self._lead = ""  # whitespace-only text held until the first real character
        self._seen_text = False
        self._asked = False
        self._skip_newline = False
        self._emitted: list[str] = []

    def feed(self, delta: str) -> str:
        self._buf += delta
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def result(self) -> Classified:
        text = "".join(self._emitted)
        kind = _kind(text, self._asked, self._mode)
        return Classified(kind, "" if kind == "empty" else text)

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        while self._buf:
            if self._skip_newline:
                self._skip_newline = False
                if self._buf.startswith("\n"):
                    self._buf = self._buf[1:]
                    continue
            idx = self._buf.find(ASK_MARKER)
            if idx >= 0:
                out.append(self._emit(self._buf[:idx]))
                self._buf = self._buf[idx + len(ASK_MARKER) :]
                if not self._seen_text:
                    self._asked = True
                    self._lead = ""
                self._skip_newline = True
                continue
            keep = 0 if final else self._partial_marker_len()
            out.append(self._emit(self._buf[: len(self._buf) - keep]))
            self._buf = self._buf[len(self._buf) - keep :]
            break
        return "".join(out)

    def _partial_marker_len(self) -> int:
        for k in range(len(ASK_MARKER) - 1, 0, -1):
            if self._buf.endswith(ASK_MARKER[:k]):
                return k
        return 0

    def _emit(self, text: str) -> str:
        if self._asked and not self._seen_text:
            text = text.lstrip()
        if not text:
            return ""
        if not self._seen_text:
            if not text.strip():
                self._lead += text
                return ""
            self._seen_text = True
            text, self._lead = self._lead + text, ""
        self._emitted.append(text)
        return text
