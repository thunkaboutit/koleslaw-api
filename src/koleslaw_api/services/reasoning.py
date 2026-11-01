"""Split model output into thinking and answer text.

Reasoning models deliver chain of thought two ways.  Some providers return it
in a separate native field (LiteLLM's ``reasoning_content``).  Others inline it
in the answer as ``<think>...</think>``.  LiteLLM 1.82.2's Ollama stream parser
only half handles the inline form, so tags, unclosed blocks and close-only
blocks (a template that pre-opens ``<think>``) reach us in ``content``.

``ReasoningSplitter`` is the one implementation.  It is pure and stateful: feed
it each chunk, then ``flush`` at the end.  ``split_reasoning`` runs it once
over a whole string for the one-shot paths.

Rules:

- Native reasoning is always thinking.
- ``<think>`` switches to thinking, ``</think>`` back to text.  Tags are never
  emitted.
- A ``</think>`` with no tag seen before it means the block was opened by the
  template.  Text in the same feed before it is thinking.  Text already
  returned by an earlier feed cannot be recalled.
- Whitespace right after a ``</think>`` is dropped from the answer.
- A tag split across chunks is held until the next feed or ``flush``.
- An unclosed ``<think>`` stays thinking to the end.
"""

from dataclasses import dataclass
from typing import Literal

PartKind = Literal["thinking", "text"]

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


@dataclass(frozen=True, slots=True)
class Part:
    kind: PartKind
    text: str


@dataclass(frozen=True, slots=True)
class SplitResult:
    thinking: str
    text: str


def _partial_tag_suffix(s: str) -> int:
    """Length of the longest suffix of ``s`` that could start a tag."""
    for n in range(min(len(s), len(CLOSE_TAG) - 1), 0, -1):
        tail = s[-n:]
        if OPEN_TAG.startswith(tail) or CLOSE_TAG.startswith(tail):
            return n
    return 0


class ReasoningSplitter:
    def __init__(self) -> None:
        self._mode: PartKind = "text"
        self._carry = ""
        self._saw_tag = False
        self._strip_leading = False

    def feed(self, content: str | None = None, reasoning: str | None = None) -> list[Part]:
        parts: list[Part] = []
        if reasoning:
            parts.append(Part("thinking", reasoning))
        if content:
            self._scan(self._carry + content, parts, final=False)
        return _merge(parts)

    def flush(self) -> list[Part]:
        parts: list[Part] = []
        if self._carry:
            self._scan(self._carry, parts, final=True)
        return _merge(parts)

    def _scan(self, s: str, parts: list[Part], *, final: bool) -> None:
        self._carry = ""
        while s:
            open_at = s.find(OPEN_TAG)
            close_at = s.find(CLOSE_TAG)
            if open_at == -1 and close_at == -1:
                hold = 0 if final else _partial_tag_suffix(s)
                self._emit(self._mode, s[: len(s) - hold], parts)
                self._carry = s[len(s) - hold :]
                return
            if close_at == -1 or (open_at != -1 and open_at < close_at):
                self._emit(self._mode, s[:open_at], parts)
                self._mode = "thinking"
                s = s[open_at + len(OPEN_TAG) :]
            else:
                before = s[:close_at]
                kind: PartKind = "thinking" if not self._saw_tag else self._mode
                self._emit(kind, before, parts)
                self._mode = "text"
                self._strip_leading = True
                s = s[close_at + len(CLOSE_TAG) :]
            self._saw_tag = True

    def _emit(self, kind: PartKind, text: str, parts: list[Part]) -> None:
        if kind == "text" and self._strip_leading:
            text = text.lstrip()
            if text:
                self._strip_leading = False
        if text:
            parts.append(Part(kind, text))


def _merge(parts: list[Part]) -> list[Part]:
    merged: list[Part] = []
    for part in parts:
        if merged and merged[-1].kind == part.kind:
            merged[-1] = Part(part.kind, merged[-1].text + part.text)
        else:
            merged.append(part)
    return merged


def split_reasoning(content: str | None, reasoning: str | None = None) -> SplitResult:
    """Split one whole response.  Same rules as the streaming splitter."""
    splitter = ReasoningSplitter()
    parts = splitter.feed(content, reasoning) + splitter.flush()
    return SplitResult(
        thinking="".join(p.text for p in parts if p.kind == "thinking"),
        text="".join(p.text for p in parts if p.kind == "text"),
    )
