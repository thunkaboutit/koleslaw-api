"""The reasoning splitter (PRD scenario 1, cases A to F).

Chunk lists mirror what LiteLLM 1.82.2 hands us in ``delta.content`` and
``delta.reasoning_content`` for each output shape.
"""

import pytest

from koleslaw_api.services.reasoning import Part, ReasoningSplitter, split_reasoning


def _run(chunks: list[tuple[str | None, str | None]]) -> tuple[str, str, list[Part]]:
    """Feed (content, reasoning) chunks, flush, return (thinking, text, parts)."""
    splitter = ReasoningSplitter()
    parts: list[Part] = []
    for content, reasoning in chunks:
        parts += splitter.feed(content, reasoning)
    parts += splitter.flush()
    thinking = "".join(p.text for p in parts if p.kind == "thinking")
    text = "".join(p.text for p in parts if p.kind == "text")
    return thinking, text, parts


def test_inline_think_block_in_one_chunk() -> None:
    thinking, text, _ = _run([("<think>a</think>b", None)])
    assert (thinking, text) == ("a", "b")


def test_native_reasoning_then_text_case_a() -> None:
    thinking, text, _ = _run(
        [
            (None, "User prompt is vague. "),
            (None, "I should ask."),
            ("What language ", None),
            ("are you using?", None),
        ]
    )
    assert thinking == "User prompt is vague. I should ask."
    assert text == "What language are you using?"


def test_reasoning_only_case_b_has_no_text() -> None:
    thinking, text, _ = _run([(None, "Too vague. "), (None, "what stack?"), ("", None)])
    assert thinking == "Too vague. what stack?"
    assert text == ""


def test_inline_tags_in_separate_chunks_case_c() -> None:
    thinking, text, _ = _run(
        [("<think>", None), ("Need more info", None), ("</think>", None), ("Which framework?", None)]
    )
    assert thinking == "Need more info"
    assert text == "Which framework?"


def test_unclosed_think_stays_thinking_case_d() -> None:
    thinking, text, _ = _run([("<think>", None), ("Need more info. Which framework?", None)])
    assert thinking == "Need more info. Which framework?"
    assert text == ""


def test_thinking_resumes_after_text_case_e() -> None:
    thinking, text, parts = _run(
        [(None, "plan"), ("Q: which DB?\n", None), (None, "assume postgres"), ("Enhanced: ...", None)]
    )
    assert thinking == "planassume postgres"
    assert text == "Q: which DB?\nEnhanced: ..."
    assert [p.kind for p in parts] == ["thinking", "text", "thinking", "text"]


def test_inline_thinking_resumes_after_text() -> None:
    thinking, text, _ = _run([("<think>a</think>b<think>c</think>d", None)])
    assert (thinking, text) == ("ac", "bd")


def test_close_only_from_pre_opened_template_case_f() -> None:
    thinking, text, _ = _run([("vague prompt, ask</think>Which framework?", None)])
    assert thinking == "vague prompt, ask"
    assert text == "Which framework?"


def test_close_only_tag_is_never_emitted_when_split_across_chunks() -> None:
    _, text, _ = _run([("vague prompt, ask", None), ("</think>", None), ("Which framework?", None)])
    assert "</think>" not in text
    assert text.endswith("Which framework?")


@pytest.mark.parametrize(
    "chunks",
    [
        ["<thi", "nk>a</th", "ink>b"],
        ["<", "t", "h", "i", "n", "k", ">", "a", "<", "/", "think", ">", "b"],
        ["<think>a<", "/think>b"],
    ],
)
def test_tags_split_across_chunk_boundaries(chunks: list[str]) -> None:
    thinking, text, _ = _run([(c, None) for c in chunks])
    assert (thinking, text) == ("a", "b")


def test_whitespace_after_close_tag_is_dropped() -> None:
    thinking, text, _ = _run([("<think>\nplan\n</think>\n\n", None), ("Answer", None)])
    assert thinking == "\nplan\n"
    assert text == "Answer"


def test_plain_text_passes_through_unchanged() -> None:
    thinking, text, _ = _run([("Hello ", None), ("world", None)])
    assert (thinking, text) == ("", "Hello world")


def test_lone_angle_bracket_is_held_then_flushed() -> None:
    splitter = ReasoningSplitter()
    assert splitter.feed("x <") == [Part("text", "x ")]
    assert splitter.feed(" y") == [Part("text", "< y")]
    assert splitter.feed("a <") == [Part("text", "a ")]
    assert splitter.flush() == [Part("text", "<")]


def test_feed_merges_adjacent_parts_of_one_kind() -> None:
    splitter = ReasoningSplitter()
    assert splitter.feed("<think>a", "native ") == [Part("thinking", "native a")]


def test_one_shot_helper_matches_streaming() -> None:
    result = split_reasoning("<think>a</think>\n\nb")
    assert (result.thinking, result.text) == ("a", "b")


def test_one_shot_helper_handles_native_reasoning_and_none() -> None:
    assert split_reasoning(None, "r").thinking == "r"
    assert split_reasoning(None).text == ""
