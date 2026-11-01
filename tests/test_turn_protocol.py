"""The ask-or-finalize turn protocol (PRD scenarios 3 and 5, protocol side).

The model marks a question by starting its reply with ``[[ASK]]``.  The
marker is stripped before any text reaches the client, so the streaming
filter must hold back only the leading text that could still be the marker.
"""

import pytest

from koleslaw_api.services import turn_protocol as tp


def _stream(deltas: list[str], mode: str = "reply") -> tuple[list[str], object]:
    """Feed deltas through a MarkerFilter, return (emitted pieces, result)."""
    f = tp.MarkerFilter(mode)
    out = [f.feed(d) for d in deltas]
    out.append(f.flush())
    return out, f.result()


# --- trailers --------------------------------------------------------------


def test_reply_trailer_is_the_prd_text() -> None:
    assert tp.trailer("reply") == (
        "If the prompt is too ambiguous to enhance well, start your reply with the line "
        "[[ASK]] and ask up to 3 short questions.  Otherwise return only the enhanced prompt."
    )


def test_finalize_trailer_forbids_questions() -> None:
    assert tp.trailer("finalize") == (
        "Do not ask questions.  Make reasonable assumptions and return only the enhanced prompt."
    )


def test_with_trailer_appends_after_a_blank_line() -> None:
    assert tp.with_trailer("write a poem", "finalize") == "write a poem\n\n" + tp.trailer("finalize")


# --- pure classification ---------------------------------------------------


def test_marker_reply_is_a_question_with_marker_stripped() -> None:
    result = tp.classify("[[ASK]]\nWhich stack?")
    assert (result.kind, result.text) == ("question", "Which stack?")


def test_leading_whitespace_before_marker_is_ignored() -> None:
    result = tp.classify("\n  [[ASK]] Which stack?")
    assert (result.kind, result.text) == ("question", "Which stack?")


def test_plain_reply_is_a_prompt_unchanged() -> None:
    result = tp.classify("Write a Python CLI that parses CSV.")
    assert (result.kind, result.text) == ("prompt", "Write a Python CLI that parses CSV.")


def test_prompt_ending_in_question_mark_is_still_a_prompt() -> None:
    result = tp.classify("You are a tutor.  What is the derivative of x^2?")
    assert result.kind == "prompt"


def test_finalize_mode_marker_reply_is_a_prompt_with_marker_stripped() -> None:
    result = tp.classify("[[ASK]]\nWhich stack?", "finalize")
    assert (result.kind, result.text) == ("prompt", "Which stack?")


@pytest.mark.parametrize("text", ["", "   \n", "[[ASK]]", "[[ASK]]\n  "])
def test_blank_or_marker_only_reply_is_empty(text: str) -> None:
    result = tp.classify(text)
    assert (result.kind, result.text) == ("empty", "")


# --- streaming filter ------------------------------------------------------


def test_stream_marker_split_across_deltas_never_reaches_the_client() -> None:
    out, result = _stream(["[[", "AS", "K]]", "\n", "Which ", "stack?"])
    assert "".join(out) == "Which stack?"
    assert not any("[" in piece or "]" in piece for piece in out)
    assert (result.kind, result.text) == ("question", "Which stack?")


def test_stream_holds_text_only_while_it_could_be_the_marker() -> None:
    f = tp.MarkerFilter("reply")
    assert f.feed("[[A") == ""
    assert f.feed("x") == "[[Ax"
    assert f.feed(" more") == " more"


def test_stream_plain_text_passes_through_from_the_first_delta() -> None:
    f = tp.MarkerFilter("reply")
    assert f.feed("Write ") == "Write "
    assert f.feed("a poem") == "a poem"
    assert f.flush() == ""
    assert (f.result().kind, f.result().text) == ("prompt", "Write a poem")


def test_stream_leading_whitespace_is_held_until_decided() -> None:
    out, result = _stream(["\n", " ", "[[ASK]]", "  ", "Which?"])
    assert "".join(out) == "Which?"
    assert result.kind == "question"


def test_stream_short_prefix_at_end_is_flushed_as_prompt() -> None:
    out, result = _stream(["[["])
    assert "".join(out) == "[["
    assert (result.kind, result.text) == ("prompt", "[[")


def test_stream_finalize_mode_strips_marker_and_says_prompt() -> None:
    out, result = _stream(["[[ASK]]", "\nWhich stack?"], "finalize")
    assert "".join(out) == "Which stack?"
    assert (result.kind, result.text) == ("prompt", "Which stack?")


@pytest.mark.parametrize("deltas", [[], ["  \n"], ["[[ASK]]"], ["[[ASK", "]]\n "]])
def test_stream_with_no_answer_text_is_empty(deltas: list[str]) -> None:
    out, result = _stream(deltas)
    assert "".join(out) == ""
    assert (result.kind, result.text) == ("empty", "")


@pytest.mark.parametrize(
    "text",
    [
        "[[ASK]]\nWhich stack?",
        "  [[ASK]] Two questions?\n1. a\n2. b",
        "Plain prompt.",
        "[[ASKED]] not a marker",
        "\n\nLeading blank prompt",
        "[[ASK]]",
        "",
        "Intro line.\n[[ASK]]\nWhich stack?",
        "Use it [[ASK]] here and [[ASK]]\nthere [[ASK]]",
        "[[ASK]] [[ASK]]\nTwice?",
        "[[[ASK]]\nx [[AS [[ASK]]K]]",
    ],
)
def test_stream_matches_classify_at_every_split(text: str) -> None:
    expected = tp.classify(text)
    for cut in range(len(text) + 1):
        out, result = _stream([text[:cut], text[cut:]])
        assert "".join(out) == expected.text
        assert result == expected


# --- marker anywhere in the text -------------------------------------------

def test_mid_text_marker_is_stripped_and_kind_is_prompt() -> None:
    result = tp.classify("Write a poem.\n[[ASK]]\nMake it short.")
    assert (result.kind, result.text) == ("prompt", "Write a poem.\nMake it short.")


def test_mid_text_marker_in_finalize_mode_is_stripped() -> None:
    result = tp.classify("Write a poem. [[ASK]]\nShort.", "finalize")
    assert (result.kind, result.text) == ("prompt", "Write a poem. Short.")


def test_every_marker_is_stripped() -> None:
    result = tp.classify("a [[ASK]]b [[ASK]]\nc [[ASK]]")
    assert (result.kind, result.text) == ("prompt", "a b c ")


def test_leading_marker_still_decides_question_when_others_follow() -> None:
    result = tp.classify("[[ASK]]\nWhich?\n[[ASK]]\nAnd?")
    assert (result.kind, result.text) == ("question", "Which?\nAnd?")


def test_stream_marker_split_mid_text_never_reaches_the_client() -> None:
    out, result = _stream(["Some text [[A", "SK]]\nmore"])
    assert "".join(out) == "Some text more"
    assert "[" not in "".join(out)
    assert (result.kind, result.text) == ("prompt", "Some text more")


def test_stream_holds_only_a_possible_marker_prefix() -> None:
    f = tp.MarkerFilter("reply")
    assert f.feed("ab [") == "ab "
    assert f.feed("[A") == ""
    assert f.feed("b") == "[[Ab"
    assert f.feed("[[") == ""
    assert f.flush() == "[["


def test_stream_matches_classify_at_every_split_with_marker_anywhere() -> None:
    text = "Intro [[ASK]]\nmid [[ASK]] end"
    expected = tp.classify(text)
    for cut in range(len(text) + 1):
        out, result = _stream([text[:cut], text[cut:]])
        assert "".join(out) == expected.text
        assert result == expected
    assert expected.text == "Intro mid  end"
