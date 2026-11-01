"""Session endpoints (PRD scenarios 3-7, 10-15 and 17, route side).

Runs the real routes against fakeredis with acompletion mocked.  Callers are
told apart by an X-Test-Key header standing in for get_current_api_key, so
ownership is exercised without a database.  No paid calls.
"""

import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import litellm
import pytest
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, TypeAdapter

from koleslaw_api.config import Settings, get_settings
from koleslaw_api.prompts.system import get_system_prompt
from koleslaw_api.routes import sessions
from koleslaw_api.schemas.session import SessionStreamEvent
from koleslaw_api.services.auth import get_current_api_key
from koleslaw_api.services.cache import PromptCache
from koleslaw_api.services.llm import LLMService
from koleslaw_api.services.session_store import SessionError, SessionStore
from koleslaw_api.services.turn_protocol import trailer

LOCAL = "ollama_chat/prompt-enhance-14b-think"
FALLBACK = "anthropic/claude-haiku-4-5"
KEY_A = "11111111-1111-1111-1111-111111111111"
KEY_B = "22222222-2222-2222-2222-222222222222"
# The portal playground key; its sha256 is the default anon_key_hashes entry.
PUBLIC_KEY = "pk_live_fNuYW07Bf16zLFLbghxSPERYV4QS1SWF4-OuLJ5JcCE"
PROMPT = "write a poem"


def _settings(**overrides) -> Settings:
    values = {
        "database_url": "postgresql+asyncpg://localhost:5432/test",
        "redis_url": "redis://localhost:6379/0",
        "api_key_hash_secret": SecretStr("test-secret"),
        "session_secret": SecretStr("test-session-secret"),
        "environment": "development",
        "default_model": LOCAL,
        "fallback_model": FALLBACK,
        "sessions_enabled": True,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return _settings()


@pytest.fixture
def redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis()


def _test_key(request: Request) -> uuid.UUID | None:
    raw = request.headers.get("x-test-key")
    return uuid.UUID(raw) if raw else None


@pytest.fixture
def app(settings: Settings, redis: fakeredis.FakeAsyncRedis, mock_db_session_factory: MagicMock) -> FastAPI:
    @asynccontextmanager
    async def noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(lifespan=noop_lifespan)
    app.state.settings = settings
    app.state.redis = redis
    app.state.db_session_factory = mock_db_session_factory
    app.state.llm_service = LLMService(settings)
    app.state.cache = PromptCache(redis)
    app.state.session_store = SessionStore(redis, ttl_seconds=settings.session_ttl_seconds)
    app.include_router(sessions.router)
    app.dependency_overrides[get_current_api_key] = _test_key
    app.dependency_overrides[get_settings] = lambda: settings

    @app.exception_handler(SessionError)
    async def handler(_request: Request, exc: Exception):
        return sessions.session_error_response(exc)

    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# --- helpers -------------------------------------------------------------------


def _chunk(content: str = "", reasoning: str | None = None, finish: str | None = None) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(delta=MagicMock(content=content, reasoning_content=reasoning), finish_reason=finish)]
    )


async def _stream(*chunks: MagicMock):
    for chunk in chunks:
        yield chunk


def _answer(text: str, reasoning: str | None = None):
    chunks = [_chunk(reasoning=reasoning)] if reasoning else []
    return _stream(*chunks, _chunk(text), _chunk(finish="stop"))


async def _broken_stream():
    yield _chunk("Partial ")
    raise litellm.APIConnectionError(message="connection reset", llm_provider="ollama", model=LOCAL)


def _conn_error() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(message="connection refused", llm_provider="ollama", model=LOCAL)


def _events(body: str) -> list[dict]:
    """Every data event, validated against the schema union, [DONE] excluded."""
    adapter = TypeAdapter(SessionStreamEvent)
    events = []
    for frame in body.strip().split("\n\n"):
        if frame.startswith("data: ") and frame != "data: [DONE]":
            event = json.loads(frame[6:])
            adapter.validate_python(event)
            events.append(event)
    return events


def _text(events: list[dict]) -> str:
    return "".join(e["delta"] for e in events if e["type"] == "text")


def _turn(events: list[dict]) -> dict:
    turns = [e for e in events if e["type"] == "turn"]
    assert len(turns) == 1
    assert events[-1] is turns[0]
    return turns[0]


def _headers(session_id: str | None = None, key: str | None = None) -> dict[str, str]:
    headers = {}
    if session_id is not None:
        headers["X-Session-Id"] = session_id
    if key is not None:
        headers["X-Test-Key"] = key
    return headers


async def _start(client: AsyncClient, answer, key: str | None = None, **body) -> tuple[str, list[dict], str]:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = answer
        response = await client.post("/v1/sessions", json={"prompt": PROMPT, **body}, headers=_headers(key=key))
    assert response.status_code == 200, response.text
    events = _events(response.text)
    return events[0]["session_id"], events, response.text


async def _reply(client: AsyncClient, session_id: str, answer, key: str | None = None, **body):
    body = body or {"message": "Python"}
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        if isinstance(answer, list):
            mock_llm.side_effect = answer
        else:
            mock_llm.return_value = answer
        response = await client.post("/v1/sessions/turns", json=body, headers=_headers(session_id, key))
    return response, mock_llm


async def _stored(redis: fakeredis.FakeAsyncRedis, session_id: str) -> dict:
    raw = await redis.get(f"sess:{hashlib.sha256(session_id.encode()).hexdigest()}")
    assert raw is not None
    return json.loads(raw)


# --- start a session, then answer (scenarios 3 and 4) ----------------------------


async def test_start_streams_session_event_first_then_the_turn(client: AsyncClient) -> None:
    session_id, events, body = await _start(client, _answer("[[ASK]]\nWhich stack?"))

    assert body.startswith(": connected\n\n")
    assert body.endswith("data: [DONE]\n\n")
    assert "[[ASK]]" not in body
    assert events[0] == {"type": "session", "session_id": session_id, "expires_in": 1800, "turns_left": 3}
    assert len(session_id) >= 22
    assert _text(events) == "Which stack?"
    assert _turn(events) == {"type": "turn", "kind": "question", "turn": 1, "turns_left": 3, "model_used": LOCAL}


async def test_session_event_comes_only_on_start(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    response, _ = await _reply(client, session_id, _answer("Enhanced prompt"))

    assert all(e["type"] != "session" for e in _events(response.text))


async def test_anon_question_then_answer_sends_the_transcript(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    response, mock_llm = await _reply(client, session_id, _answer("Enhanced prompt"))

    assert response.status_code == 200
    events = _events(response.text)
    assert _text(events) == "Enhanced prompt"
    assert _turn(events) == {"type": "turn", "kind": "prompt", "turn": 2, "turns_left": 2, "model_used": LOCAL}
    # Ollama turns carry no system message; the trailer rides on the latest
    # user turn only, and the model sees its own marker on the question.
    assert mock_llm.call_args.kwargs["messages"] == [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "[[ASK]]\nWhich stack?"},
        {"role": "user", "content": f"Python\n\n{trailer('reply')}"},
    ]


async def test_first_turn_carries_the_trailer_and_the_context(client: AsyncClient) -> None:
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _answer("Enhanced prompt")
        await client.post("/v1/sessions", json={"prompt": PROMPT, "context": "for kids"})

    (message,) = mock_llm.call_args.kwargs["messages"]
    assert message["role"] == "user"
    assert message["content"].startswith(PROMPT)
    assert "for kids" in message["content"]
    assert message["content"].endswith(trailer("reply"))
    assert mock_llm.call_args.kwargs["stream"] is True


async def test_signed_in_session_belongs_to_the_key(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"), key=KEY_A)
    response, _ = await _reply(client, session_id, _answer("Enhanced prompt"), key=KEY_A)

    assert response.status_code == 200
    stored = await _stored(redis, session_id)
    assert stored["owner"] == KEY_A
    assert stored["user_turns"] == 2
    assert [t["kind"] for t in stored["turns"]] == [None, "question", None, "prompt"]


async def test_thinking_streams_as_thinking_events(client: AsyncClient) -> None:
    _, events, _ = await _start(client, _answer("Enhanced prompt", reasoning="let me think"))

    assert [e["delta"] for e in events if e["type"] == "thinking"] == ["let me think"]
    assert _text(events) == "Enhanced prompt"


# --- finalize (scenario 5) -------------------------------------------------------


async def test_finalize_forbids_questions_and_a_marker_is_still_a_prompt(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    response, mock_llm = await _reply(client, session_id, _answer("[[ASK]] Enhanced prompt"), action="finalize")

    events = _events(response.text)
    assert _turn(events)["kind"] == "prompt"
    assert _text(events) == "Enhanced prompt"
    assert mock_llm.call_args.kwargs["messages"][-1] == {"role": "user", "content": trailer("finalize")}


async def test_finalize_may_carry_a_message(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    _, mock_llm = await _reply(client, session_id, _answer("Enhanced"), action="finalize", message="Python")

    assert mock_llm.call_args.kwargs["messages"][-1]["content"] == f"Python\n\n{trailer('finalize')}"


# --- caps (scenarios 6 and 7) ----------------------------------------------------


async def test_fourth_turn_is_forced_to_finalize_and_a_fifth_is_full(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    for turn, left in ((2, 2), (3, 1)):
        response, _ = await _reply(client, session_id, _answer("[[ASK]]\nMore?"))
        assert _turn(_events(response.text)) == {
            "type": "turn",
            "kind": "question",
            "turn": turn,
            "turns_left": left,
            "model_used": LOCAL,
        }

    response, mock_llm = await _reply(client, session_id, _answer("[[ASK]]\nStill asking"))
    turn = _turn(_events(response.text))
    assert turn["kind"] == "prompt"
    assert turn["turn"] == 4
    assert turn["turns_left"] == 0
    assert mock_llm.call_args.kwargs["messages"][-1]["content"].endswith(trailer("finalize"))

    response, mock_llm = await _reply(client, session_id, _answer("never"))
    assert response.status_code == 409
    assert response.json() == {"error": "session full", "code": "session_full", "detail": None}
    mock_llm.assert_not_called()


async def test_turn_over_the_token_cap_is_full(app: FastAPI, client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    app.state.settings = _settings(session_max_input_tokens=200)

    response, mock_llm = await _reply(client, session_id, _answer("never"), message="word " * 400)

    assert response.status_code == 409
    assert response.json()["code"] == "session_full"
    mock_llm.assert_not_called()


async def test_start_over_the_token_cap_is_full(app: FastAPI, client: AsyncClient) -> None:
    app.state.settings = _settings(session_max_input_tokens=200)
    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        response = await client.post("/v1/sessions", json={"prompt": "word " * 400})

    assert response.status_code == 409
    assert response.json()["code"] == "session_full"
    mock_llm.assert_not_called()


async def test_reply_over_the_length_cap_is_422(app: FastAPI, client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    app.state.settings = _settings(session_max_reply_chars=20)

    response, mock_llm = await _reply(client, session_id, _answer("never"), message="x" * 21)

    assert response.status_code == 422
    mock_llm.assert_not_called()


async def test_input_tokens_estimate_is_stored(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))

    assert (await _stored(redis, session_id))["input_tokens"] > 0


# --- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [{}, {"prompt": ""}, {"prompt": "x" * 50001}, {"prompt": "hi", "enhancement_type": "rewrite"}],
)
async def test_bad_start_body_is_422(client: AsyncClient, body: dict) -> None:
    response = await client.post("/v1/sessions", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [{}, {"action": "reply"}, {"message": ""}, {"message": "x" * 10001}, {"action": "stop", "message": "hi"}],
)
async def test_bad_turn_body_is_422(client: AsyncClient, body: dict) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    response = await client.post("/v1/sessions/turns", json=body, headers=_headers(session_id))
    assert response.status_code == 422


async def test_turn_without_a_session_header_is_422(client: AsyncClient) -> None:
    response = await client.post("/v1/sessions/turns", json={"message": "Python"})
    assert response.status_code == 422


# --- delete, expiry, ownership, concurrency (scenarios 8, 10, 11, 12) -------------


async def test_delete_is_204_and_idempotent_and_the_id_is_dead(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))

    for _ in range(2):
        response = await client.delete("/v1/sessions", headers=_headers(session_id))
        assert response.status_code == 204
        assert response.content == b""

    response, mock_llm = await _reply(client, session_id, _answer("never"))
    assert response.status_code == 404
    assert response.json() == {"error": "session not found", "code": "session_not_found", "detail": None}
    mock_llm.assert_not_called()


async def test_delete_of_an_unknown_id_is_204(client: AsyncClient) -> None:
    response = await client.delete("/v1/sessions", headers=_headers("no-such-session"))
    assert response.status_code == 204


async def test_expired_session_is_404(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    await redis.delete(f"sess:{hashlib.sha256(session_id.encode()).hexdigest()}")

    response, _ = await _reply(client, session_id, _answer("never"))
    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"


async def test_wrong_owner_gets_404_and_cannot_delete(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"), key=KEY_A)
    before = await _stored(redis, session_id)

    response, mock_llm = await _reply(client, session_id, _answer("never"), key=KEY_B)
    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"
    mock_llm.assert_not_called()

    response = await client.delete("/v1/sessions", headers=_headers(session_id, KEY_B))
    assert response.status_code == 204
    assert await _stored(redis, session_id) == before


async def test_concurrent_turn_is_busy(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    await redis.set(f"sess:{hashlib.sha256(session_id.encode()).hexdigest()}:lock", "other", ex=120)

    response, mock_llm = await _reply(client, session_id, _answer("never"))
    assert response.status_code == 409
    assert response.json() == {"error": "session busy", "code": "session_busy", "detail": None}
    mock_llm.assert_not_called()


async def test_the_lock_is_released_after_each_turn(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    lock = f"sess:{hashlib.sha256(session_id.encode()).hexdigest()}:lock"
    assert await redis.get(lock) is None

    await _reply(client, session_id, _answer("Enhanced prompt"))
    assert await redis.get(lock) is None


# --- fallback and stream errors (scenario 13) ------------------------------------


async def test_fallback_before_first_byte_serves_the_turn_with_the_cloud_prompt(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"), context="for kids")
    response, mock_llm = await _reply(client, session_id, [_conn_error(), _answer("Enhanced prompt")])

    events = _events(response.text)
    assert _turn(events)["model_used"] == FALLBACK
    assert _text(events) == "Enhanced prompt"
    retry = mock_llm.call_args_list[1].kwargs
    assert retry["model"] == FALLBACK
    assert retry["messages"] == [
        {"role": "system", "content": get_system_prompt("expand", "for kids")},
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "[[ASK]]\nWhich stack?"},
        {"role": "user", "content": f"Python\n\n{trailer('reply')}"},
    ]


async def test_usage_is_logged_with_the_served_model(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"), key=KEY_A)
    with patch("koleslaw_api.routes.sessions.log_usage", new_callable=AsyncMock) as log_usage:
        await _reply(client, session_id, [_conn_error(), _answer("Enhanced prompt")], key=KEY_A)

    log_usage.assert_awaited_once()
    args, kwargs = log_usage.call_args
    assert args[1] == uuid.UUID(KEY_A)
    assert args[2] == FALLBACK
    assert kwargs["cached"] is False
    assert kwargs["output_tokens"] > 0


async def test_mid_stream_failure_is_an_error_event_and_nothing_is_stored(
    client: AsyncClient, redis: fakeredis.FakeAsyncRedis
) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    before = await _stored(redis, session_id)

    response, mock_llm = await _reply(client, session_id, _broken_stream())

    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    events = _events(response.text)
    assert all(e["type"] != "turn" for e in events)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "connection_error"
    assert "connection reset" not in events[-1]["message"]
    assert mock_llm.call_count == 1
    assert await _stored(redis, session_id) == before

    # The lock was released: the user can resend at once.
    response, _ = await _reply(client, session_id, _answer("Enhanced prompt"))
    assert _turn(_events(response.text))["turn"] == 2


async def test_failure_with_no_fallback_left_is_an_error_event(app: FastAPI, client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    response, _ = await _reply(client, session_id, [_conn_error(), _conn_error()])

    events = _events(response.text)
    assert [e["type"] for e in events] == ["error"]
    assert response.text.endswith("data: [DONE]\n\n")


async def test_failed_first_turn_keeps_the_session_for_a_resend(
    client: AsyncClient, redis: fakeredis.FakeAsyncRedis
) -> None:
    session_id, events, _ = await _start(client, _broken_stream())

    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "error"
    assert (await _stored(redis, session_id))["turns"] == []

    response, mock_llm = await _reply(client, session_id, _answer("Enhanced prompt"), message=PROMPT)
    assert _turn(_events(response.text)) == {
        "type": "turn",
        "kind": "prompt",
        "turn": 1,
        "turns_left": 3,
        "model_used": LOCAL,
    }
    assert mock_llm.call_args.kwargs["messages"] == [{"role": "user", "content": f"{PROMPT}\n\n{trailer('reply')}"}]


# --- empty turn (scenario 14) ----------------------------------------------------


async def test_reasoning_only_turn_is_empty_and_not_stored(client: AsyncClient, redis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    before = await _stored(redis, session_id)

    response, _ = await _reply(client, session_id, _stream(_chunk(reasoning="hmm"), _chunk(finish="stop")))

    events = _events(response.text)
    assert [e["type"] for e in events] == ["thinking", "turn"]
    assert _turn(events) == {"type": "turn", "kind": "empty", "turn": 2, "turns_left": 3, "model_used": LOCAL}
    assert (await _stored(redis, session_id))["turns"] == before["turns"]
    assert (await _stored(redis, session_id))["user_turns"] == 1


async def test_finalize_with_nothing_to_finalize_is_422(client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _stream(_chunk(reasoning="hmm"), _chunk(finish="stop")))

    response, mock_llm = await _reply(client, session_id, _answer("never"), action="finalize")
    assert response.status_code == 422
    mock_llm.assert_not_called()


# --- quota, rate limit, cache (scenario 15) ------------------------------------


async def test_daily_quota_is_counted_once_per_session(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    counter = f"anon_daily:{datetime.now(UTC).strftime('%Y-%m-%d')}:203.0.113.7"
    await redis.set(counter, 9)
    anon = {"Authorization": f"Bearer {PUBLIC_KEY}", "CF-Connecting-IP": "203.0.113.7"}

    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _answer("[[ASK]]\nWhich stack?")
        response = await client.post("/v1/sessions", json={"prompt": PROMPT}, headers=anon)
    session_id = _events(response.text)[0]["session_id"]

    for _ in range(3):
        with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _answer("[[ASK]]\nMore?")
            response = await client.post(
                "/v1/sessions/turns", json={"message": "Python"}, headers={**anon, "X-Session-Id": session_id}
            )
        assert response.status_code == 200
        assert _turn(_events(response.text))["kind"] in {"question", "prompt"}

    assert int(await redis.get(counter)) == 10

    response = await client.post("/v1/sessions", json={"prompt": PROMPT}, headers=anon)
    assert response.status_code == 429


async def test_every_turn_passes_the_rate_limit(app: FastAPI, client: AsyncClient) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))

    def limited() -> None:
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"})

    app.dependency_overrides[get_current_api_key] = limited
    response, mock_llm = await _reply(client, session_id, _answer("never"))

    assert response.status_code == 429
    mock_llm.assert_not_called()


async def test_sessions_never_touch_the_prompt_cache(client: AsyncClient, redis: fakeredis.FakeAsyncRedis) -> None:
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    await _reply(client, session_id, _answer("Enhanced prompt"))

    assert await redis.keys("prompt_cache:*") == []


async def test_the_session_id_is_never_logged(client: AsyncClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    session_id, _, _ = await _start(client, _answer("[[ASK]]\nWhich stack?"))
    await _reply(client, session_id, [_conn_error(), _answer("Enhanced prompt")])
    await _reply(client, session_id, _broken_stream())
    await client.delete("/v1/sessions", headers=_headers(session_id))

    assert session_id not in caplog.text


# --- kill switch (scenario 17) ---------------------------------------------------


async def test_kill_switch_returns_503_before_quota(app: FastAPI, client: AsyncClient, redis) -> None:
    app.state.settings = _settings(sessions_enabled=False)
    anon = {"Authorization": f"Bearer {PUBLIC_KEY}", "CF-Connecting-IP": "203.0.113.7"}

    with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
        responses = [
            await client.post("/v1/sessions", json={"prompt": PROMPT}, headers=anon),
            await client.post("/v1/sessions/turns", json={"message": "hi"}, headers={"X-Session-Id": "x"}),
            await client.delete("/v1/sessions", headers={"X-Session-Id": "x"}),
        ]

    for response in responses:
        assert response.status_code == 503
        assert response.json() == {"error": "sessions disabled", "code": "sessions_disabled", "detail": None}
    mock_llm.assert_not_called()
    assert await redis.keys("anon_daily:*") == []
