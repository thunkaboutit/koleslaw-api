"""The Redis session store (PRD scenarios 8, 10, 11 and 12, store side).

Runs against fakeredis, which models TTL and SET NX.  The client is left at
the app default (bytes, no decode_responses) to match prod.
"""

import asyncio
import hashlib
import json

import fakeredis
import pytest

from koleslaw_api.services import session_store as ss

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis()


@pytest.fixture
def store(redis: fakeredis.FakeAsyncRedis):
    return ss.SessionStore(redis, ttl_seconds=1800, lock_seconds=120)


def _key(session_id: str) -> str:
    return f"sess:{hashlib.sha256(session_id.encode()).hexdigest()}"


def _turns() -> list:
    return [
        ss.Turn(role="user", content="write a poem"),
        ss.Turn(role="assistant", content="About what?", kind="question"),
    ]


# --- create and get ----------------------------------------------------------


async def test_create_returns_a_128_bit_url_safe_id(store) -> None:
    session_id, _ = await store.create(OWNER)
    assert len(session_id) == 22
    assert all(c.isalnum() or c in "-_" for c in session_id)


async def test_create_ids_are_unique(store) -> None:
    ids = {(await store.create(OWNER))[0] for _ in range(50)}
    assert len(ids) == 50


async def test_key_is_the_sha256_of_the_id_and_the_raw_id_is_never_stored(store, redis) -> None:
    session_id, _ = await store.create(OWNER, context="ctx", enhancement_type="expand")
    keys = [k.decode() for k in await redis.keys("*")]
    assert keys == [_key(session_id)]
    assert ss.session_key(session_id) == _key(session_id)
    raw = await redis.get(_key(session_id))
    assert session_id.encode() not in raw


async def test_create_stores_the_session_fields(store) -> None:
    session_id, created = await store.create(OWNER, context="ctx", enhancement_type="expand")
    session = await store.get(session_id, OWNER)
    assert session == created
    assert session.owner == OWNER
    assert session.turns == []
    assert session.user_turns == 0
    assert session.input_tokens == 0
    assert session.context == "ctx"
    assert session.enhancement_type == "expand"


async def test_create_defaults(store) -> None:
    session_id, _ = await store.create(OWNER)
    session = await store.get(session_id, OWNER)
    assert (session.context, session.enhancement_type) == (None, "auto")


async def test_stored_value_is_json_with_the_prd_fields(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    await store.append(session_id, OWNER, _turns(), input_tokens=12)
    data = json.loads(await redis.get(_key(session_id)))
    assert set(data) == {"owner", "turns", "user_turns", "input_tokens", "context", "enhancement_type"}
    assert data["turns"] == [
        {"role": "user", "content": "write a poem", "kind": None},
        {"role": "assistant", "content": "About what?", "kind": "question"},
    ]


async def test_create_sets_the_ttl(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    assert 1790 <= await redis.ttl(_key(session_id)) <= 1800


async def test_get_unknown_id_is_not_found(store) -> None:
    with pytest.raises(ss.SessionNotFoundError):
        await store.get("no-such-session", OWNER)


async def test_get_does_not_extend_the_ttl(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    await redis.expire(_key(session_id), 10)
    await store.get(session_id, OWNER)
    assert await redis.ttl(_key(session_id)) <= 10


# --- ownership (D2: api_key_id only) -------------------------------------------


async def test_wrong_owner_is_indistinguishable_from_not_found(store) -> None:
    session_id, _ = await store.create(OWNER)
    with pytest.raises(ss.SessionNotFoundError) as wrong:
        await store.get(session_id, OTHER)
    with pytest.raises(ss.SessionNotFoundError) as missing:
        await store.get("no-such-session", OTHER)
    assert type(wrong.value) is type(missing.value)
    assert str(wrong.value) == str(missing.value)


async def test_wrong_owner_cannot_append_and_session_is_unchanged(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    before = await redis.get(_key(session_id))
    with pytest.raises(ss.SessionNotFoundError):
        await store.append(session_id, OTHER, _turns(), input_tokens=12)
    assert await redis.get(_key(session_id)) == before


async def test_wrong_owner_delete_is_a_silent_no_op(store) -> None:
    session_id, created = await store.create(OWNER)
    await store.delete(session_id, OTHER)
    assert await store.get(session_id, OWNER) == created


# --- append and sliding ttl ----------------------------------------------------


async def test_append_adds_turns_and_counts_user_turns(store) -> None:
    session_id, _ = await store.create(OWNER)
    await store.append(session_id, OWNER, _turns(), input_tokens=12)
    session = await store.append(
        session_id,
        OWNER,
        [ss.Turn(role="user", content="haiku"), ss.Turn(role="assistant", content="A haiku.", kind="prompt")],
        input_tokens=30,
    )
    assert session == await store.get(session_id, OWNER)
    assert [(t.role, t.content, t.kind) for t in session.turns] == [
        ("user", "write a poem", None),
        ("assistant", "About what?", "question"),
        ("user", "haiku", None),
        ("assistant", "A haiku.", "prompt"),
    ]
    assert session.user_turns == 2
    assert session.input_tokens == 30


async def test_append_resets_the_ttl(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    await redis.expire(_key(session_id), 10)
    await store.append(session_id, OWNER, _turns(), input_tokens=12)
    assert 1790 <= await redis.ttl(_key(session_id)) <= 1800


async def test_append_after_delete_does_not_recreate_the_session(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    await store.delete(session_id, OWNER)
    with pytest.raises(ss.SessionNotFoundError):
        await store.append(session_id, OWNER, _turns(), input_tokens=12)
    assert await redis.exists(_key(session_id)) == 0


# --- delete and expiry ---------------------------------------------------------


async def test_delete_destroys_the_session(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    await store.delete(session_id, OWNER)
    assert await redis.exists(_key(session_id)) == 0
    with pytest.raises(ss.SessionNotFoundError):
        await store.get(session_id, OWNER)


async def test_delete_unknown_id_is_a_silent_no_op(store) -> None:
    await store.delete("no-such-session", OWNER)


async def test_expired_session_is_not_found(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    await redis.pexpire(_key(session_id), 1)
    await asyncio.sleep(0.01)
    with pytest.raises(ss.SessionNotFoundError):
        await store.get(session_id, OWNER)


async def test_ttl_comes_from_the_store_setting(redis) -> None:
    store = ss.SessionStore(redis, ttl_seconds=60)
    session_id, _ = await store.create(OWNER)
    assert 50 <= await redis.ttl(_key(session_id)) <= 60


# --- turn lock -----------------------------------------------------------------


async def test_lock_key_is_set_nx_with_a_120s_expiry(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    async with store.lock(session_id):
        assert 110 <= await redis.ttl(_key(session_id) + ":lock") <= 120
    assert await redis.exists(_key(session_id) + ":lock") == 0


async def test_second_concurrent_turn_is_busy(store) -> None:
    session_id, _ = await store.create(OWNER)
    async with store.lock(session_id):
        with pytest.raises(ss.SessionBusyError):
            async with store.lock(session_id):
                pass


async def test_lock_is_released_when_the_turn_fails(store) -> None:
    session_id, _ = await store.create(OWNER)
    with pytest.raises(RuntimeError):
        async with store.lock(session_id):
            raise RuntimeError("stream broke")
    async with store.lock(session_id):
        pass


async def test_releasing_an_expired_lock_leaves_a_new_holder_alone(store, redis) -> None:
    session_id, _ = await store.create(OWNER)
    lock_key = _key(session_id) + ":lock"
    async with store.lock(session_id):
        await redis.set(lock_key, "someone-else", ex=120)
    assert await redis.get(lock_key) == b"someone-else"


async def test_errors_share_one_base_and_never_carry_the_raw_id(store) -> None:
    session_id, _ = await store.create(OWNER)
    assert issubclass(ss.SessionNotFoundError, ss.SessionError)
    assert issubclass(ss.SessionBusyError, ss.SessionError)
    assert issubclass(ss.SessionFullError, ss.SessionError)
    async with store.lock(session_id):
        with pytest.raises(ss.SessionBusyError) as busy:
            async with store.lock(session_id):
                pass
    with pytest.raises(ss.SessionNotFoundError) as missing:
        await store.get(session_id, OTHER)
    assert session_id not in str(busy.value)
    assert session_id not in str(missing.value)


def test_session_settings_defaults(settings) -> None:
    assert settings.sessions_enabled is False
    assert settings.session_ttl_seconds == 1800
    assert settings.session_max_turns == 4
    assert settings.session_max_input_tokens == 8000
    assert settings.session_max_reply_chars == 10000
