"""Server-side state for model sessions, in Redis.

The session id is a 128-bit capability token.  Only its sha256 is used, as
the key ``sess:{hash}``; the raw id is never stored or logged.  A session
belongs to the ``api_key_id`` that created it (D2: no IP binding, so every
anonymous visitor shares the public key's id and the token is the guard).
A wrong owner reads exactly like a missing session.

The TTL slides: set on create, reset on every accepted turn, and nothing
else expires a session.  A turn holds ``sess:{hash}:lock`` (SET NX EX) while
it streams.  There is no in-memory fallback: if Redis is down, callers fail.
"""

import hashlib
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Literal

import anyio
from redis.asyncio import Redis

from koleslaw_api.services.turn_protocol import TurnKind

Role = Literal["user", "assistant"]


class SessionError(Exception):
    """Base for session errors.  Messages never carry the session id."""


class SessionNotFoundError(SessionError):
    def __init__(self) -> None:
        super().__init__("session not found")


class SessionBusyError(SessionError):
    def __init__(self) -> None:
        super().__init__("session busy")


class SessionFullError(SessionError):
    def __init__(self) -> None:
        super().__init__("session full")


@dataclass(frozen=True, slots=True)
class Turn:
    role: Role
    content: str
    kind: TurnKind | None = None


@dataclass(slots=True)
class Session:
    owner: str
    turns: list[Turn] = field(default_factory=list)
    user_turns: int = 0
    input_tokens: int = 0
    context: str | None = None
    enhancement_type: str = "auto"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Session":
        data = json.loads(raw)
        data["turns"] = [Turn(**t) for t in data["turns"]]
        return cls(**data)


def new_session_id() -> str:
    return secrets.token_urlsafe(16)


def session_key(session_id: str) -> str:
    return f"sess:{hashlib.sha256(session_id.encode()).hexdigest()}"


class SessionStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 1800, lock_seconds: int = 120) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._lock_ttl = lock_seconds

    async def create(
        self, owner: str, *, context: str | None = None, enhancement_type: str = "auto"
    ) -> tuple[str, Session]:
        session_id = new_session_id()
        session = Session(owner=owner, context=context, enhancement_type=enhancement_type)
        await self._redis.set(session_key(session_id), session.to_json(), ex=self._ttl)
        return session_id, session

    async def get(self, session_id: str, owner: str) -> Session:
        raw = await self._redis.get(session_key(session_id))
        if raw is None:
            raise SessionNotFoundError()
        session = Session.from_json(raw)
        if session.owner != owner:
            raise SessionNotFoundError()
        return session

    async def append(self, session_id: str, owner: str, turns: list[Turn], *, input_tokens: int) -> Session:
        """Store an accepted turn, set the token estimate, reset the TTL."""
        session = await self.get(session_id, owner)
        session.turns.extend(turns)
        session.user_turns += sum(1 for t in turns if t.role == "user")
        session.input_tokens = input_tokens
        # xx: a session deleted or expired mid-turn stays gone.
        stored = await self._redis.set(session_key(session_id), session.to_json(), ex=self._ttl, xx=True)
        if not stored:
            raise SessionNotFoundError()
        return session

    async def delete(self, session_id: str, owner: str) -> None:
        """Idempotent.  Missing or someone else's session is a silent no-op."""
        try:
            await self.get(session_id, owner)
        except SessionNotFoundError:
            return
        key = session_key(session_id)
        await self._redis.delete(key, f"{key}:lock")

    @asynccontextmanager
    async def lock(self, session_id: str) -> AsyncIterator[None]:
        """Hold the turn lock.  Raises ``SessionBusyError`` if a turn is running."""
        lock_key = f"{session_key(session_id)}:lock"
        token = secrets.token_hex(8)
        if not await self._redis.set(lock_key, token, nx=True, ex=self._lock_ttl):
            raise SessionBusyError()
        try:
            yield
        finally:
            # Shielded: a client abort cancels the stream's task group, and
            # without the shield these awaits are cancelled too, leaving the
            # lock held until it expires.
            with anyio.CancelScope(shield=True):
                held = await self._redis.get(lock_key)
                if held is not None and (held.decode() if isinstance(held, bytes) else held) == token:
                    await self._redis.delete(lock_key)
