"""Server-sent event framing shared by the streaming routes."""

import asyncio
from collections.abc import AsyncIterator

# Cloudflare drops proxied connections that stay silent for 100s (not
# configurable below Enterprise). The thinking phase of a reasoning model
# produces no enhance deltas, so long generations 524'd. SSE comment lines
# every KEEPALIVE_SECONDS keep bytes flowing; spec-compliant and line-based
# ("data: "-prefixed) parsers both ignore them.
KEEPALIVE_SECONDS = 20.0
_KEEPALIVE = ": keepalive\n\n"


def sse_event(data: str) -> str:
    return f"data: {data}\n\n"


async def with_keepalive(agen: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield from an async generator, emitting SSE comments during stalls."""
    next_task = None
    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(anext(agen))
            done, _ = await asyncio.wait({next_task}, timeout=KEEPALIVE_SECONDS)
            if not done:
                yield _KEEPALIVE
                continue
            next_task = None
            try:
                yield done.pop().result()
            except StopAsyncIteration:
                return
    finally:
        if next_task is not None:
            next_task.cancel()
