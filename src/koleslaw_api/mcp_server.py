"""Remote MCP endpoint — a second front door onto /v1/enhance/sync.

Claude Code (and any other MCP client) speaks streamable HTTP here, so the
plugin needs no local runtime: a URL and a bearer token are the whole client
side. Everything behind the tool boundary is the REST path's own —
`authenticate_llm_caller` for auth and caps, `enhance_once` for the work — so
this module only translates between MCP and that path.

Mounted stateless, with JSON responses: every call is independent, which keeps
any API task able to serve any request (no session pinned to one task behind
the ALB) and holds no connection open between calls.

The canonical URL carries a trailing slash — `https://api.koleslaw.ai/v1/mcp/`.
The bare form works only for clients that follow redirects: Starlette's router
answers it with a 307 to the slashed form, and a client that does not follow
that reads the redirect body as the response and fails on its content type.
Configure clients with the slash.
"""

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from koleslaw_api.schemas.enhance import EnhanceRequest, EnhanceResponse
from koleslaw_api.services.auth import authenticate_llm_caller
from koleslaw_api.services.enhance import enhance_once
from koleslaw_api.services.llm import LLMServiceError

logger = logging.getLogger(__name__)

# Under /v1 on purpose: the ALB routes /auth/* and /v1/* to this service and
# everything else to the web UI, so a top-level /mcp would need a new listener
# rule and a terraform apply to be reachable at all.
MCP_PATH = "/v1/mcp"
TOOL_NAME = "enhance_prompt"

EnhancementType = Literal["auto", "expand", "refine", "structure"]

# The taxonomy the plugin's skill documents. Keep these four strings stable:
# the skill tells the model what to do with each one.
AUTH = "auth"
INVALID_REQUEST = "invalid_request"
RATE_LIMIT = "rate_limit"
BACKEND_UNAVAILABLE = "backend_unavailable"

_STATUS_CODES = {
    401: AUTH,
    403: AUTH,
    400: INVALID_REQUEST,
    404: INVALID_REQUEST,
    422: INVALID_REQUEST,
    429: RATE_LIMIT,
}

TOOL_DESCRIPTION = """\
Rewrite a terse or vague prompt into a clearer, more specific one.

Returns the enhanced prompt for the user to review — it is a suggestion, not a
replacement for what they asked. Send the user's own wording as `prompt`; use
`context` only for a short, relevant excerpt (a CLAUDE.md rule, a function
signature), never a whole file. Skip this tool for prompts that are already
well-specified.\
"""


class _ParentAppScope:
    """Put the parent app back in the ASGI scope.

    Starlette's mount hands the sub-app the scope, and the sub-app stamps
    itself as `scope["app"]` — so `request.app` inside a tool would be the MCP
    app, not ours. The auth chain reads settings, redis, and the session
    factory off `app.state`, so restore the real owner of the request. This
    runs as middleware *inside* the sub-app, which is the only place that sees
    the scope after the sub-app has stamped it.
    """

    def __init__(self, app: ASGIApp, parent: FastAPI) -> None:
        self._app = app
        self._parent = parent

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope["app"] = self._parent
        await self._app(scope, receive, send)


def _tool_error(code: str, message: str) -> ToolError:
    """An anticipated failure, tagged with the taxonomy the skill documents."""
    return ToolError(f"{code}: {message}")


def _build_request(prompt: str, context: str | None, enhancement_type: EnhancementType) -> EnhanceRequest:
    """Validate tool arguments through the REST body model.

    Reusing EnhanceRequest means the prompt and context length caps (T2-E) are
    the same bounds on both doors, and `model` is left unset — the router
    picks, and callers cannot steer it (T1-A).
    """
    try:
        return EnhanceRequest(prompt=prompt, context=context, enhancement_type=enhancement_type, stream=False)
    except ValidationError as exc:
        raise _tool_error(INVALID_REQUEST, _first_validation_message(exc)) from exc


def _first_validation_message(exc: ValidationError) -> str:
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"]) or "arguments"
    return f"{field}: {error['msg']}"


def create_mcp_server(app: FastAPI) -> MCPServer:
    """Build the MCP server, reading collaborators off `app.state` at call time.

    `app` is captured rather than its state: main.py populates state in the
    lifespan, well after the mount.
    """
    mcp = MCPServer(name="koleslaw", version=app.version)

    @mcp.tool(name=TOOL_NAME, description=TOOL_DESCRIPTION)
    async def enhance_prompt(
        prompt: str,
        ctx: Context,
        context: str | None = None,
        enhancement_type: EnhancementType = "auto",
    ) -> EnhanceResponse:
        request = ctx.request_context.request
        if request is None:  # pragma: no cover — only reachable off an HTTP transport
            raise _tool_error(BACKEND_UNAVAILABLE, "The MCP endpoint is only available over HTTP.")

        try:
            # Auth first: an unauthenticated caller gets no further, not even
            # as far as learning the argument bounds.
            api_key_id = await authenticate_llm_caller(request)
            body = _build_request(prompt, context, enhancement_type)
            return await enhance_once(
                body,
                api_key_id,
                llm=app.state.llm_service,
                cache=app.state.cache,
                settings=app.state.settings,
                db_session_factory=app.state.db_session_factory,
            )
        except HTTPException as exc:
            code = _STATUS_CODES.get(exc.status_code, BACKEND_UNAVAILABLE)
            raise _tool_error(code, str(exc.detail)) from exc
        except LLMServiceError as exc:
            raise _tool_error(BACKEND_UNAVAILABLE, str(exc)) from exc
        except ToolError:
            raise
        except Exception as exc:
            # Redis or Postgres unreachable, or anything else unforeseen. The
            # caller still gets a code it can act on, but the text is fixed —
            # the detail belongs in the logs, not in a tool result (T3-I).
            logger.exception("MCP %s failed", TOOL_NAME)
            raise _tool_error(BACKEND_UNAVAILABLE, "The enhancement service is temporarily unavailable.") from exc

    return mcp


def mount_mcp_server(app: FastAPI) -> MCPServer:
    """Mount the MCP endpoint at MCP_PATH and return the server.

    The caller must run `server.session_manager.run()` for the life of the app:
    the session manager's task group belongs to the parent's lifespan, and
    without it the endpoint 500s on the first call.
    """
    settings = app.state.settings
    mcp = create_mcp_server(app)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=settings.mcp_allowed_hosts),
    )
    mcp_app.add_middleware(_ParentAppScope, parent=app)
    app.mount(MCP_PATH, mcp_app)
    return mcp
