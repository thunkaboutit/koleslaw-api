import logging
import os
import re
from collections.abc import AsyncGenerator

import httpx
import litellm
from fastapi import HTTPException
from litellm import acompletion

from koleslaw_api.config import Settings
from koleslaw_api.prompts.system import classify_enhancement_type, get_system_prompt
from koleslaw_api.schemas.enhance import (
    EnhanceChunk,
    EnhanceRequest,
    EnhanceResponse,
    ModelInfo,
    Usage,
)

# How much of a rejected model string to echo back in the 400 — enough to
# debug a legitimate typo, capped so a hostile over-long value isn't reflected.
_MAX_MODEL_ECHO = 80


def resolve_allowed_model(requested: str | None, settings: Settings) -> str:
    """Resolve a caller-supplied model to a permitted model id, or raise 400.

    Callers may select only ``settings.default_model`` or a tag explicitly
    listed in ``settings.allowed_models``. Cloud models are server-internal
    fallback only — never caller-selectable — so a caller cannot route requests
    to a paid provider on our account, whether directly (``openai/gpt-4o``) or
    by naming a bogus ``ollama_chat/*`` tag that would 404 into the cloud
    fallback. Matching is exact (after trimming); a prefix rule would miss that
    second path. See docs/security-review-2026-07-19.md (T1-A).
    """
    if requested is None or not requested.strip():
        return settings.default_model
    model = requested.strip()
    allowed = {settings.default_model, *settings.allowed_models}
    if model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model[:_MAX_MODEL_ECHO]}' is not available. See GET /v1/models for the models you can use."
            ),
        )
    return model


# ── Binary attachment routing ────────────────────────────────────────
#
# /v1/chat/upload turns an attached binary file into an image_url content
# block, which only a vision-capable model can accept. Our self-hosted default
# is a text-only fine-tune, so it answers such a request with a 400 ("model
# does not support multimodal requests"). Those requests therefore have to be
# routed to the vision-capable cloud tier or refused up front.

# The formats the vision-capable cloud tier actually accepts. Anything else
# binary (PDF, audio, video, office documents) used to be base64-encoded into
# an image_url block anyway and failed deep inside the provider; refusing it
# here turns a confusing backend error into a clear one, and stops a large
# unusable file from being encoded and shipped at all.
SUPPORTED_IMAGE_MIMES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)

# Plan that may not spend metered provider budget on attachments. The seeded
# public playground key is IP-capped, not account-capped, so routing it to the
# cloud tier would put unauthenticated traffic straight onto paid inference,
# five files of up to 10 MB a time.
ATTACHMENT_DENIED_PLAN = "anonymous"


def model_supports_vision(model: str) -> bool:
    """Whether LiteLLM reports image-input support for this model id.

    An id LiteLLM does not know answers False. Our own Ollama tags are not in
    its capability map, and False is the safe direction: guessing "yes" sends
    images to a model that rejects them, which is the bug this replaces.
    """
    try:
        return bool(litellm.supports_vision(model=model))
    except Exception:
        return False


def resolve_attachment_model(model: str, settings: Settings, plan: str | None) -> str:
    """Return a model able to accept image input, or raise 403/422.

    The returned model is chosen by the server, never by the caller, so this is
    the same server-internal lane the T1-A allowlist in ``resolve_allowed_model``
    keeps callers out of. A caller still cannot name a cloud model; it can only
    attach an image and have the server decide where that has to run.
    """
    if model_supports_vision(model):
        return model

    if plan == ATTACHMENT_DENIED_PLAN:
        raise HTTPException(
            status_code=403,
            detail=(
                "Image attachments need an account. Sign in at https://koleslaw.ai "
                "to create a free API key, or attach a text file instead."
            ),
        )

    fallback = settings.fallback_model
    if not fallback or not model_supports_vision(fallback):
        raise HTTPException(
            status_code=422,
            detail=("Image attachments are not available right now. Attach a text file instead."),
        )
    return fallback


class LLMServiceError(Exception):
    def __init__(self, message: str, code: str, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.detail = detail


# Failures that mean the backend serving the model is unreachable or not
# serving it right now — worth retrying on another tier, unlike auth or
# bad-request errors, which would fail identically anywhere. NotFoundError
# counts: a GPU instance that lost its model registration answers 404, which
# says nothing about whether a cloud model can take the request.
_AVAILABILITY_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.NotFoundError,
)

# LiteLLM reports some provider HTTP errors as APIConnectionError no matter what
# status the provider actually sent: an Ollama 400 arrives as one, wrapping the
# provider's JSON body. APIConnectionError is also the exact signal the
# availability fallback exists for (a reclaimed GPU), so without this a plain
# bad request would be retried on the metered cloud tier and fail there too, at
# cost. Parse the wrapped status and treat the codes that mean "this request is
# wrong" as non-retryable.
_UPSTREAM_STATUS_RE = re.compile(
    r"""(?:status_?code|code)\\?["']?\s*[:=]\s*\\?["']?(\d{3})|error\s+code:\s*(\d{3})""",
    re.IGNORECASE,
)

# 404 is deliberately absent: an instance that lost its model registration
# answers 404, and that IS an availability problem (see _AVAILABILITY_ERRORS).
# 408, 429 and every 5xx are absent for the same reason — another tier may well
# serve what this one could not.
_NON_RETRYABLE_UPSTREAM_STATUSES = frozenset({400, 401, 403, 413, 422})

_GENERIC_LLM_ERROR = "Something went wrong generating a response. Please try again."


def wrapped_upstream_status(exc: Exception) -> int | None:
    """The provider-reported HTTP status embedded in a LiteLLM error message."""
    match = _UPSTREAM_STATUS_RE.search(str(exc))
    if match is None:
        return None
    raw = match.group(1) or match.group(2)
    return int(raw) if raw else None


def user_facing_llm_error(exc: Exception) -> str:
    """A short, safe message for a client.

    Provider exception text carries backend identity, model tags and raw
    upstream JSON. It used to reach callers verbatim — streamed into the chat
    body, and put in the error field of every LLMServiceError response. Full
    detail still goes to the log; only this string leaves the process.
    """
    if isinstance(exc, litellm.ContextWindowExceededError):
        return "That request is too large for the model. Try shortening it or removing an attachment."
    if isinstance(exc, litellm.RateLimitError):
        return "The service is busy right now. Please try again in a moment."
    if isinstance(exc, litellm.AuthenticationError):
        return "The service is temporarily misconfigured. Please try again later."
    if isinstance(exc, litellm.BadRequestError):
        return "That request could not be processed. Try changing the prompt or removing an attachment."
    if isinstance(exc, _AVAILABILITY_ERRORS):
        # Includes a wrapped provider 4xx, which reads as a transport error
        # here; the generic retry wording is still accurate enough for a client.
        return "The model is temporarily unavailable. Please try again in a moment."
    return _GENERIC_LLM_ERROR


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key.get_secret_value()
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
        os.environ["OLLAMA_API_BASE"] = settings.ollama_base_url

    def _resolve_model(self, request: EnhanceRequest) -> str:
        # Defense in depth: routes validate at the boundary, but the service
        # never calls acompletion with an unvalidated model either (T1-A).
        return resolve_allowed_model(request.model, self._settings)

    @staticmethod
    def resolve_enhancement_type(request: EnhanceRequest) -> str:
        if request.enhancement_type == "auto":
            return classify_enhancement_type(request.prompt)
        return request.enhancement_type

    def _build_messages(self, request: EnhanceRequest, resolved_type: str, model: str) -> list[dict[str, str]]:
        # Ollama-served models define their SYSTEM prompt in the Modelfile
        # (kept next to the weights in the model repo / S3). Sending no
        # system message lets it apply — a request-level one would override
        # it. Cloud fallback models have no Modelfile, so they still get the
        # per-type prompt; without it they'd answer the prompt instead of
        # enhancing it.
        if model.startswith("ollama_chat/"):
            content = request.prompt
            if request.context:
                content += (
                    "\n\n--- Additional Context ---\n"
                    "Context about the use case, to inform enhancement "
                    f"decisions:\n\n{request.context}"
                )
            return [{"role": "user", "content": content}]
        system_prompt = get_system_prompt(resolved_type, request.context)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt},
        ]

    @staticmethod
    def _log_llm_error(exc: Exception, model: str) -> None:
        logging.getLogger(__name__).error(
            "LLM call failed: model=%s, type=%s, error=%s", model, type(exc).__name__, exc
        )

    def _primary_retries(self, model: str) -> int:
        """LiteLLM retries for the first call.  A local-tier model with a
        cloud fallback gets none: a dead or reclaimed GPU fails every retry
        the same way, and each one can burn a full timeout (4 x 90s sync)
        before the fallback answers.  Fail once, fall back at once."""
        fallback = self._settings.fallback_model
        if fallback and fallback != model and model.startswith("ollama_chat/"):
            return 0
        return 3

    def _fall_back_or_raise(self, model: str, exc: Exception) -> str:
        """Return the fallback model for an availability failure on the local
        tier, or raise the mapped error.

        A GPU instance replacement (deploy, future spot reclaim) is a ~4 min
        serving gap; requests degrade to the configured cloud model instead
        of surfacing it. Only ollama_chat/* primaries qualify — a cloud
        model failing is not an availability problem another cloud model
        is likely to dodge, and its errors should stay visible.
        """
        self._log_llm_error(exc, model)
        fallback = self._settings.fallback_model
        if (
            not fallback
            or fallback == model
            or not model.startswith("ollama_chat/")
            or not isinstance(exc, _AVAILABILITY_ERRORS)
            # A provider 4xx that LiteLLM wrapped as a transport error is a bad
            # request, not an outage; paying the cloud tier to fail identically
            # helps nobody.
            or wrapped_upstream_status(exc) in _NON_RETRYABLE_UPSTREAM_STATUSES
        ):
            raise self._map_exception(exc) from exc
        logging.getLogger(__name__).warning(
            "Model %s unavailable (%s); falling back to %s", model, type(exc).__name__, fallback
        )
        return fallback

    def _map_exception(self, exc: Exception) -> LLMServiceError:
        # The message is the sanitised one: every caller here has already sent
        # the raw exception to the log via _log_llm_error, and this string is
        # what the HTTP error handler and the MCP tool hand back to a caller.
        message = user_facing_llm_error(exc)
        if isinstance(exc, litellm.AuthenticationError):
            return LLMServiceError(message, "authentication_error")
        if isinstance(exc, litellm.RateLimitError):
            return LLMServiceError(message, "rate_limit_error")
        if isinstance(exc, litellm.BadRequestError):
            return LLMServiceError(message, "bad_request")
        if isinstance(exc, litellm.NotFoundError):
            return LLMServiceError(message, "model_not_found")
        # A wrapped provider 4xx reports as a transport error but is really a
        # client error; 502 would tell the caller to retry something that fails
        # the same way every time.
        wrapped = wrapped_upstream_status(exc)
        if wrapped in (401, 403):
            return LLMServiceError(message, "authentication_error")
        if wrapped in _NON_RETRYABLE_UPSTREAM_STATUSES:
            return LLMServiceError(message, "bad_request")
        if isinstance(exc, (litellm.Timeout, litellm.APIConnectionError)):
            return LLMServiceError(message, "connection_error")
        if isinstance(exc, litellm.ServiceUnavailableError):
            return LLMServiceError(message, "service_unavailable")
        return LLMServiceError(message, "llm_error", detail=type(exc).__name__)

    async def enhance(self, request: EnhanceRequest, resolved_type: str | None = None) -> EnhanceResponse:
        model = self._resolve_model(request)
        resolved = resolved_type or self.resolve_enhancement_type(request)
        # The sync path gets no bytes until the whole generation is done, so
        # this timeout covers queue wait + generation. A local-tier timeout
        # raises before any byte reaches the caller — the availability
        # fallback below turns it into a fast cloud answer, and Ollama drops
        # the abandoned request from its queue on disconnect.
        timeout = self._settings.llm_sync_timeout_seconds
        try:
            response = await acompletion(
                model=model,
                messages=self._build_messages(request, resolved, model),
                timeout=timeout,
                num_retries=self._primary_retries(model),
            )
        except Exception as exc:
            model = self._fall_back_or_raise(model, exc)
            try:
                response = await acompletion(
                    model=model,
                    messages=self._build_messages(request, resolved, model),
                    timeout=timeout,
                    num_retries=3,
                )
            except Exception as fallback_exc:
                self._log_llm_error(fallback_exc, model)
                raise self._map_exception(fallback_exc) from fallback_exc

        return EnhanceResponse(
            enhanced_prompt=response.choices[0].message.content,
            model_used=model,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    async def enhance_stream(
        self, request: EnhanceRequest, resolved_type: str | None = None
    ) -> AsyncGenerator[EnhanceChunk, None]:
        model = self._resolve_model(request)
        resolved = resolved_type or self.resolve_enhancement_type(request)
        # Streaming timeouts are between-read, not total-duration: this caps
        # how long a request may sit in Ollama's queue waiting for its first
        # token (the server sends nothing until a slot frees), while an
        # active generation streams tokens far more often than the ceiling
        # and is never cut off. Ollama serves one request at a time, so under
        # a burst this is what turns "hang for minutes" into "cloud fallback
        # in seconds" — see also OLLAMA_MAX_QUEUE on the instance, which
        # 503s excess queue depth instantly for the same reason.
        timeout = self._settings.llm_stream_timeout_seconds
        try:
            response = await acompletion(
                model=model,
                messages=self._build_messages(request, resolved, model),
                stream=True,
                timeout=timeout,
                num_retries=self._primary_retries(model),
            )
        except Exception as exc:
            # Fallback only engages before the first byte reaches the
            # client; a mid-stream failure can't be retried without
            # duplicating output, so it propagates as before.
            model = self._fall_back_or_raise(model, exc)
            try:
                response = await acompletion(
                    model=model,
                    messages=self._build_messages(request, resolved, model),
                    stream=True,
                    timeout=timeout,
                    num_retries=3,
                )
            except Exception as fallback_exc:
                self._log_llm_error(fallback_exc, model)
                raise self._map_exception(fallback_exc) from fallback_exc

        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            finish_reason = chunk.choices[0].finish_reason
            if delta or finish_reason:
                yield EnhanceChunk(delta=delta, finish_reason=finish_reason)

    async def get_available_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []

        # Query Ollama for locally available models
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._settings.ollama_base_url}/api/tags")
                resp.raise_for_status()
                for m in resp.json().get("models", []):
                    name = m.get("name", "")
                    if not name:
                        continue
                    models.append(
                        ModelInfo(
                            id=f"ollama_chat/{name}",
                            name=name,
                            provider="ollama",
                            tier="local",
                        )
                    )
        except Exception:
            logging.getLogger(__name__).warning("Failed to fetch models from Ollama")

        # Add cloud provider models if API keys are configured
        if self._settings.anthropic_api_key:
            models.extend(
                [
                    ModelInfo(
                        id="anthropic/claude-haiku-4-5", name="Claude Haiku 4.5", provider="anthropic", tier="fast"
                    ),
                    ModelInfo(
                        id="anthropic/claude-sonnet-4-5", name="Claude Sonnet 4", provider="anthropic", tier="standard"
                    ),
                ]
            )
        if self._settings.openai_api_key:
            models.extend(
                [
                    ModelInfo(id="openai/gpt-4o-mini", name="GPT-4o Mini", provider="openai", tier="fast"),
                    ModelInfo(id="openai/gpt-4o", name="GPT-4o", provider="openai", tier="premium"),
                ]
            )

        # Only advertise what a caller can actually select — same allowlist
        # resolve_allowed_model enforces — so /v1/models never lists a model the
        # LLM endpoints would 400 (cloud/fallback + unserved tags drop out; T1-A).
        allowed = {self._settings.default_model, *self._settings.allowed_models}
        return [m for m in models if m.id in allowed]
