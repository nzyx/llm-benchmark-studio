"""Abstract Provider interface.

Every inference engine (OpenAI-compatible server, llama.cpp, vLLM, and later
hosted APIs such as OpenAI / DeepSeek / Claude / Gemini) is wrapped by an
adapter that implements exactly this surface.

Transport is not the adapter's problem any more - ``BaseProvider`` owns an
:class:`~app.client.ApiClient` and hands it to subclasses as ``self.api``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ..client import ApiClient, RetryPolicy, redact
from ..errors import ApiError

# Historical name kept so callers (main.py, smoke tests) do not have to change.
ProviderError = ApiError

__all__ = [
    "ApiError",
    "ProviderError",
    "BaseProvider",
    "ModelInfo",
    "StreamEvent",
    "redact",
]

#: Tiny generation budget for the thinking-control probe.  A reasoning model
#: emits thinking even for a trivial question, so a handful of tokens is
#: enough to see the difference, and two requests cost well under a second.
THINKING_PROBE_TOKENS = 16
#: A prompt that reliably provokes reasoning in a thinking model while still
#: being answerable in one short line when thinking is disabled.
_THINKING_PROBE_PROMPT = (
    "A train covers 90 km in 45 minutes. What is its average speed in km/h? "
    "Answer with the number only."
)


@dataclass
class StreamEvent:
    """A single step of a streaming generation.

    ``t`` is seconds elapsed since the request was sent, measured with
    ``time.perf_counter`` for maximum resolution.
    """

    type: str                      # "delta" | "usage" | "done" | "error"
    t: float = 0.0
    text: str = ""                 # visible content delta
    reasoning: str = ""            # reasoning / thinking delta
    usage: dict[str, Any] | None = None
    #: Server-side timings block, when the engine emits one (llama.cpp forks
    #: do).  ``cache_n`` is a direct statement of how many prompt tokens were
    #: served from cache - far better evidence than inferring from TTFT.
    timings: dict[str, Any] | None = None
    error: str | None = None
    raw: dict[str, Any] | None = None
    finish_reason: str | None = None

    def is_token(self) -> bool:
        return self.type == "delta" and (bool(self.text) or bool(self.reasoning))


@dataclass
class ModelInfo:
    id: str
    #: short, human-readable name for the UI.  llama.cpp reports the full
    #: GGUF path as the model id, which is unusable in a dropdown.
    label: str | None = None
    max_model_len: int | None = None
    context_length: int | None = None
    owned_by: str | None = None
    max_tokens_cap: int | None = None
    max_tokens_default: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label or self.id,
            "max_model_len": self.max_model_len,
            "context_length": self.context_length,
            "owned_by": self.owned_by,
            "max_tokens_cap": self.max_tokens_cap,
            "max_tokens_default": self.max_tokens_default,
            "meta": self.meta,
        }

    def to_param_values(self) -> dict[str, Any]:
        """Feed the read-only ``model`` parameter category."""
        return {
            "context_window": self.max_model_len or self.context_length,
            "server_max_output": self.max_tokens_cap,
            "server_default_output": self.max_tokens_default,
        }


class BaseProvider(abc.ABC):
    """The uniform adapter contract.

    Optional, engine-specific surface (defaults are no-ops so unsupported
    engines simply report ``None``):

      * :meth:`tokenize`      - exact token counting for input construction
      * :meth:`get_props`     - engine property dump (llama.cpp ``/props``)
      * :meth:`get_slots`     - per-slot state (llama.cpp ``/slots``)
      * :attr:`capabilities`  - what this engine can actually do
    """

    kind: str = "unknown"
    engine: str | None = None
    api_version: str | None = None
    metrics_family: str = "generic"
    capabilities: dict[str, bool] = {}
    #: the engine's default thinking strength ("xhigh", "medium", ...) when it
    #: reports one.  Lives outside `capabilities` because that dict is typed
    #: as booleans and the UI renders every key in it as a flag.
    reasoning_effort_default: str | None = None
    #: True when the engine *demonstrably* emits reasoning before content.
    #: Measured, not declared: engines like vLLM state nothing, so asking is
    #: the only way to know, and getting it wrong means a reasoning model
    #: spends the whole output budget on thinking and emits no answer.
    thinks_by_default: bool | None = None
    #: raw result of :meth:`probe_thinking_control`, for the report
    thinking_probe: dict[str, Any] | None = None

    def __init__(self, base_url: str, api_key: str | None = None,
                 timeout_s: float = 120.0, retries: int = 2,
                 trust_env: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.trust_env = trust_env
        self.api = ApiClient(
            self.base_url,
            api_key=api_key,
            timeout_s=self.timeout_s,
            retry=RetryPolicy.from_retries(self.retries),
            trust_env=trust_env,
        )

    # -- key hygiene -------------------------------------------------------
    @property
    def masked_key(self) -> str:
        """Safe-to-log representation of the credential."""
        return redact(self.api._api_key)  # noqa: SLF001 - intentional

    # -- lifecycle ---------------------------------------------------------
    async def aclose(self) -> None:
        await self.api.aclose()

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- uniform API -------------------------------------------------------
    @abc.abstractmethod
    async def get_models(self) -> list[ModelInfo]:
        """Return the models exposed by the service."""

    @abc.abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Return a normalised health / capability description."""

    @abc.abstractmethod
    async def generate(self, model: str, prompt: str | None = None,
                       messages: list[dict[str, Any]] | None = None,
                       **kwargs: Any) -> dict[str, Any]:
        """Non-streaming generation. Returns a normalised result dict."""

    @abc.abstractmethod
    def stream_generate(self, model: str, prompt: str | None = None,
                        messages: list[dict[str, Any]] | None = None,
                        **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Streaming generation as an async iterator of :class:`StreamEvent`."""

    @abc.abstractmethod
    async def get_metrics(self) -> dict[str, Any]:
        """Return engine metrics (Prometheus text parsed into a dict)."""

    # -- optional engine-specific surface ----------------------------------
    async def tokenize(self, text: str, model: str | None = None
                       ) -> list[int] | None:
        """Return exact token ids, or ``None`` when unsupported."""
        return None

    async def detokenize(self, tokens: list[int], model: str | None = None
                         ) -> str | None:
        """Inverse of :meth:`tokenize`; used to trim an input to an exact
        token count for the Context benchmark."""
        return None

    async def get_props(self) -> dict[str, Any] | None:
        """Engine property dump (llama.cpp ``/props``), or ``None``."""
        return None

    async def get_slots(self) -> list[dict[str, Any]] | None:
        """Per-slot state (llama.cpp ``/slots``), or ``None``."""
        return None

    def supports(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability))

    async def ping(self, timeout_s: float = 5.0) -> bool:
        """Is the service answering right now?

        Deliberately cheap and short: it is used *between* benchmark steps to
        decide whether the engine died, so it must not itself hang for the
        full request timeout, and it must never raise.  A service that is
        listening but wedged - the accept backlog case - fails this check on
        the read timeout, which is the honest answer.
        """
        status, _ = await self.api.probe("/v1/models", timeout_s=timeout_s)
        return status == 200

    async def probe_thinking_control(self, model: str,
                                     timeout_s: float = 60.0) -> dict[str, Any]:
        """Measure whether ``chat_template_kwargs.enable_thinking`` works.

        Some engines advertise their thinking controls (llama.cpp puts them in
        ``/props.chat_template_caps``); vLLM advertises nothing, and a static
        "not declared, therefore unsupported" conclusion silently throws away
        the only switch that makes a reasoning model produce visible output.

        So this asks the engine directly, with two tiny requests, and reports
        what actually happened.  The cost is a couple of hundred milliseconds
        and a handful of tokens - cheap next to measuring the wrong thing.

        Returns a dict; ``supported`` is ``None`` when neither request produced
        any reasoning (nothing to steer, so nothing was learned).
        """
        messages = [{"role": "user", "content": _THINKING_PROBE_PROMPT}]
        result: dict[str, Any] = {
            "supported": None,
            "probe_tokens": THINKING_PROBE_TOKENS,
        }
        try:
            base = await self.generate(
                model, messages=messages, max_tokens=THINKING_PROBE_TOKENS,
                temperature=0.0, timeout_s=timeout_s)
            off = await self.generate(
                model, messages=messages, max_tokens=THINKING_PROBE_TOKENS,
                temperature=0.0, timeout_s=timeout_s,
                options={"chat_template_kwargs": {"enable_thinking": False}})
        except ApiError as exc:
            result["error"] = str(exc)
            return result
        except Exception as exc:  # noqa: BLE001 - a probe must never break a run
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result

        b_reason = len(base.get("reasoning") or "")
        o_reason = len(off.get("reasoning") or "")
        b_text = len(base.get("text") or "")
        o_text = len(off.get("text") or "")
        result.update({
            "baseline_reasoning_chars": b_reason,
            "baseline_content_chars": b_text,
            "off_reasoning_chars": o_reason,
            "off_content_chars": o_text,
            "field": "chat_template_kwargs.enable_thinking",
        })
        if b_reason == 0 and o_reason == 0:
            # no thinking observed at all: the question is moot, and saying
            # "unsupported" would be a claim we cannot support
            result["supported"] = None
            result["conclusion"] = "no_reasoning_observed"
        elif o_reason < b_reason:
            result["supported"] = True
            result["conclusion"] = "disabling_it_removes_reasoning"
        else:
            result["supported"] = False
            result["conclusion"] = "no_effect"
        return result

    # -- helpers -----------------------------------------------------------
    async def infer_thinking_control(self, model_id: str,
                                     health: dict[str, Any]) -> None:
        """Fill in ``thinking_control`` by measurement when nothing declared it.

        Engines that publish a chat template (llama.cpp's
        ``/props.chat_template_caps``) already answered this question, so they
        are left alone.  Everyone else gets the behavioural probe, and the
        outcome is attached to ``health`` for the report.
        """
        if self.supports("thinking_control"):
            return
        probe = await self.probe_thinking_control(model_id)
        self.thinking_probe = probe
        self.thinks_by_default = bool(probe.get("baseline_reasoning_chars") or 0)
        health["thinking_probe"] = probe
        if probe.get("supported") is True:
            # per-instance copy: `capabilities` is a class-level default and
            # mutating it in place would leak across providers
            self.capabilities = {**self.capabilities, "thinking_control": True}
        elif probe.get("supported") is False:
            health.setdefault("_warnings", []).append(
                "已实测：`chat_template_kwargs.enable_thinking` 对该引擎无效，"
                "思考链无法关闭"
            )

    def provider_info(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "engine": self.engine,
            "base_url": self.base_url,
            "api_version": self.api_version,
        }
