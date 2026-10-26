"""Generic OpenAI-compatible provider adapter.

Works with llama.cpp server, vLLM, halogen, LM Studio, Ollama's
OpenAI-compatible endpoint and any other server that speaks
``/v1/chat/completions`` + ``/v1/completions`` + ``/v1/models``.

Phase 2: all transport concerns (pooling, timeout, retry, streaming lines)
live in :class:`app.client.ApiClient`; this module only deals with protocol
shape and timing.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, AsyncIterator

from ..client import redact
from ..metrics_normalize import human_summary, normalize
from ..prometheus import collapse, parse_exposition
from .base import BaseProvider, ModelInfo, ProviderError, StreamEvent

__all__ = ["OpenAICompatProvider", "redact", "short_model_label"]

_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)

# Engines disagree on the name of the thinking/reasoning delta:
#
#   reasoning_content   llama.cpp forks, DeepSeek-style servers
#   reasoning           vLLM (verified on 0.29.0.dev0, Qwen3.8)
#
# Reading only one of them silently drops *every* reasoning chunk, which makes
# the client count zero output tokens and record a successful request as a
# failure.  Seen in the wild, so both are accepted.
_REASONING_FIELDS: tuple[str, ...] = ("reasoning_content", "reasoning")


def _delta_reasoning(delta: dict[str, Any]) -> str:
    """First non-empty reasoning field in ``delta``, whatever it is called."""
    for key in _REASONING_FIELDS:
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _message_reasoning(message: dict[str, Any]) -> str:
    """Same, for a non-streaming ``choices[].message``."""
    for key in (*_REASONING_FIELDS, "analysis"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def short_model_label(model_id: str) -> str | None:
    """Turn ``/home/nn/models/Foo-UD-Q4_K_XL-00001-of-00004.gguf`` into
    ``Foo-UD-Q4_K_XL`` for display, keeping the full id for API calls.

    Returns ``None`` when there is nothing to shorten, i.e. the id is already
    a clean name.
    """
    if "/" not in model_id and "\\" not in model_id:
        return None
    base = re.split(r"[/\\]", model_id)[-1]
    base = _SHARD_RE.sub("", base)
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    return base or None


class OpenAICompatProvider(BaseProvider):
    kind = "openai_compatible"

    # ------------------------------------------------------------------ #
    # model discovery
    # ------------------------------------------------------------------ #
    async def get_models(self) -> list[ModelInfo]:
        data = await self.api.json("GET", "/v1/models")
        out: list[ModelInfo] = []
        for item in (data.get("data") or []):
            meta = item.get("meta") or {}
            model_id = item.get("id", "unknown")
            out.append(ModelInfo(
                id=model_id,
                label=short_model_label(model_id),
                max_model_len=item.get("max_model_len"),
                context_length=(item.get("context_length")
                                or meta.get("n_ctx")),
                owned_by=item.get("owned_by"),
                max_tokens_cap=item.get("max_tokens_cap"),
                max_tokens_default=item.get("max_tokens_default"),
                meta=meta,
            ))
        return out

    # ------------------------------------------------------------------ #
    # health / fingerprint
    # ------------------------------------------------------------------ #
    async def health_check(self) -> dict[str, Any]:
        health: dict[str, Any] = {"reachable": False}
        try:
            health = await self.api.json("GET", "/health")
            health["reachable"] = True
            health["_source"] = "/health"
        except ProviderError:
            models = await self.get_models()
            health = {
                "reachable": True,
                "model": models[0].id if models else None,
                "_source": "/v1/models",
            }

        ver = health.get("version") or {}
        if isinstance(ver, dict):
            self.api_version = ver.get("api") or ver.get("engine")
            self.engine = ver.get("engine")

        return health

    # ------------------------------------------------------------------ #
    # request building
    # ------------------------------------------------------------------ #
    def _build_payload(self, model: str, prompt: str | None,
                       messages: list[dict[str, Any]] | None,
                       stream: bool, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens")
        extra = kwargs.get("extra_body") or {}

        if messages is not None:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
            }
            path = "/v1/chat/completions"
        else:
            payload = {
                "model": model,
                "prompt": prompt or "",
                "temperature": temperature,
                "stream": stream,
            }
            path = "/v1/completions"

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        for k, v in (kwargs.get("options") or {}).items():
            if v is not None:
                payload[k] = v
        payload.update(extra)

        if stream:
            # ask the server for an exact usage block at the end of the stream
            so = dict(payload.get("stream_options") or {})
            so.setdefault("include_usage", True)
            payload["stream_options"] = so
        return path, payload

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    async def generate(self, model: str, prompt: str | None = None,
                       messages: list[dict[str, Any]] | None = None,
                       **kwargs: Any) -> dict[str, Any]:
        path, payload = self._build_payload(model, prompt, messages, False, **kwargs)
        t0 = time.perf_counter()
        data = await self.api.json("POST", path, body=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") if messages is not None else choice.get("text")
        return {
            "text": text or "",
            "reasoning": _message_reasoning(msg),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "timings": data.get("timings") or None,
            "elapsed_ms": elapsed_ms,
            "raw_id": data.get("id"),
        }

    async def stream_generate(self, model: str, prompt: str | None = None,
                              messages: list[dict[str, Any]] | None = None,
                              **kwargs: Any) -> AsyncIterator[StreamEvent]:
        # a per-request timeout override, used by the max-context binary search
        timeout_s = kwargs.pop("timeout_s", None)
        path, payload = self._build_payload(model, prompt, messages, True, **kwargs)
        t0 = time.perf_counter()
        usage: dict[str, Any] | None = None
        timings: dict[str, Any] | None = None
        try:
            async for line in self.api.stream_lines("POST", path, body=payload,
                                                    timeout_s=timeout_s):
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # llama.cpp forks attach a server-side timings block to the
                # last chunks; the final one wins (it is the complete one)
                if isinstance(obj.get("timings"), dict):
                    timings = obj["timings"]
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = _delta_reasoning(delta)
                finish = ch.get("finish_reason")
                if text or reasoning or finish:
                    yield StreamEvent(
                        type="delta", t=time.perf_counter() - t0,
                        text=text, reasoning=reasoning, finish_reason=finish,
                    )
        except ProviderError as exc:
            yield StreamEvent(type="error", t=time.perf_counter() - t0,
                              error=str(exc))
            return
        if usage:
            yield StreamEvent(type="usage", t=time.perf_counter() - t0,
                              usage=usage, timings=timings)
        yield StreamEvent(type="done", t=time.perf_counter() - t0,
                          usage=usage, timings=timings)

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #
    async def get_metrics(self) -> dict[str, Any]:
        """Fetch ``/metrics`` and normalise it into the shared schema.

        Subclasses only need to set :attr:`metrics_family`.
        """
        try:
            text = await self.api.text("GET", "/metrics", timeout_s=self.timeout_s)
        except ProviderError as exc:
            return {
                "available": False, "family": self.metrics_family,
                "error": str(exc), "samples": {}, "normalized": {}, "summary": [],
            }

        samples_list, metric_types = parse_exposition(text)
        # histogram bucket series carry no meaning once labels are folded
        collapsed = collapse([s for s in samples_list
                              if not s.name.endswith("_bucket")])
        normalized = normalize(collapsed, self.metrics_family)

        return {
            "available": bool(collapsed),
            "family": self.metrics_family,
            "series_count": len(samples_list),
            "metric_types": metric_types,
            "samples": collapsed,
            "normalized": normalized,
            "summary": human_summary(normalized, self.metrics_family),
        }
