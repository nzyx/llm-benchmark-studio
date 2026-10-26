"""llama.cpp family adapter.

Targets ``llama-server`` and its forks.  Everything OpenAI-compatible is
inherited; this module adds the native surface that makes llama.cpp-family
engines measurable rather than merely callable:

  * ``/props``  - real ``n_ctx``, slot count, build info, generation defaults
  * ``/slots``  - per-slot state (who is holding KV cache right now)
  * ``/tokenize`` - exact token ids, needed to build a precise input length
  * ``/metrics`` - ``llamacpp:*`` counters normalised into the shared schema

Note: forks exist.  The engine on hand exposes ``llamacpp:*`` metrics and a
rich ``/health`` but answers 404 on ``/props`` and ``/slots``, so every
native call here is capability-gated and returns ``None`` when absent.
"""

from __future__ import annotations

import json
from typing import Any

from ..errors import ApiError
from .base import ModelInfo
from .openai_compat import OpenAICompatProvider

# Fields of /props worth keeping - everything else is noise or huge.
_PROPS_KEEP = (
    "n_ctx", "total_slots", "model_path", "build_info", "modalities",
    "chat_template", "bos_token", "eos_token", "is_sleeping",
)
_GEN_SETTINGS_KEEP = (
    "n_predict", "temperature", "top_p", "top_k", "min_p", "seed",
    "repeat_penalty", "n_ctx", "cache_prompt", "stop",
)
_SLOT_KEEP = (
    "id", "state", "n_ctx", "is_processing", "n_past", "n_prompt_tokens",
    "n_prompt_tokens_processed", "next_token", "task_id", "prompt_hash",
)


def _select(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: source[k] for k in keys if k in source}


def trim_props(props: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce ``/props`` to the part that matters (chat_template can be ~100KB)."""
    if not isinstance(props, dict):
        return None
    out = _select(props, _PROPS_KEEP)
    ct = out.get("chat_template")
    if isinstance(ct, str):
        out["chat_template"] = {
            "length": len(ct),
            "sha256": props.get("chat_template_sha256"),
            "preview": ct[:160],
        }
    gen = props.get("default_generation_settings")
    if isinstance(gen, dict):
        # newer builds nest the sampler params one level deeper under "params"
        flat = {**gen, **(gen.get("params") or {})}
        out["default_generation_settings"] = _select(flat, _GEN_SETTINGS_KEEP)
    return out


def props_n_ctx(props: dict[str, Any] | None) -> int | None:
    """llama.cpp moved ``n_ctx`` around between builds; check both places."""
    if not isinstance(props, dict):
        return None
    if props.get("n_ctx"):
        return int(props["n_ctx"])
    gen = props.get("default_generation_settings")
    if isinstance(gen, dict) and gen.get("n_ctx"):
        return int(gen["n_ctx"])
    return None


def summarize_slots(slots: Any) -> list[dict[str, Any]] | None:
    """Reduce ``/slots`` (which embeds prompt text) to a compact status list."""
    if not isinstance(slots, list):
        return None
    out: list[dict[str, Any]] = []
    for slot in slots:
        if isinstance(slot, dict):
            out.append(_select(slot, _SLOT_KEEP))
    return out


class LlamaCppProvider(OpenAICompatProvider):
    kind = "llamacpp"
    metrics_family = "llamacpp"

    # ------------------------------------------------------------------ #
    async def _fetch_raw_props(self) -> dict[str, Any] | None:
        status, text = await self.api.probe("/props", timeout_s=15.0)
        if status != 200:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        # keep the raw payload reachable for version sniffing, but trimmed
        self._raw_props = data
        return data

    async def get_props(self) -> dict[str, Any] | None:
        cached = getattr(self, "_trimmed_props", None)
        if cached is not None:
            return cached
        raw = getattr(self, "_raw_props", None) or await self._fetch_raw_props()
        if raw is None:
            return None
        self._trimmed_props = trim_props(raw)
        return self._trimmed_props

    async def get_slots(self) -> list[dict[str, Any]] | None:
        status, text = await self.api.probe("/slots", timeout_s=15.0)
        if status != 200:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return summarize_slots(data)

    async def tokenize(self, text: str, model: str | None = None
                       ) -> list[int] | None:
        """Exact token ids via the native endpoint.

        This is what lets the Context benchmark hit a target input length
        precisely instead of estimating from character counts.
        """
        for path, body in (("/tokenize", {"content": text}),
                           ("/v1/tokenize", {"content": text})):
            try:
                data = await self.api.json("POST", path, body=body)
            except ApiError:
                continue
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return tokens
        return None

    async def detokenize(self, tokens: list[int], model: str | None = None
                         ) -> str | None:
        """Exact inverse of :meth:`tokenize` (llama.cpp ``/detokenize``)."""
        if not tokens:
            return None
        for path in ("/detokenize", "/v1/detokenize"):
            try:
                data = await self.api.json("POST", path, body={"tokens": tokens})
            except ApiError:
                continue
            content = data.get("content")
            if isinstance(content, str):
                return content
        return None

    # ------------------------------------------------------------------ #
    async def health_check(self) -> dict[str, Any]:
        health = await super().health_check()

        raw = getattr(self, "_raw_props", None)
        if raw is None:
            raw = await self._fetch_raw_props() or {}
        props = await self.get_props()

        if not self.engine:
            build = raw.get("build_info") or health.get("build_info")
            if isinstance(build, str):
                self.engine = build.split()[0] if build else None
        if self.engine:
            health.setdefault("version", {})
            if isinstance(health["version"], dict):
                health["version"].setdefault("engine", self.engine)

        n_ctx = props_n_ctx(raw)
        if n_ctx and not health.get("n_ctx"):
            health["n_ctx"] = n_ctx
        if raw.get("total_slots") and not health.get("total_slots"):
            health["total_slots"] = raw["total_slots"]
        if raw.get("model_alias") and not health.get("model_alias"):
            health["model_alias"] = raw["model_alias"]

        # /props and /slots are best-effort; a fork may 404 both.
        if props and not health.get("_props"):
            health["_props"] = props
        if not health.get("_slots_detail"):
            slots = await self.get_slots()
            if slots is not None:
                health["_slots_detail"] = slots

        # A fork may expose neither /props nor chat_template_caps, in which
        # case nothing has declared whether thinking can be steered.  Ask the
        # engine rather than concluding "unsupported" from silence.
        try:
            models = await self.get_models()
        except Exception:  # noqa: BLE001 - capability probing is optional
            models = []
        if models:
            await self.infer_thinking_control(models[0].id, health)
        return health

    # ------------------------------------------------------------------ #
    def prompt_cache_hint(self, health: dict[str, Any]) -> dict[str, Any]:
        """Report what we can honestly say about prompt caching."""
        pc = health.get("prompt_cache")
        if isinstance(pc, dict):
            return {"enabled": bool(pc.get("enabled")), "source": "/health"}
        props = health.get("_props") or {}
        gen = props.get("default_generation_settings") or {}
        if "cache_prompt" in gen:
            return {"enabled": bool(gen["cache_prompt"]), "source": "/props"}
        return {"enabled": None, "source": None}

    def enrich_models(self, models: list[ModelInfo], health: dict[str, Any]
                      ) -> list[ModelInfo]:
        """Backfill context length from engine-native fields when absent."""
        n_ctx = health.get("n_ctx") or health.get("context")
        if not n_ctx:
            return models
        for m in models:
            if not m.max_model_len:
                m.max_model_len = int(n_ctx)
            if not m.context_length:
                m.context_length = int(n_ctx)
        return models
