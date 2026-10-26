"""vLLM adapter.

vLLM speaks the OpenAI protocol and adds two things worth adapting for:

  * ``/health`` returns **200 with an empty body** (not JSON) - which is a
    useful fingerprint, and a trap for any code that assumes JSON.
  * ``/metrics`` exposes ``vllm:*`` series, including request-level latency
    histograms (``time_to_first_token_seconds`` etc.).  Those are the
    *server's* view of TTFT/TPOT and are a good cross-check against what the
    client measures itself.

There is no ``/props`` or ``/slots``.  Concurrency limits are expressed as
``--max-num-seqs`` and surface only through the metrics.
"""

from __future__ import annotations

from typing import Any

from ..errors import ApiError
from ..prometheus import info_metric, parse_exposition
from .openai_compat import OpenAICompatProvider


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


class VLLMProvider(OpenAICompatProvider):
    kind = "vllm"
    metrics_family = "vllm"

    # ------------------------------------------------------------------ #
    async def health_check(self) -> dict[str, Any]:
        status, text = await self.api.probe("/health", timeout_s=15.0)
        health: dict[str, Any] = {
            "reachable": status == 200,
            "_source": "/health",
        }
        # vLLM answers 200 with an empty body; some builds return JSON.
        if status == 200 and text.strip():
            import json

            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    health.update(parsed)
                    health.pop("reachable", None)
                    health["reachable"] = True
            except json.JSONDecodeError:
                health["_body_preview"] = text[:200]
        if status != 200:
            models = await self.get_models()
            health = {"reachable": True, "_source": "/v1/models",
                      "model": models[0].id if models else None}

        # vLLM states its cache configuration as an ``info`` metric.  Reading
        # it here (rather than only at detect time) means the KV pool size and
        # the concurrency ceiling are available to every run.
        cfg = await self.cache_config()
        if cfg:
            health["cache_config"] = cfg
            pool = _num(cfg.get("kv_cache_size_tokens"))
            if pool:
                # same meaning as llama.cpp's kv_pool_positions: how many
                # tokens the KV cache can hold in total
                health["kv_pool_positions"] = int(pool)
            ceiling = _num(cfg.get("kv_cache_max_concurrency"))
            if ceiling:
                health["kv_cache_max_concurrency"] = ceiling
            flag = str(cfg.get("enable_prefix_caching", "")).strip().lower()
            if flag in ("true", "false"):
                health["prompt_cache"] = {"enabled": flag == "true",
                                          "_source": "vllm:cache_config_info"}

        # Speculative decoding changes what a tok/s number means: one scheduler
        # step yields a *variable* number of tokens (draft + accepted), so the
        # rate depends on how well the drafter guessed that step.  vLLM states
        # this only through counters, never in /health - reading them is what
        # keeps the report from presenting spec-decode throughput as if it were
        # plain decode.
        spec = await self.spec_decode_stats()
        if spec:
            health["speculative_decoding"] = spec

        # vLLM declares no thinking controls anywhere, but the chat template
        # usually has them.  Measuring beats assuming: without this, a
        # reasoning model spends every output token on thinking and the report
        # contains no answer text at all.
        try:
            models = await self.get_models()
        except Exception:  # noqa: BLE001 - capability probing is optional
            models = []
        if models:
            await self.infer_thinking_control(models[0].id, health)
        return health

    # ------------------------------------------------------------------ #
    async def cache_config(self) -> dict[str, str]:
        """The engine's own cache configuration, or ``{}`` when unavailable."""
        try:
            text = await self.api.text("GET", "/metrics", timeout_s=20.0)
        except ApiError:
            return {}
        parsed, _ = parse_exposition(text)
        return info_metric(parsed, "vllm:cache_config_info")

    # ------------------------------------------------------------------ #
    async def spec_decode_stats(self) -> dict[str, Any]:
        """Speculative-decoding counters, when the engine is running any.

        These are ``*_total`` counters, so the absolute values only mean
        "speculative decoding has been used at least once" - the useful derived
        number is the **acceptance rate** (``accepted / drafted``), which says
        how often the drafter's guess survived verification.  A low rate means
        each scheduler step yields an unpredictable number of tokens, which is
        exactly what makes a decode speed fluctuate between samples.

        ``{}`` when the engine does not expose the series at all; an explicit
        ``enabled: False`` when it exposes them at zero (configured but unused).
        """
        try:
            text = await self.api.text("GET", "/metrics", timeout_s=20.0)
        except ApiError:
            return {}
        parsed, _ = parse_exposition(text)

        # A counter can be split across label sets (one per engine / model);
        # summing is the only correct way to read a `_total`.
        names = {
            "drafts": "vllm:spec_decode_num_drafts_total",
            "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
            "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
        }
        totals: dict[str, float] = {}
        for key, name in names.items():
            values = [s.value for s in parsed if s.name == name]
            if not values:
                continue
            total = sum(v for v in values if v == v and v not in (
                float("inf"), float("-inf")))
            totals[key] = total
        if not totals:
            return {}
        if not any(v > 0 for v in totals.values()):
            return {"enabled": False, "_source": "vllm:spec_decode_*"}

        drafts = totals.get("drafts") or 0.0
        drafted = totals.get("draft_tokens") or 0.0
        accepted = totals.get("accepted_tokens") or 0.0
        out: dict[str, Any] = {
            "enabled": True,
            "_source": "vllm:spec_decode_*",
            "drafts": int(drafts),
            "draft_tokens": int(drafted),
            "accepted_tokens": int(accepted),
        }
        if drafted > 0:
            out["acceptance_rate"] = round(accepted / drafted, 4)
        if drafts > 0:
            # tokens produced per verified step = 1 guaranteed + accepted
            out["tokens_per_step"] = round(1 + accepted / drafts, 3)
            out["draft_tokens_per_step"] = round(drafted / drafts, 3)
        return out

    # ------------------------------------------------------------------ #
    async def tokenize(self, text: str, model: str | None = None
                       ) -> list[int] | None:
        """vLLM's ``/tokenize`` requires an explicit model name."""
        if not model:
            return None
        for path in ("/tokenize", "/v1/tokenize"):
            try:
                data = await self.api.json(
                    "POST", path, body={"model": model, "prompt": text})
            except ApiError:
                continue
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return tokens
        return None

    async def detokenize(self, tokens: list[int], model: str | None = None
                         ) -> str | None:
        """vLLM's ``/detokenize`` takes ``prompt`` in and returns it as text."""
        if not tokens or not model:
            return None
        for path in ("/detokenize", "/v1/detokenize"):
            try:
                data = await self.api.json(
                    "POST", path, body={"model": model, "tokens": tokens})
            except ApiError:
                continue
            for key in ("prompt", "content"):
                if isinstance(data.get(key), str):
                    return data[key]
        return None

    # ------------------------------------------------------------------ #
    def concurrency_hint(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """vLLM exposes live queue depth; that is the closest thing to a
        concurrency ceiling we can observe without CLI access."""
        norm = metrics.get("normalized") or {}
        return {
            "requests_running": norm.get("requests_running"),
            "requests_waiting": norm.get("requests_waiting"),
            "source": "vllm:num_requests_running / num_requests_waiting",
        }
