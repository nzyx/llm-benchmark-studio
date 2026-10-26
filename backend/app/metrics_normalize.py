"""Normalise engine metrics into one schema.

llama.cpp and vLLM name the same concepts completely differently:

    concept              llama.cpp                     vLLM
    -------------------  ----------------------------  ---------------------------------
    running requests     llamacpp:requests_processing  vllm:num_requests_running
    queued requests      llamacpp:requests_deferred    vllm:num_requests_waiting
    KV cache usage       llamacpp:kv_cache_usage_ratio vllm:gpu_cache_usage_perc
                                                       vllm:kv_cache_usage_perc
    prompt tokens        llamacpp:prompt_tokens_total  vllm:prompt_tokens_total
    generated tokens     llamacpp:tokens_predicted_*   vllm:generation_tokens_*
    TTFT                 (not exposed)                 vllm:time_to_first_token_seconds_*

The vLLM column lists *both* names where the series has been renamed across
releases.  Guessing one and missing the other silently reports ``None`` for a
metric the engine is publishing, which is worse than not looking at all.

The report and the dashboard must not care which engine produced a run, so
everything is folded into :data:`NORMALIZED_KEYS` here.  Unknown series stay
available under ``raw`` for engine-specific display.
"""

from __future__ import annotations

from .prometheus import as_ratio, histogram_mean, pick

# Keys guaranteed to exist (value may be None when the engine is silent).
NORMALIZED_KEYS: tuple[str, ...] = (
    # live state
    "requests_running",
    "requests_waiting",
    "kv_cache_usage_ratio",
    "kv_cache_tokens",
    "kv_pool_positions",
    # cumulative work
    "prompt_tokens_total",
    "generation_tokens_total",
    "prompt_seconds_total",
    "generation_seconds_total",
    "prompt_tokens_cached_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "preemptions_total",
    "draft_tokens_total",
    "draft_tokens_accepted_total",
    "requests_success_total",
    # derived throughputs (computed, not scraped)
    "prompt_tokens_per_s",
    "generation_tokens_per_s",
    "draft_accept_ratio",
    "prefix_cache_hit_ratio",
    # request-level latency histograms (vLLM)
    "ttft_seconds_mean",
    "tpot_seconds_mean",
    "e2e_seconds_mean",
    # tokens actually computed in prefill: lower than prompt_tokens means the
    # engine reused cached context, which is a *direct* statement about caching
    "prefill_kv_tokens_mean",
)


LLAMACPP_ALIASES: dict[str, tuple[str, ...]] = {
    "requests_running": ("llamacpp:requests_processing",),
    "requests_waiting": ("llamacpp:requests_deferred",),
    "kv_cache_usage_ratio": ("llamacpp:kv_cache_usage_ratio",),
    "kv_cache_tokens": ("llamacpp:kv_cache_tokens",),
    "kv_pool_positions": ("halogen:kv_pool_positions",),
    "prompt_tokens_total": ("llamacpp:prompt_tokens_total",),
    "generation_tokens_total": ("llamacpp:tokens_predicted_total",),
    "prompt_seconds_total": ("llamacpp:prompt_seconds_total",),
    "generation_seconds_total": ("llamacpp:tokens_predicted_seconds_total",),
    "prompt_tokens_cached_total": ("halogen:prompt_tokens_cached_total",),
    "draft_tokens_total": ("halogen:draft_tokens_total",),
    "draft_tokens_accepted_total": ("halogen:draft_tokens_accepted_total",),
    "requests_success_total": ("halogen:requests_total",),
    # llama.cpp exposes instantaneous gauges instead of histograms
    "_prompt_tokens_per_s_gauge": ("llamacpp:prompt_tokens_seconds",),
    "_generation_tokens_per_s_gauge": ("llamacpp:predicted_tokens_seconds",),
}

VLLM_ALIASES: dict[str, tuple[str, ...]] = {
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    # renamed across releases; both carry the same 0..1 ratio
    "kv_cache_usage_ratio": ("vllm:gpu_cache_usage_perc",
                             "vllm:kv_cache_usage_perc"),
    "kv_cache_tokens": ("vllm:gpu_cache_usage_perc_sum",),
    "prompt_tokens_total": ("vllm:prompt_tokens_total",),
    "generation_tokens_total": ("vllm:generation_tokens_total",),
    "prompt_tokens_cached_total": (
        "vllm:prompt_tokens_cached_total",
        "vllm:prefix_cache_hits_total",
    ),
    "prefix_cache_queries_total": ("vllm:prefix_cache_queries_total",),
    "prefix_cache_hits_total": ("vllm:prefix_cache_hits_total",),
    # preemptions mean requests were pushed out of the cache to make room:
    # the classic saturation symptom under concurrency + long context
    "preemptions_total": ("vllm:num_preemptions_total",),
    "requests_success_total": ("vllm:request_success_total",),
    "_ttft_base": ("vllm:time_to_first_token_seconds",),
    "_tpot_base": ("vllm:time_per_output_token_seconds",),
    "_e2e_base": ("vllm:e2e_request_latency_seconds",),
    "_prefill_kv_base": ("vllm:request_prefill_kv_computed_tokens",),
}

GENERIC_ALIASES: dict[str, tuple[str, ...]] = {}


def _aliases_for(family: str) -> dict[str, tuple[str, ...]]:
    if family == "llamacpp":
        return LLAMACPP_ALIASES
    if family == "vllm":
        return VLLM_ALIASES
    return GENERIC_ALIASES


def normalize(samples: dict[str, float], family: str) -> dict[str, float | None]:
    """Map raw ``samples`` (labels already collapsed) onto the shared schema."""
    aliases = _aliases_for(family)
    out: dict[str, float | None] = {k: None for k in NORMALIZED_KEYS}

    for key, names in aliases.items():
        if key.startswith("_"):
            continue
        out[key] = pick(samples, *names)

    # percentages reported 0..100 -> 0..1
    out["kv_cache_usage_ratio"] = as_ratio(
        out.get("kv_cache_usage_ratio"), percent_style=True)

    # ---- histograms (vLLM) ------------------------------------------
    for norm_key, alias_key in (("ttft_seconds_mean", "_ttft_base"),
                                ("tpot_seconds_mean", "_tpot_base"),
                                ("e2e_seconds_mean", "_e2e_base"),
                                ("prefill_kv_tokens_mean", "_prefill_kv_base")):
        names = aliases.get(alias_key, ())
        for base in names:
            mean = histogram_mean(samples, base)
            if mean is not None:
                out[norm_key] = mean
                break

    # ---- derived -----------------------------------------------------
    p_tok, p_sec = out.get("prompt_tokens_total"), out.get("prompt_seconds_total")
    if p_tok is not None and p_sec:
        out["prompt_tokens_per_s"] = p_tok / p_sec
    else:
        gauge = pick(samples, *aliases.get("_prompt_tokens_per_s_gauge", ()))
        out["prompt_tokens_per_s"] = gauge

    g_tok = out.get("generation_tokens_total")
    g_sec = out.get("generation_seconds_total")
    if g_tok is not None and g_sec:
        out["generation_tokens_per_s"] = g_tok / g_sec
    else:
        gauge = pick(samples, *aliases.get("_generation_tokens_per_s_gauge", ()))
        out["generation_tokens_per_s"] = gauge

    drafted, accepted = out.get("draft_tokens_total"), out.get("draft_tokens_accepted_total")
    if drafted and accepted is not None:
        out["draft_accept_ratio"] = accepted / drafted

    queries = out.get("prefix_cache_queries_total")
    hits = out.get("prefix_cache_hits_total")
    if queries and hits is not None:
        out["prefix_cache_hit_ratio"] = hits / queries

    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in out.items()}


def human_summary(norm: dict[str, float | None], family: str) -> list[dict[str, object]]:
    """A few labelled rows ready for the UI/report."""
    rows: list[dict[str, object]] = []

    def add(label: str, key: str, unit: str, digits: int = 2) -> None:
        value = norm.get(key)
        if value is None:
            return
        rows.append({
            "label": label,
            "key": key,
            "unit": unit,
            "value": round(value, digits) if isinstance(value, float) else value,
        })

    add("运行中请求", "requests_running", "路", 0)
    add("排队请求", "requests_waiting", "路", 0)
    add("KV cache 占用", "kv_cache_usage_ratio", "ratio", 4)
    add("Prompt 累计 tokens", "prompt_tokens_total", "tokens", 0)
    add("生成累计 tokens", "generation_tokens_total", "tokens", 0)
    add("Prompt 吞吐(累计)", "prompt_tokens_per_s", "tok/s", 1)
    add("生成吞吐(累计)", "generation_tokens_per_s", "tok/s", 1)
    add("缓存命中 prompt tokens", "prompt_tokens_cached_total", "tokens", 0)
    # a non-zero hit ratio is the engine admitting prefix caching is at work
    add("prefix cache 命中率", "prefix_cache_hit_ratio", "ratio", 4)
    add("被抢占请求", "preemptions_total", "次", 0)
    if family == "llamacpp":
        add("投机解码接受率", "draft_accept_ratio", "ratio", 3)
    add("TTFT 均值(服务端)", "ttft_seconds_mean", "s", 4)
    add("TPOT 均值(服务端)", "tpot_seconds_mean", "s", 4)
    return rows
