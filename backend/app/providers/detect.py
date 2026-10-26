"""Engine fingerprinting and adapter selection.

A benchmark tool that guesses wrong about the engine produces confidently
wrong numbers, so detection is done from *observable evidence* only, and the
evidence is reported back to the user instead of being hidden.

Signals, cheapest first:

  ``/metrics``   ``vllm:*``  -> vLLM ; ``llamacpp:*`` -> llama.cpp family
  ``/props``     llama.cpp model properties (absent on vLLM and on forks)
  ``/slots``     llama.cpp per-slot state
  ``/health``    empty 200 -> vLLM convention
                 ``{"status":"ok"}`` -> vanilla llama.cpp
                 slot / KV-pool fields -> llama.cpp fork
  route probing  a ``405`` on a GET proves the POST route exists, which lets
                 us detect ``/completion`` and ``/tokenize`` without spending
                 any generation tokens.

Every native probe is treated as optional.  A fork that answers 404 on
``/props`` must still be classified as llama.cpp family - that is exactly the
engine we have on hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ..client import ApiClient, RetryPolicy
from ..config import TRUST_ENV_PROXY
from ..prometheus import collapse, parse_exposition
from .base import BaseProvider
from .llamacpp import LlamaCppProvider
from .openai_compat import OpenAICompatProvider
from .vllm import VLLMProvider

PROBE_TIMEOUT_S = 12.0
METRICS_TIMEOUT_S = 20.0

# A GET returning any of these means "the route exists, wrong method or bad input".
_ROUTE_EXISTS = {200, 201, 202, 204, 400, 401, 403, 405, 415, 422}

# Bodies used to probe POST-only routes.  Both are designed to be free:
# tokenizing one character costs nothing, and `n_predict: 0` makes llama.cpp
# return immediately with ``tokens_predicted: 0``.
_TOKENIZE_PROBE = {"content": "x"}
_COMPLETION_PROBE = {"prompt": "", "n_predict": 0}
# Health fields that only ever appear on llama.cpp-family engines.
_LLAMACPP_HEALTH_FIELDS = {
    "slots", "kv_pool_positions", "slot_ctx", "drafters_available",
    "prompt_cache", "n_ctx", "total_slots", "cache_counters",
}

ADAPTER_FOR_FAMILY = {
    "llamacpp": LlamaCppProvider,
    "vllm": VLLMProvider,
    "openai_compatible": OpenAICompatProvider,
}


@dataclass
class Capabilities:
    """What the engine can actually do - drives UI hints and safe defaults."""

    props: bool = False
    slots: bool = False
    tokenize: bool = False
    native_completion: bool = False
    metrics_prometheus: bool = False
    kv_cache_metrics: bool = False
    request_latency_histograms: bool = False
    cache_prompt_control: bool = False
    prompt_cache_enabled: bool | None = None
    speculative_decoding: bool = False
    vision: bool = False
    tool_calls: bool = False
    thinking_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Detection:
    family: str
    confidence: float
    adapter: str
    evidence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    capabilities: Capabilities = field(default_factory=Capabilities)
    health: dict[str, Any] = field(default_factory=dict)
    props: dict[str, Any] | None = None
    slots: list[dict[str, Any]] | None = None
    metrics_family: str | None = None
    metrics_samples: dict[str, float] = field(default_factory=dict)
    #: engine-stated cache configuration (from an ``*_info`` metric)
    cache_config: dict[str, str] = field(default_factory=dict)
    #: the engine's own default thinking strength, when it reports one
    reasoning_effort_default: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "confidence": self.confidence,
            "adapter": self.adapter,
            "evidence": self.evidence,
            "notes": self.notes,
            "capabilities": self.capabilities.to_dict(),
            "health": self.health,
            "props": self.props,
            "slots": self.slots,
            "metrics_family": self.metrics_family,
            "metrics_available": bool(self.metrics_samples),
            "cache_config": self.cache_config,
            "reasoning_effort_default": self.reasoning_effort_default,
        }


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
async def _route_exists(client: ApiClient, path: str) -> bool:
    """GET-based route discovery.

    Works when the server answers ``405 Method Not Allowed`` for a POST-only
    route (llama.cpp forks such as halogen do).  Some builds answer ``404``
    instead, which is why :func:`_post_route_exists` exists as the primary
    signal.
    """
    status, _ = await client.probe(path, timeout_s=8.0)
    return status in _ROUTE_EXISTS


async def _post_route_exists(client: ApiClient, path: str,
                             body: Any) -> bool:
    """POST-based route discovery with a deliberately free payload.

    Cheaper to reason about than the 405 trick and correct on every build
    seen so far: a real route answers 2xx/4xx-other-than-404, while a missing
    one answers 404.
    """
    status, _ = await client.probe(path, method="POST", body=body,
                                   timeout_s=30.0)
    return status in _ROUTE_EXISTS


def _looks_like_llamacpp_health(health: dict[str, Any]) -> bool:
    return bool(_LLAMACPP_HEALTH_FIELDS & set(health))


def _as_int(value: Any) -> int | None:
    """``info`` metrics report numbers as strings (``"1532"``); coerce."""
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _collect_health_capabilities(caps: Capabilities, health: dict[str, Any],
                                 evidence: list[str]) -> None:
    if not isinstance(health, dict):
        return
    pc = health.get("prompt_cache")
    if isinstance(pc, dict):
        caps.prompt_cache_enabled = bool(pc.get("enabled"))
        evidence.append(f"/health reports prompt_cache.enabled={caps.prompt_cache_enabled}")

    if health.get("drafters_available") or health.get("drafter_default"):
        caps.speculative_decoding = True
        drafter = health.get("drafter_default") or health.get("drafters_available")
        evidence.append(f"speculative decoding configured ({drafter})")

    # vLLM never declares speculative decoding in /health - the only trace is
    # the ``vllm:spec_decode_*`` counters.  Missing this made the report claim
    # "no speculative decoding" on a node whose every scheduler step produced
    # ~2.3 tokens, so single-token-per-step intuition did not hold.
    spec = health.get("speculative_decoding")
    if isinstance(spec, dict) and spec.get("enabled"):
        caps.speculative_decoding = True
        rate = spec.get("acceptance_rate")
        per_step = spec.get("tokens_per_step")
        bits = []
        if per_step is not None:
            bits.append(f"每步产出约 {per_step:g} tokens")
        if rate is not None:
            bits.append(f"草稿接受率 {rate * 100:.1f}%")
        evidence.append(
            "投机解码已启用（vllm:spec_decode_* 计数非零"
            + ("，" + "，".join(bits) if bits else "") + "）"
        )

    vision = health.get("vision")
    if isinstance(vision, dict):
        caps.vision = bool(vision.get("enabled"))
    modalities = health.get("modalities")
    if isinstance(modalities, dict) and modalities.get("vision"):
        caps.vision = True

    if health.get("tool_calls"):
        caps.tool_calls = True

    template = health.get("chat_template")
    if isinstance(template, dict) and template.get("thinking_control"):
        caps.thinking_control = True
    elif "enable_thinking" in (health.get("supported") or []):
        caps.thinking_control = True


# ---------------------------------------------------------------------- #
# detection
# ---------------------------------------------------------------------- #
async def detect(base_url: str, api_key: str | None = None, *,
                 timeout_s: float = 120.0, retries: int = 1,
                 client: ApiClient | None = None) -> Detection:
    owns_client = client is None
    api = client or ApiClient(
        base_url, api_key, timeout_s=timeout_s,
        retry=RetryPolicy.from_retries(retries), trust_env=TRUST_ENV_PROXY)
    try:
        return await _detect_with(api)
    finally:
        if owns_client:
            await api.aclose()


async def _detect_with(api: ApiClient) -> Detection:
    evidence: list[str] = []
    notes: list[str] = []
    caps = Capabilities()

    # ---- /metrics (most definitive) -----------------------------------
    samples: dict[str, float] = {}
    # ``info`` metrics carry their payload in *labels*, not in the value, so
    # they are kept separately - ``collapse`` would reduce them to a bare 1.0.
    info: dict[str, dict[str, str]] = {}
    metrics_family: str | None = None
    m_status, m_text = await api.probe("/metrics", timeout_s=METRICS_TIMEOUT_S)
    if m_status == 200 and m_text.strip():
        parsed, _ = parse_exposition(m_text)
        for s in parsed:
            if s.labels and s.name.endswith("_info"):
                info.setdefault(s.name, {}).update(s.labels)
        # histogram buckets are meaningless once labels are folded
        samples = collapse([s for s in parsed if not s.name.endswith("_bucket")])
        caps.metrics_prometheus = bool(samples)
        if "vllm:" in m_text:
            metrics_family = "vllm"
        elif "llamacpp:" in m_text:
            metrics_family = "llamacpp"
        if metrics_family is None and caps.metrics_prometheus:
            notes.append("/metrics 可用，但没有 vllm:/llamacpp: 前缀，指标映射可能不完整")
    else:
        reason = ""
        if m_text:
            try:
                message = (json.loads(m_text).get("error") or {}).get("message")
                if message:
                    reason = f"（{message}）"
            except (json.JSONDecodeError, AttributeError):
                pass
        notes.append("/metrics 不可用" + reason
                     + "，服务端指标只能退回到 /health 的瞬时字段")

    # ---- /health -------------------------------------------------------
    health: dict[str, Any] = {}
    health_is_json = False
    h_status, h_text = await api.probe("/health", timeout_s=PROBE_TIMEOUT_S)
    if h_status == 200 and h_text.strip():
        try:
            loaded = json.loads(h_text)
            if isinstance(loaded, dict):
                health, health_is_json = loaded, True
        except json.JSONDecodeError:
            health_is_json = False
    elif h_status == 200:
        health_is_json = False  # empty 200 body

    # ---- /props --------------------------------------------------------
    props: dict[str, Any] | None = None
    p_status, p_text = await api.probe("/props", timeout_s=PROBE_TIMEOUT_S)
    if p_status == 200:
        try:
            candidate = json.loads(p_text)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and (
            {"default_generation_settings", "model_path", "total_slots", "n_ctx"}
            & set(candidate)
        ):
            props = candidate
            caps.props = True
            evidence.append("/props 返回 llama.cpp 模型属性")

    # ---- /slots --------------------------------------------------------
    slots: list[Any] | None = None
    s_status, s_text = await api.probe("/slots", timeout_s=PROBE_TIMEOUT_S)
    if s_status == 200:
        try:
            candidate = json.loads(s_text)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            slots = candidate
            caps.slots = True
            evidence.append(f"/slots 返回 {len(candidate)} 个槽位")

    # ---- native route discovery ----------------------------------------
    # POST-first with a free payload, then fall back to the 405-on-GET trick
    # for builds that do answer 405 instead of 404 on a wrong method.
    tokenize_ok = (
        await _post_route_exists(api, "/tokenize", _TOKENIZE_PROBE)
        or await _post_route_exists(api, "/v1/tokenize",
                                    {**_TOKENIZE_PROBE, "model": "probe"})
        or await _route_exists(api, "/tokenize")
        or await _route_exists(api, "/v1/tokenize")
    )
    completion_ok = (
        await _post_route_exists(api, "/completion", _COMPLETION_PROBE)
        or await _route_exists(api, "/completion")
    )
    caps.tokenize = tokenize_ok
    caps.native_completion = completion_ok
    if tokenize_ok:
        evidence.append("/tokenize 可用（可精确构造输入长度，Phase 5 需要）")
    if completion_ok:
        evidence.append("/completion 原生端点可用")

    # ---- decide family -------------------------------------------------
    family = "openai_compatible"
    confidence = 0.30

    if metrics_family == "vllm":
        family, confidence = "vllm", 0.95
        evidence.append("/metrics 含 `vllm:*` 序列")
    elif metrics_family == "llamacpp":
        family, confidence = "llamacpp", 0.95
        evidence.append("/metrics 含 `llamacpp:*` 序列")
    elif caps.props or caps.slots:
        family, confidence = "llamacpp", 0.85
        evidence.append("llama.cpp 原生端点（/props 或 /slots）有响应")
    elif health_is_json and _looks_like_llamacpp_health(health):
        family, confidence = "llamacpp", 0.70
        evidence.append("/health 暴露 llama.cpp 系的槽位 / KV pool 字段")
    elif h_status == 200 and not h_text.strip():
        family, confidence = "vllm", 0.60
        evidence.append("/health 返回空 200（vLLM 惯例）")
    elif health_is_json and str(health.get("status", "")).lower() == "ok":
        family, confidence = "llamacpp", 0.55
        evidence.append('/health 返回 {"status":"ok"}（llama.cpp 惯例）')

    # ---- capability refinement ----------------------------------------
    if family == "llamacpp":
        gen_defaults = {}
        if isinstance(props, dict):
            gen_defaults = props.get("default_generation_settings") or {}
            # newer builds nest sampler params one level deeper
            gen_defaults = {**gen_defaults,
                            **(gen_defaults.get("params") or {})}
        caps.cache_prompt_control = bool(
            caps.native_completion or "cache_prompt" in gen_defaults)
        caps.kv_cache_metrics = any(
            k in samples for k in ("llamacpp:kv_cache_usage_ratio",
                                   "llamacpp:kv_cache_tokens"))
        if caps.cache_prompt_control:
            evidence.append("原生 /completion 存在，`cache_prompt` 为该端点文档字段")
        if not (caps.props or caps.slots):
            notes.append(
                "该 llama.cpp 系引擎未提供 /props 与 /slots（疑似 fork），"
                "原生属性与槽位视图不可用"
            )
        if not caps.cache_prompt_control:
            notes.append(
                "未能确认可关闭 prompt cache：prefill 指标可能被缓存影响，"
                "请用不重复的输入做正式测试"
            )
        # an explicit negative flag from /props is worth surfacing
        if isinstance(props, dict) and props.get("endpoint_metrics") is False:
            notes.append("/props 明确声明 metrics 端点未启用（需启动时加 --metrics）")

        # llama.cpp exposes chat-template capabilities rather than a single
        # `thinking_control` flag; reasoning support implies it can be steered.
        template_caps = (props or {}).get("chat_template_caps")
        if isinstance(template_caps, dict):
            supports = ("supports_reasoning_effort", "supports_preserve_reasoning")
            if any(template_caps.get(k) for k in supports):
                caps.thinking_control = True
                evidence.append(
                    "/props.chat_template_caps 声明支持 reasoning 控制"
                    "（enable_thinking 经 chat_template_kwargs 传递）"
                )
            if template_caps.get("supports_tool_calls") or template_caps.get("supports_tools"):
                caps.tool_calls = True
    elif family == "vllm":
        # The gauge has been renamed across releases: `gpu_cache_usage_perc`
        # up to ~0.9, `kv_cache_usage_perc` after.  Both mean the same thing.
        caps.kv_cache_metrics = any(
            k in samples for k in ("vllm:gpu_cache_usage_perc",
                                   "vllm:kv_cache_usage_perc"))
        caps.request_latency_histograms = any(
            k.startswith("vllm:time_to_first_token_seconds") for k in samples)
        caps.tokenize = caps.tokenize  # vLLM needs an explicit model name
        notes.append("vLLM 无 /props 与 /slots，并发上限需通过 --max-num-seqs 在启动参数确认")

        # ``vllm:cache_config_info`` is the engine *stating* its own cache
        # configuration.  That is the strongest possible evidence about
        # prefix caching, and it matters: with it on, a repeated prompt is
        # served partly from cache and TTFT stops describing the engine.
        cache_cfg = info.get("vllm:cache_config_info") or {}
        if cache_cfg:
            flag = str(cache_cfg.get("enable_prefix_caching", "")).strip().lower()
            if flag in ("true", "false"):
                caps.prompt_cache_enabled = flag == "true"
                evidence.append(
                    f"/metrics 的 cache_config_info 声明 "
                    f"enable_prefix_caching={flag}")
            pool = _as_int(cache_cfg.get("kv_cache_size_tokens"))
            if pool:
                evidence.append(f"KV cache 容量 {pool:,} tokens（长上下文与并发的真天花板）")
            ceiling = _as_float(cache_cfg.get("kv_cache_max_concurrency"))
            if ceiling:
                evidence.append(f"cache_config_info 推算的最大并发 {ceiling:g} 路")
            dtype = cache_cfg.get("cache_dtype")
            if dtype and dtype not in ("auto", "None"):
                evidence.append(f"KV cache dtype = {dtype}")
            if caps.prompt_cache_enabled and not caps.cache_prompt_control:
                notes.append(
                    "该 vLLM 开启了 prefix caching 且**没有关掉的 API 开关**："
                    "重复的输入会被部分复用，Prefill / TTFT 会偏乐观 —— "
                    "正式测试请使用不重复的输入（本工具会自动改为唯一输入）"
                )
        if caps.request_latency_histograms:
            evidence.append("/metrics 含 TTFT / E2E 请求级直方图（可与客户端测量交叉校验）")

        # Speculative decoding is the single biggest reason a vLLM node's
        # tok/s fluctuates: one scheduler step yields a *variable* number of
        # tokens (the drafter's guess, minus whatever verification rejected).
        # vLLM never mentions it in /health, so the counters are the only
        # evidence - and a client that assumes "one stream event = one token"
        # will mis-derive both the rate and the per-token latency.
        drafts = samples.get("vllm:spec_decode_num_drafts_total") or 0.0
        drafted = samples.get("vllm:spec_decode_num_draft_tokens_total") or 0.0
        accepted = samples.get("vllm:spec_decode_num_accepted_tokens_total") or 0.0
        if drafts > 0 or drafted > 0:
            caps.speculative_decoding = True
            bits = []
            if drafts > 0:
                bits.append(f"每调度步草拟 {drafted / drafts:.1f} tokens")
            if drafted > 0:
                bits.append(f"接受率 {accepted / drafted * 100:.0f}%")
            if drafts > 0:
                bits.append(f"实际每步产出约 {1 + accepted / drafts:.2f} tokens")
            evidence.append(
                "/metrics 的 spec_decode 计数非零，投机解码已启用"
                + ("（" + "，".join(bits) + "）" if bits else "")
            )
            notes.append(
                "该引擎启用了**投机解码**：一个流事件可能携带多个 token，"
                "decode 速度会随草稿接受率波动，样本间差异比单 token 引擎大是正常的"
            )
    else:
        notes.append("未能识别具体引擎，使用通用 OpenAI 兼容适配器")
        if caps.tokenize:
            notes.append("检测到 /tokenize，但未识别引擎家族，不做精确 token 计数")

    _collect_health_capabilities(caps, health, evidence)

    # A thinking-by-default engine will spend the output budget on reasoning
    # first.  Say so at probe time, where the user can still act on it, rather
    # than letting them discover it in an empty `content` field.
    effort = health.get("reasoning_effort_default") if health_is_json else None
    reasoning_effort = str(effort) if isinstance(effort, str) else None
    if caps.thinking_control and reasoning_effort in ("high", "xhigh"):
        budget = (health.get("server_defaults") or {}).get(
            "HALOGEN_MAX_THINKING_TOKENS") if health_is_json else None
        extra = f"，思考预算上限 {budget}" if isinstance(budget, int) else ""
        notes.append(
            f"该引擎默认开启思考链（reasoning_effort={reasoning_effort}{extra}）—— "
            "小 max_tokens 会被思考链吃光、正文为空；"
            "做性能/上下文测试建议把 `enable_thinking` 设为 false"
        )

    adapter_cls = ADAPTER_FOR_FAMILY.get(family, OpenAICompatProvider)
    return Detection(
        family=family,
        confidence=round(confidence, 2),
        adapter=adapter_cls.__name__,
        evidence=evidence,
        notes=notes,
        capabilities=caps,
        health=health,
        props=props,
        slots=slots,
        metrics_family=metrics_family,
        metrics_samples=samples,
        cache_config=info.get("vllm:cache_config_info") or info.get("halogen:cache_config_info") or {},
        reasoning_effort_default=reasoning_effort,
    )


# ---------------------------------------------------------------------- #
# provider construction
# ---------------------------------------------------------------------- #
def build_provider(
    base_url: str,
    api_key: str | None = None,
    *,
    family: str = "openai_compatible",
    timeout_s: float = 120.0,
    retries: int = 2,
    detection: Detection | None = None,
) -> BaseProvider:
    cls = ADAPTER_FOR_FAMILY.get(family, OpenAICompatProvider)
    provider = cls(base_url, api_key, timeout_s=timeout_s, retries=retries,
                   trust_env=TRUST_ENV_PROXY)
    if detection is not None and detection.family == family:
        provider.capabilities = detection.capabilities.to_dict()
        provider.metrics_family = detection.metrics_family or provider.metrics_family
        provider.reasoning_effort_default = detection.reasoning_effort_default
    return provider


async def select_provider(
    base_url: str,
    api_key: str | None = None,
    *,
    family: str | None = None,
    timeout_s: float = 120.0,
    retries: int = 2,
) -> tuple[BaseProvider, Detection | None]:
    """Detect (unless the caller pinned ``family``) and return a ready adapter."""
    detection: Detection | None = None
    if family is None:
        detection = await detect(base_url, api_key, timeout_s=timeout_s,
                                 retries=min(retries, 1))
        family = detection.family
    provider = build_provider(base_url, api_key, family=family,
                              timeout_s=timeout_s, retries=retries,
                              detection=detection)
    return provider, detection
