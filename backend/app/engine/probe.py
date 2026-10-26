"""One measured streaming request.

Extracted from the performance task so the Context (and later Concurrency)
tasks measure identically.  If two tasks measured TTFT differently the report
would contain numbers that are not comparable, which defeats the point.

Metric definitions (kept here, next to the code that produces them):

  ``ttft_ms``          first token of any kind (includes reasoning)
  ``ttft_content_ms``  first *visible* content token
  ``decode_window_ms`` ``e2e_ms - ttft_ms``
  ``tpot_ms``          ``decode_window_ms / (output_tokens - 1)``
  ``decode_tps``       ``1000 / tpot_ms``
  ``prefill_tps``      ``input_tokens / ttft_s``  (only honest when the prompt
                       cache is disabled - see the ``cache_prompt`` parameter)
  ``queue_wait_ms``    ``ttft_ms - server_prompt_ms`` when the engine reports
                       its own prefill time: the part of TTFT that is *not*
                       prefill (queueing / retry / cache restore)
  ``prefill_contaminated``  whether that wait is large enough to make
                       ``prefill_tps`` describe waiting rather than computing
  ``itl_ms``           mean gap between consecutive stream chunks

Every task must measure *and aggregate* the same way, because the three result
views sit side by side in the dashboard.  Two rules are therefore enforced
here rather than in each task:

  * **warmup samples are never part of a statistic** (:func:`summarize_samples`)
  * **the samples behind one number must all have used distinct inputs**
    (:func:`input_identity_warning`) - otherwise the second request of a group
    is served partly from the engine's prefix cache and the mean describes a
    cache hit rather than the engine.  Measured on a real node: identical
    inputs inflated prefill from 252 to 2000 tok/s (7.9x) within one ladder
    step, which is exactly the kind of "weird number" a user cannot diagnose.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any

from .. import metrics as M
from ..errors import classify_error

# Fixed-width tag: callers swap it between samples, so a varying width would
# quietly change each sample's input length (and therefore its prefill cost).
MARKER_PLACEHOLDER = "00000000"
_MARKER_TEMPLATE = "[bench {tag}] "
MARKER_WIDTH = len(_MARKER_TEMPLATE.format(tag=MARKER_PLACEHOLDER))

#: Minimum time between two *separately observable* stream events, in ms.
#: This is a statement about the client's ability to resolve a pulse train,
#: not about how fast an engine may legitimately be: if every token event
#: arrives inside the same read, the timestamps describe the socket flush
#: rather than the model, and no decode rate can be derived from them.
#: Measured on a live vLLM (Qwen3.8, spec decoding on): a stable 135-152 tok/s
#: at max_tokens>=32, but at max_tokens=8 the whole output was delivered with a
#: 0.036 ms span, which the naive formula turned into 196,078 tok/s.
MIN_STREAM_GAP_MS = 1.0

#: Client TTFT minus the engine's own prefill time is time the *engine did not
#: spend on this request*: queueing behind another job, a retry, or a cache
#: restore.  When that gap is large the sample's `prefill_tps` describes the
#: wait, not the compute.  Measured on a real llama.cpp fork: a 128K warmup died
#: on a read error at 50 s while its prefill needed ~85 s, the engine kept the
#: slot busy, and the NEXT request came back at 121,144 ms of TTFT against
#: 85,410 ms of the engine's own prefill time (41.8% overhead) - reported as
#: 1083 tok/s where the engine's own accounting says 1536.
QUEUE_CONTAMINATION_MIN_MS = 1000.0
QUEUE_CONTAMINATION_MIN_SHARE = 0.15


def input_marker(unique: bool = True) -> str:
    """A leading tag that makes one input distinguishable from another.

    It goes at the **front**.  Engines cache a *prefix*, so a tag appended
    after the prompt leaves the whole prompt as a matchable prefix - measured
    on a real llama.cpp fork, `cache_n` still reported 64 of 103 tokens served
    from cache (62%) with the tag at the end.  Leading with it makes the very
    first token differ, which no prefix cache can reuse.

    ``unique=False`` returns the placeholder, which is what a caller needs
    when it must first *measure* the input (and therefore needs a stable
    string) before substituting a real tag per sample.
    """
    tag = secrets.token_hex(4) if unique else MARKER_PLACEHOLDER
    return _MARKER_TEMPLATE.format(tag=tag)


def _r(seconds: float | None, scale: float) -> float | None:
    return round(seconds * scale, 3) if seconds is not None else None


def input_fingerprint(messages: list[dict[str, Any]] | None,
                      prompt: str | None) -> str:
    """A stable id for the *content* that was sent.

    This is what decides whether an engine can reuse its prefix cache, so it
    is the honest way to check that two samples are actually independent.
    Twelve hex characters: long enough that a collision is not a concern at
    benchmark scale, short enough to read in a dump.
    """
    if messages is not None:
        blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    else:
        blob = prompt or ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _num(v: Any) -> float | None:
    """Tolerant numeric coercion: servers report numbers as ints, floats or
    occasionally strings, and a benchmark must not crash on the difference."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def _server_timings(timings: dict[str, Any] | None,
                    usage: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise a server-reported timings block into comparable fields.

    Two different servers report the same idea in two different places:
    llama.cpp forks put ``cache_n`` in a top-level ``timings`` object, while
    the OpenAI schema puts ``prompt_tokens_details.cached_tokens`` in
    ``usage``.  Either one is a *direct statement* that part of the prompt came
    from cache, so both are folded into one field and the maximum is taken.
    """
    t = timings or {}
    details = (usage or {}).get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    candidates = [_num(t.get("cache_n")), _num(details.get("cached_tokens"))]
    present = [c for c in candidates if c is not None]
    return {
        "server_timings": bool(t),
        "server_prompt_ms": _num(t.get("prompt_ms")),
        "server_prefill_tps": _num(t.get("prompt_per_second")),
        "server_decode_tps": _num(t.get("predicted_per_second")),
        "server_predicted_ms": _num(t.get("predicted_ms")),
        "server_prompt_n": _num(t.get("prompt_n")),
        #: prompt tokens the engine served from its prompt cache.  Non-zero
        #: means the prefill numbers of this sample are not cold compute.
        "cache_hit_tokens": max(present) if present else None,
        "draft_tokens": _num(t.get("draft_n")),
        "draft_accepted": _num(t.get("draft_n_accepted")),
    }


async def measure_once(
    provider: Any,
    *,
    model: str,
    messages: list[dict[str, Any]] | None,
    prompt: str | None,
    max_tokens: int | None,
    temperature: float = 0.0,
    options: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run one streaming generation and return its raw measurements."""
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if options:
        kwargs["options"] = options
    if timeout_s is not None:
        kwargs["timeout_s"] = timeout_s

    t0 = time.perf_counter()
    token_times: list[float] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: str | None = None
    error_kind: str | None = None
    ttft_s: float | None = None
    ttft_content_s: float | None = None

    try:
        async for ev in provider.stream_generate(
            model=model, prompt=prompt, messages=messages, **kwargs
        ):
            if ev.type == "delta":
                if ev.is_token():
                    token_times.append(ev.t)
                    if ttft_s is None:
                        ttft_s = ev.t
                if ev.reasoning:
                    reasoning_parts.append(ev.reasoning)
                if ev.text:
                    text_parts.append(ev.text)
                    if ttft_content_s is None:
                        ttft_content_s = ev.t
                if ev.finish_reason:
                    finish_reason = ev.finish_reason
            elif ev.type == "usage":
                usage = ev.usage or usage
                timings = ev.timings or timings
            elif ev.type == "done":
                usage = usage or ev.usage
                timings = timings or ev.timings
            elif ev.type == "error":
                error = ev.error
                error_kind = classify_error(ev.error)
    except TimeoutError as exc:
        error = f"timeout: {exc}"
        error_kind = classify_error(exc)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        error_kind = classify_error(exc)

    e2e_s = time.perf_counter() - t0
    text = "".join(text_parts)
    reasoning = "".join(reasoning_parts)
    n_tokens = len(token_times)

    # ---- token accounting -------------------------------------------
    if usage:
        usage_src = "server"
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens")
    else:
        usage_src = "chunk_count"
        prompt_tokens = None
        completion_tokens = n_tokens
        total_tokens = n_tokens
        reasoning_tokens = None

    ok = error is None and n_tokens > 0

    # ---- server-side timings and cache-hit evidence -------------------
    # A fork that reports `cache_n > 0` has *told* us the prefill was partly
    # served from cache, so prefill/TTFT cannot represent cold compute.  That
    # is a fact rather than an inference, which is why it is worth carrying.
    server = _server_timings(timings, usage)

    sample: dict[str, Any] = {
        "ok": ok,
        "error": error,
        # stable classification of the failure, see app/errors.classify_error
        "error_kind": error_kind,
        # the engine started answering but did not finish: prefill may have
        # succeeded, the decode is incomplete - a distinct situation from
        # "never answered at all"
        "truncated": bool(error is not None and n_tokens > 0),
        # identity of the content that was sent.  Two samples with the same
        # fingerprint cannot be independent on an engine with a prefix cache,
        # so this is what makes "are these samples comparable?" checkable
        # instead of a matter of opinion.
        "input_hash": input_fingerprint(messages, prompt),
        "input_chars": len(prompt or "") or len(
            json.dumps(messages, ensure_ascii=False) if messages else ""),
        **server,
        "ttft_ms": _r(ttft_s, 1000),
        "ttft_content_ms": _r(ttft_content_s, 1000),
        "e2e_ms": round(e2e_s * 1000, 3),
        "decode_window_ms": None,
        #: how many stream events carried a token.  ``usage.completion_tokens``
        #: says how many tokens were *produced*; this says how many separately
        #: timed arrivals the client actually saw, which is what a client-side
        #: decode rate must be derived from.
        "stream_events": n_tokens,
        "decode_span_ms": None,
        #: why decode could not be measured, when it could not be
        "decode_unmeasurable": None,
        #: the slice of TTFT that was not prefill, when the engine tells us
        #: how long its own prefill took
        "queue_wait_ms": None,
        "prefill_contaminated": None,
        "tpot_ms": None,
        "decode_tps": None,
        "prefill_tps": None,
        "itl_ms": None,
        "itl_p95_ms": None,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "usage_source": usage_src,
        "finish_reason": finish_reason,
        "output_chars": len(text) + len(reasoning),
        "content_chars": len(text),
        # measured from the stream, not from usage: engines are inconsistent
        # about reporting completion_tokens_details
        "reasoning_chars": len(reasoning),
        "text_preview": (text or reasoning)[:160],
    }

    if ok and ttft_s is not None:
        window_s = max(e2e_s - ttft_s, 1e-6)
        out_tok = completion_tokens or n_tokens
        sample["decode_window_ms"] = round(window_s * 1000, 3)

        # ---- decode rate, only when the arrivals are resolvable ---------
        # A decode rate needs the tokens to have been *delivered over time*.
        # Engines with speculative decoding or a short output can flush the
        # whole generation in a single burst; the timestamps then measure the
        # socket, not the model.  Live vLLM example: 8 tokens, a 0.036 ms span,
        # which divided out to 196,078 tok/s - a number that then went into a
        # report mean.  Withholding it is the honest answer.
        span_s = (token_times[-1] - token_times[0]) if len(token_times) > 1 else 0.0
        sample["decode_span_ms"] = round(span_s * 1000, 3)
        min_span_ms = MIN_STREAM_GAP_MS * max(len(token_times) - 1, 1)
        if len(token_times) < 2:
            sample["decode_unmeasurable"] = "single_flush"
        elif span_s * 1000 < min_span_ms:
            sample["decode_unmeasurable"] = "coalesced_flush"
        elif out_tok and out_tok > 1:
            tpot_ms = span_s * 1000 / (out_tok - 1)
            sample["tpot_ms"] = round(tpot_ms, 3)
            if tpot_ms > 0:
                sample["decode_tps"] = round(1000 / tpot_ms, 2)

        if prompt_tokens and ttft_s > 0:
            sample["prefill_tps"] = round(prompt_tokens / ttft_s, 2)

        # ---- was this request actually prefill-ing? --------------------
        # `prefill_tps = input/ttft` silently assumes TTFT is prefill time.
        # When the engine reports its own prefill duration, that assumption is
        # checkable: the difference is queueing, a retry, or a cache restore.
        # Publishing such a sample unqualified is how a ladder produces
        # "longer context is faster" and then a lone absurd point.
        srv_prefill_ms = server.get("server_prompt_ms")
        if srv_prefill_ms:
            queue_wait = ttft_s * 1000 - float(srv_prefill_ms)
            sample["queue_wait_ms"] = round(queue_wait, 1)
            threshold = max(QUEUE_CONTAMINATION_MIN_MS,
                            QUEUE_CONTAMINATION_MIN_SHARE * ttft_s * 1000)
            sample["prefill_contaminated"] = bool(queue_wait >= threshold)
        if len(token_times) > 2:
            diffs = [
                (token_times[i + 1] - token_times[i]) * 1000
                for i in range(len(token_times) - 1)
            ]
            diffs.sort()
            sample["itl_ms"] = round(sum(diffs) / len(diffs), 3)
            sample["itl_p95_ms"] = round(M.percentile(diffs, 0.95) or 0.0, 3)
    return sample


def scored_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Samples that belong in a statistic: warmups are not measurements.

    A warmup exists to load weights, fill a cache or pay a one-off cost; its
    timing describes the *first* request, not the steady state.  Averaging it
    in silently drags every mean towards a cold start, and the three tasks
    must agree about that - so the rule lives here, not in each task.
    """
    return [s for s in samples if not s.get("warmup")]


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Percentile summary over the successful *scored* samples."""
    scored = scored_samples(samples)
    ok_samples = [s for s in scored if s.get("ok")]
    keys = [
        "ttft_ms", "ttft_content_ms", "e2e_ms", "decode_window_ms",
        "tpot_ms", "decode_tps", "prefill_tps", "itl_ms", "itl_p95_ms",
        "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
        # server-reported, so the client's numbers can be cross-checked
        "server_prompt_ms", "server_prefill_tps", "server_decode_tps",
        "cache_hit_tokens", "draft_tokens", "draft_accepted",
        # the non-prefill slice of TTFT: how much of it was waiting
        "queue_wait_ms",
    ]
    stats = {k: M.summarize([s.get(k) for s in ok_samples]) for k in keys}
    return {
        "stats": stats,
        "ok_samples": ok_samples,
        "scored": len(scored),
        "warmup_excluded": len(samples) - len(scored),
    }


def input_identity_warning(samples: list[dict[str, Any]],
                           scope: str = "本次统计") -> str | None:
    """Warn when the samples behind one number did not use distinct inputs.

    This is the check that turns a confusing result into a diagnosis.  Engines
    cache a *prefix*, so re-sending the same text means the second request
    skips most of its prefill: measured on a real node, prefill went from
    252 tok/s (cold) to 2000 tok/s (cached) inside a single ladder step, and
    TTFT moved by 8x.  The mean of those two is not a property of the engine.
    """
    scored = [s for s in scored_samples(samples) if s.get("ok")]
    hashes = [s.get("input_hash") for s in scored if s.get("input_hash")]
    if len(hashes) < 2:
        return None
    duplicates = len(hashes) - len(set(hashes))
    if not duplicates:
        return None
    return (
        f"{scope}内有 {duplicates}/{len(hashes)} 个样本重复使用了同一段输入 —— "
        "有 prompt cache 的引擎会把后续请求的一部分从缓存里取出，"
        "TTFT 会被压低、Prefill 会被抬高，这个均值既不是冷启动也不是稳态。"
        "请让每个样本使用不重复的输入。"
    )


def input_length_warning(samples: list[dict[str, Any]],
                         scope: str = "本次统计") -> str | None:
    """Warn when the scored samples differ in input length.

    ``prefill_tps = input_tokens / ttft`` only compares across samples if the
    numerator is comparable, so a spread here invalidates the comparison.
    """
    inputs = [s.get("input_tokens") for s in scored_samples(samples)
              if s.get("ok") and s.get("input_tokens")]
    if len(inputs) < 2:
        return None
    lo, hi = min(inputs), max(inputs)
    mean = sum(inputs) / len(inputs)
    if hi - lo <= max(4, 0.10 * mean):
        return None
    return (f"{scope}内各样本输入长度不一致（{lo}–{hi} tokens），"
            "prefill 指标跨样本不可直接比较")


# Worst-first: a single "engine is gone" outranks any number of timeouts.
_KIND_RANK = {
    "unreachable": 0,
    "auth": 1,
    "too_long": 2,
    "server_error": 3,
    "timeout": 4,
    "unknown": 5,
}


def worst_kind(samples: list[dict[str, Any]]) -> str | None:
    """The most consequential failure kind among the samples, if any."""
    kinds = [s.get("error_kind") for s in samples if s.get("error_kind")]
    if not kinds:
        return None
    return min(kinds, key=lambda k: _KIND_RANK.get(k, 99))


def cache_warning(samples: list[dict[str, Any]]) -> str | None:
    """A warning when the engine says it served part of the prompt from cache.

    This is the honest version of a suspicious TTFT: instead of guessing from
    a number that looks too good, quote the engine's own accounting.  Returns
    ``None`` when there is nothing to report (including when the engine does
    not report timings at all, which is not evidence either way).
    """
    ok = [s for s in samples
          if s.get("ok") and s.get("cache_hit_tokens") is not None]
    if not ok:
        return None
    hit = [s for s in ok if (s.get("cache_hit_tokens") or 0) > 0]
    if not hit:
        return None
    worst = max(ok, key=lambda s: (s.get("cache_hit_tokens") or 0))
    share = None
    if worst.get("input_tokens"):
        share = (worst["cache_hit_tokens"] or 0) / worst["input_tokens"]
    pct = f"，最多的一次占输入 {share * 100:.0f}%" if share else ""
    return (
        f"引擎自报有 {len(hit)}/{len(ok)} 次请求命中了 prompt cache"
        f"（`cache_n` 最多 {(worst['cache_hit_tokens'] or 0):.0f} tokens{pct}）—— "
        "这些样本的 Prefill / TTFT **不是冷启动算力**，请用唯一输入或关闭缓存后重测"
    )


def prefill_contamination_warning(samples: list[dict[str, Any]],
                                  scope: str = "本次统计") -> str | None:
    """Warn when TTFT contained a wait that was not prefill.

    ``prefill_tps = input_tokens / ttft`` is only a throughput if TTFT *is*
    prefill.  Engines that report their own prefill time make that checkable,
    and the gap is not a rounding error: measured on a real llama.cpp fork, a
    128K step came back at 121,144 ms of client TTFT against 85,410 ms of
    engine-reported prefill.  The 35.7 s in between was the next request
    queueing behind a warmup the client had already abandoned - and the report
    said 1083 tok/s where the engine's own accounting says 1536.

    A contaminated sample is not wrong data to throw away: the engine's own
    numbers are still good.  It just cannot be read as a prefill rate.
    """
    bad = [s for s in scored_samples(samples)
           if s.get("ok") and s.get("prefill_contaminated")]
    if not bad:
        return None
    worst = max(bad, key=lambda s: (s.get("queue_wait_ms") or 0))
    wait = worst.get("queue_wait_ms") or 0.0
    ttft = worst.get("ttft_ms") or 0.0
    share = (wait / ttft * 100) if ttft else 0.0
    return (
        f"{scope}内有 {len(bad)}/{len(scored_samples(samples))} 个样本的 TTFT "
        f"里混入了非 prefill 的等待（最多 {wait / 1000:.1f}s，占 TTFT "
        f"{share:.0f}%）—— 这段时间不在引擎的 prefill 里，"
        "常见原因是上一次请求被中断后引擎仍在算、后续请求排队，"
        "也可能是重试或缓存恢复。这些样本的 `prefill_tps` 测的是等待而不是算力，"
        "请参考引擎自报的 prefill 时间或重测该档位。"
    )


def decode_coalesced_warning(samples: list[dict[str, Any]],
                             scope: str = "本次统计") -> str | None:
    """Warn when decode could not be measured because the stream was flushed.

    This is the counterweight to publishing a speed: when an engine delivers a
    whole short generation in one burst, the client's timestamps span
    microseconds and any rate derived from them is fiction.  Measured on a live
    vLLM node: the same engine gives a steady 135-152 tok/s at
    ``max_tokens >= 32`` and an apparent 196,078 tok/s at ``max_tokens = 8``.

    Silence here would be the dangerous outcome, because the withheld samples
    simply vanish from the mean and the remaining ones look fine.
    """
    scored = [s for s in scored_samples(samples) if s.get("ok")]
    bad = [s for s in scored if s.get("decode_unmeasurable")]
    if not bad:
        return None
    single = sum(1 for s in bad if s["decode_unmeasurable"] == "single_flush")
    spans = [s.get("decode_span_ms") for s in bad
             if s.get("decode_span_ms") is not None]
    span_txt = ""
    if spans:
        span_txt = f"（最紧的一次首末 token 只隔 {min(spans):.3f} ms）"
    outs = [s.get("output_tokens") for s in bad if s.get("output_tokens")]
    out_txt = f"，输出仅 {max(outs)} tokens" if outs else ""
    detail = ("整个输出被合并成一次 flush" if single
              else "多个事件挤在同一次读取里到达")
    return (
        f"{scope}内有 {len(bad)}/{len(scored)} 个样本的 decode 无法测量{span_txt}："
        f"{detail}{out_txt}，token 到达时间无法分辨，"
        "据此算出的 decode 速度描述的是网络刷新而不是模型。"
        "这些样本的 decode_tps 已置空（不计入均值）。"
        "该引擎需要更长的输出来暴露真实解码节奏，建议把 `max_tokens` 提到 64 以上。"
    )
