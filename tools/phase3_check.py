"""Phase 3 integration check: engine detection, metric normalisation, cache control.

Runs three layers of verification:

  A. in-process   - Prometheus parser, then detect() + get_metrics() against
                    every mock flavour, asserting the normalised schema.
  B. black-box    - the backend's /api/detect, to prove the API surfaces what
                    the adapters discovered.
  C. end-to-end   - a real benchmark run against the llama.cpp-vanilla mock
                    with prompt caching on vs off, proving cache_prompt
                    actually reaches the engine and changes the measurement.
  E. end-to-end   - server-reported timings, and the automatic switch to
                    unique input when the cache cannot be disabled.
  F. end-to-end   - vLLM specifics found on a live node: the `delta.reasoning`
                    field name, and prefix caching stated by the engine with
                    no API switch to turn it off.

Prerequisite: the backend must already be running on 127.0.0.1:8000.

Usage:
    python tools/phase3_check.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

API = "http://127.0.0.1:8000"
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

FLAVOUR_PORTS = {
    "llamacpp-vanilla": 9101,
    "llamacpp-fork": 9102,
    "vllm": 9103,
    "generic": 9104,
}

# Extra engines with behaviour that a plain flavour cannot express.  Kept
# separate so the flavour matrix above stays a clean "one engine each" table.
#   vllm-thinking: vLLM's `delta.reasoning` field + prefix caching that cannot
#                  be switched off (both were real failures on a live node)
EXTRA_MOCKS = {
    "vllm-thinking": {
        "port": 9105,
        "args": ["--flavor", "vllm",
                 "--thinking-field", "reasoning",
                 "--thinking-tokens", "24",
                 "--prefix-caching"],
    },
}

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def fetch(url: str, payload=None, timeout: float = 300.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    try:
        with NO_PROXY.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
async def _probe_port(port: int):
    """Detect the engine on ``port`` and read its normalised metrics."""
    from app.providers import detect
    from app.providers.detect import build_provider

    base = f"http://127.0.0.1:{port}"
    det = await detect(base, None, timeout_s=10.0, retries=0)
    pv = build_provider(base, None, family=det.family, detection=det)
    metrics = await pv.get_metrics()
    await pv.aclose()
    return det, metrics


# ---------------------------------------------------------------------- #
# A. in-process
# ---------------------------------------------------------------------- #
def layer_a() -> None:
    print("\n=== A. Prometheus parser ===")
    from app.prometheus import collapse, histogram_mean, parse_exposition

    text = """
# HELP foo A counter.
# TYPE foo counter
foo{a="1",b="x\\"y"} 1.5
foo{a="2",b="x\\"y"} 2.5
bar_bucket{le="0.1"} 3
bar_bucket{le="+Inf"} 4
bar_sum 9.0
bar_count 4.0
baz 7
""".strip()
    samples, types = parse_exposition(text)
    collapsed = collapse(samples)
    check(types.get("foo") == "counter", "parser reads # TYPE")
    check(collapsed.get("foo") == 4.0, "parser sums labelled series",
          f"foo={collapsed.get('foo')}")
    check(any(s.labels.get("b") == 'x"y' for s in samples),
          "parser unescapes label values")
    check(histogram_mean(collapsed, "bar") == 2.25,
          "histogram mean = _sum / _count", f"{histogram_mean(collapsed, 'bar')}")
    check(collapsed.get("baz") == 7, "parser reads a bare sample")

    print("\n=== A. info metrics (labels are the payload) ===")
    from app.prometheus import info_metric

    info_text = (
        '# TYPE vllm:cache_config_info gauge\n'
        'vllm:cache_config_info{enable_prefix_caching="True",'
        'kv_cache_size_tokens="1103309",kv_cache_max_concurrency="16.83",'
        'cache_dtype="fp8"} 1.0\n'
    )
    info_samples, _ = parse_exposition(info_text)
    cfg = info_metric(info_samples, "vllm:cache_config_info")
    check(cfg.get("enable_prefix_caching") == "True",
          "info metric labels are readable", str(cfg.get("enable_prefix_caching")))
    check(cfg.get("kv_cache_size_tokens") == "1103309",
          "info metric keeps string payloads as strings")
    check(collapse(info_samples).get("vllm:cache_config_info") == 1.0,
          "collapse alone would lose the info metric (hence the separate read)")

    print("\n=== A. reasoning delta field names ===")
    from app.providers.openai_compat import _delta_reasoning, _message_reasoning

    check(_delta_reasoning({"reasoning_content": "a"}) == "a",
          "llama.cpp/fork spelling `reasoning_content` is understood")
    check(_delta_reasoning({"reasoning": "b"}) == "b",
          "vLLM spelling `reasoning` is understood")
    check(_delta_reasoning({"content": "c"}) == "",
          "a plain content delta carries no reasoning")
    check(_delta_reasoning({"reasoning": ""}) == "",
          "an empty reasoning field is not treated as reasoning")
    check(_message_reasoning({"reasoning": "d"}) == "d",
          "non-streaming `message.reasoning` is understood too")

    print("\n=== A. adapter detection + normalised metrics ===")
    async def run_all():
        return {f: await _probe_port(p) for f, p in FLAVOUR_PORTS.items()}

    results = asyncio.run(run_all())

    # --- llamacpp-vanilla
    det, met = results["llamacpp-vanilla"]
    caps = det.capabilities
    check(det.family == "llamacpp", "vanilla: family=llamacpp", det.family)
    check(det.adapter == "LlamaCppProvider", "vanilla: adapter", det.adapter)
    check(det.confidence >= 0.9, "vanilla: high confidence", str(det.confidence))
    check(caps.props and caps.slots, "vanilla: /props + /slots detected")
    check(caps.tokenize, "vanilla: /tokenize detected (405-on-GET)")
    check(caps.native_completion, "vanilla: /completion detected")
    check(caps.cache_prompt_control, "vanilla: cache_prompt controllable")
    check(caps.thinking_control,
          "vanilla: thinking control inferred from chat_template_caps")
    norm = met["normalized"]
    # The mock's busy gauge is dynamic (0 when idle).  A hard-coded non-zero
    # value is not a harmless placeholder: the harness polls this gauge to wait
    # for the engine to drain after a failed sample, and a constant 1 makes it
    # spin until the timeout instead of returning.
    check(norm["requests_running"] == 0, "vanilla: requests_running (idle)",
          str(norm["requests_running"]))
    check(abs((norm["kv_cache_usage_ratio"] or 0) - 0.25) < 1e-6,
          "vanilla: kv_cache_usage_ratio", str(norm["kv_cache_usage_ratio"]))
    check(norm["prompt_tokens_total"] == 12000, "vanilla: prompt_tokens_total")
    check(norm["generation_tokens_total"] == 54000, "vanilla: generation_tokens_total")
    check(abs((norm["prompt_tokens_per_s"] or 0) - 500.0) < 0.01,
          "vanilla: derived prompt tok/s", str(norm["prompt_tokens_per_s"]))
    check(abs((norm["generation_tokens_per_s"] or 0) - 54.0) < 0.01,
          "vanilla: derived generation tok/s", str(norm["generation_tokens_per_s"]))

    # --- llamacpp-fork (mirrors the real engine: no /props, no /slots)
    det, met = results["llamacpp-fork"]
    caps = det.capabilities
    check(det.family == "llamacpp", "fork: family=llamacpp", det.family)
    check(not caps.props and not caps.slots,
          "fork: absent /props + /slots tolerated", f"props={caps.props} slots={caps.slots}")
    check(caps.metrics_prometheus, "fork: metrics still drive detection")
    check(caps.kv_cache_metrics, "fork: kv cache metrics detected")
    norm = met["normalized"]
    check(norm["kv_pool_positions"] == 32768, "fork: kv_pool_positions mapped")
    check(norm["requests_success_total"] == 90, "fork: fork-specific counter mapped")
    check(abs((norm["draft_accept_ratio"] or 0) - 0.75) < 1e-6,
          "fork: draft accept ratio derived", str(norm["draft_accept_ratio"]))
    check(any("fork" in n for n in det.notes),
          "fork: a note explains the missing native endpoints")

    # --- vllm
    det, met = results["vllm"]
    caps = det.capabilities
    check(det.family == "vllm", "vllm: family=vllm", det.family)
    check(det.adapter == "VLLMProvider", "vllm: adapter", det.adapter)
    check(not caps.props and not caps.slots, "vllm: no /props or /slots")
    check(caps.tokenize, "vllm: /tokenize detected")
    check(caps.request_latency_histograms, "vllm: latency histograms detected")
    norm = met["normalized"]
    # both series are rewritten to the engine's real state by the mock, so an
    # idle engine must read 0 - see the note on the vanilla assertion above
    check(norm["requests_running"] == 0,
          "vllm: num_requests_running (idle)", str(norm["requests_running"]))
    check(norm["requests_waiting"] == 0,
          "vllm: num_requests_waiting (idle)", str(norm["requests_waiting"]))
    check(abs((norm["kv_cache_usage_ratio"] or 0) - 0.42) < 1e-6,
          "vllm: gpu_cache_usage_perc -> ratio", str(norm["kv_cache_usage_ratio"]))
    check(norm["requests_success_total"] == 10,
          "vllm: labelled counter summed across finished_reason",
          str(norm["requests_success_total"]))
    check(abs((norm["ttft_seconds_mean"] or 0) - 0.35) < 1e-6,
          "vllm: server-side TTFT mean from histogram", str(norm["ttft_seconds_mean"]))
    check(abs((norm["e2e_seconds_mean"] or 0) - 2.5) < 1e-6,
          "vllm: server-side e2e mean from histogram", str(norm["e2e_seconds_mean"]))

    # --- generic
    det, met = results["generic"]
    check(det.family == "openai_compatible", "generic: family falls back", det.family)
    check(det.adapter == "OpenAICompatProvider", "generic: adapter", det.adapter)
    check(det.confidence < 0.5, "generic: low confidence reported", str(det.confidence))
    check(not met["available"], "generic: metrics correctly reported unavailable")

    print("\n=== A. vLLM series that were renamed across releases ===")
    from app.metrics_normalize import normalize

    renamed = normalize({
        # newer vLLM rename; the older name is absent on purpose
        "vllm:kv_cache_usage_perc": 0.37,
        "vllm:prefix_cache_queries_total": 400.0,
        "vllm:prefix_cache_hits_total": 90.0,
        "vllm:num_preemptions_total": 7.0,
        "vllm:request_prefill_kv_computed_tokens_sum": 4000.0,
        "vllm:request_prefill_kv_computed_tokens_count": 8.0,
    }, "vllm")
    check(abs((renamed["kv_cache_usage_ratio"] or 0) - 0.37) < 1e-6,
          "renamed `kv_cache_usage_perc` is mapped",
          str(renamed["kv_cache_usage_ratio"]))
    check(abs((renamed["prefix_cache_hit_ratio"] or 0) - 0.225) < 1e-6,
          "prefix cache hit ratio derived",
          str(renamed["prefix_cache_hit_ratio"]))
    check(renamed["preemptions_total"] == 7.0, "preemptions are surfaced")
    check(abs((renamed["prefill_kv_tokens_mean"] or 0) - 500.0) < 1e-6,
          "tokens actually computed in prefill (cache reuse made visible)",
          str(renamed["prefill_kv_tokens_mean"]))

    legacy = normalize({"vllm:gpu_cache_usage_perc": 0.42}, "vllm")
    check(abs((legacy["kv_cache_usage_ratio"] or 0) - 0.42) < 1e-6,
          "the older `gpu_cache_usage_perc` still works",
          str(legacy["kv_cache_usage_ratio"]))
    percent = normalize({"vllm:kv_cache_usage_perc": 42.0}, "vllm")
    check(abs((percent["kv_cache_usage_ratio"] or 0) - 0.42) < 1e-6,
          "a 0..100 style reading is still normalised to 0..1",
          str(percent["kv_cache_usage_ratio"]))


# ---------------------------------------------------------------------- #
# B. black-box through the backend
# ---------------------------------------------------------------------- #
def layer_b() -> None:
    print("\n=== B. backend /api/detect ===")
    expected_family = {
        "llamacpp-vanilla": "llamacpp",
        "llamacpp-fork": "llamacpp",
        "vllm": "vllm",
        # nothing identifiable -> the generic adapter, honestly reported as such
        "generic": "openai_compatible",
    }
    for flavor, port in FLAVOUR_PORTS.items():
        st, body = fetch(f"{API}/api/detect",
                         {"base_url": f"http://127.0.0.1:{port}"}, timeout=60)
        if st != 200:
            check(False, f"detect {flavor}", str(body)[:160])
            continue
        caps = body.get("capabilities", {})
        check(body["family"] == expected_family[flavor],
              f"detect {flavor}: family", body["family"])
        if flavor == "generic":
            # an unidentifiable engine must say so instead of inventing evidence
            check(not body.get("evidence") and body["confidence"] < 0.5,
                  "detect generic: low confidence with no fabricated evidence",
                  f"conf={body['confidence']} notes={len(body.get('notes', []))}")
        else:
            check(bool(body.get("evidence")),
                  f"detect {flavor}: evidence reported",
                  f"{len(body['evidence'])} item(s)")
        if flavor == "vllm":
            check(caps.get("request_latency_histograms") is True,
                  "detect vllm: histogram capability exposed")

    st, probe = fetch(f"{API}/api/probe",
                      {"base_url": f"http://127.0.0.1:{FLAVOUR_PORTS['llamacpp-fork']}"},
                      timeout=60)
    if st == 200:
        check(probe.get("adapter") == "llamacpp", "probe reports the chosen adapter",
              str(probe.get("adapter")))
        vals = (probe.get("model_param_values") or {}).get("mock-qwen3-flash") or {}
        check(vals.get("context_window") == 32768,
              "probe exposes context_window as a read-only param", str(vals.get("context_window")))
        check(vals.get("server_slots") == 4,
              "probe exposes server_slots from /health", str(vals.get("server_slots")))
        check(vals.get("server_kv_pool") == 32768,
              "probe exposes server_kv_pool", str(vals.get("server_kv_pool")))
    else:
        check(False, "probe against fork mock", str(probe)[:160])


# ---------------------------------------------------------------------- #
# C. end-to-end cache control
# ---------------------------------------------------------------------- #
def run_benchmark(port: int, overrides: dict, label: str,
                  timeout_s: float = 180.0) -> dict | None:
    perf = {"repeats": 2, "warmup": 1, "max_tokens": 16}
    perf.update(overrides)
    st, run = fetch(f"{API}/api/runs", {
        "base_url": f"http://127.0.0.1:{port}",
        "preset": "quick",
        "performance": perf,
    })
    if st != 200:
        check(False, f"{label}: create run", str(run)[:160])
        return None
    rid = run["run_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st, doc = fetch(f"{API}/api/runs/{rid}", timeout=30)
        if doc.get("status") not in ("queued", "running", "warmup", "testing",
                                     "connect", "pending"):
            return doc.get("result")
        time.sleep(1)
    return None


def layer_c() -> None:
    print("\n=== C. cache_prompt reaches the engine ===")
    port = FLAVOUR_PORTS["llamacpp-vanilla"]

    # `unique_prompt` defaults to ON (the whole tool defaults to cold-start
    # measurement), so a test that wants to *observe* prompt caching has to
    # turn uniqueness off explicitly.  Both sides get the same setting so the
    # comparison isolates the cache switch.
    on = run_benchmark(port, {"cache_prompt": True, "unique_prompt": False},
                       "cache on")
    off = run_benchmark(port, {"cache_prompt": False, "unique_prompt": False},
                        "cache off")
    if not on or not off:
        check(False, "cache comparison runs completed")
        return

    on_ttft = (on["result"]["primary"] or {}).get("ttft_ms")
    off_ttft = (off["result"]["primary"] or {}).get("ttft_ms")
    check(on_ttft is not None and off_ttft is not None,
          "both runs produced TTFT", f"on={on_ttft} off={off_ttft}")
    if on_ttft and off_ttft:
        check(off_ttft > on_ttft * 3,
              "disabling prompt cache raises TTFT as expected",
              f"{on_ttft:.1f}ms (cached) -> {off_ttft:.1f}ms (cold)")
        # the mock sleeps 200ms for a cold prefill and 10ms for a cache hit,
        # so a correct streaming implementation must land near those numbers
        check(on_ttft < 120, "cached TTFT is near the mock's 10ms hit path",
              f"{on_ttft:.1f}ms")
        check(off_ttft > 150, "cold TTFT is near the mock's 200ms prefill",
              f"{off_ttft:.1f}ms")

    opts = (on.get("config") or {}).get("options") or {}
    check(opts.get("cache_prompt") is True,
          "cache_prompt=True recorded in the run config", json.dumps(opts))

    check((off.get("capabilities") or {}).get("cache_prompt_control") is True,
          "capability gate: engine advertised cache control")


# ---------------------------------------------------------------------- #
# E. phase 6 prep: server-reported timings + the cache we cannot switch off
# ---------------------------------------------------------------------- #
def layer_e() -> None:
    print("\n=== E. server timings + auto unique input ===")
    # llamacpp-fork mirrors the real halogen engine: its /health says prompt
    # cache is on, and it has no native /completion to turn it off
    port = FLAVOUR_PORTS["llamacpp-fork"]

    doc = run_benchmark(port, {"unique_prompt": False}, "fork, cache uncontrolled")
    if not doc:
        return

    caps = doc.get("capabilities") or {}
    check(caps.get("prompt_cache_enabled") is True,
          "fork: engine reports prompt cache enabled")
    check(not caps.get("cache_prompt_control"),
          "fork: and reports no way to switch it off")

    warnings = doc.get("settings_warnings") or []
    check(any("不重复输入" in w for w in warnings),
          "a warning explains the automatic switch to unique input",
          "; ".join(warnings)[:160] or "(none)")
    cfg = doc.get("config") or {}
    check(cfg.get("unique_prompt") is True,
          "unique_prompt was forced on even though the client asked for off",
          str(cfg.get("unique_prompt")))

    samples = [s for s in (doc.get("samples") or []) if s.get("ok")]
    check(bool(samples), "the run produced samples", str(len(samples)))
    check(all(s.get("server_timings") for s in samples),
          "the engine's timings block was captured")
    check(any(s.get("server_prompt_ms") is not None for s in samples),
          "server-side prompt_ms was parsed",
          str([s.get("server_prompt_ms") for s in samples]))
    # The whole point: with unique input the engine's own cache counter must
    # read zero.  Without it, the mock would honour its cache and report the
    # full prompt as cached.
    check(all((s.get("cache_hit_tokens") or 0) == 0 for s in samples),
          "engine reports zero cache hits once the input is unique",
          str([s.get("cache_hit_tokens") for s in samples]))
    server = (doc.get("result") or {}).get("server") or {}
    check(server.get("available") is True and server.get("cache_hit_samples") == 0,
          "the result carries the server/client cross-check",
          json.dumps(server, ensure_ascii=False)[:140])

    # a repeated prompt with the cache un-switchable must be caught by the
    # engine's own accounting - this is the negative control for the above
    print("       (negative control: same prompt, cache on, unique off)")
    port2 = FLAVOUR_PORTS["llamacpp-vanilla"]
    cached = run_benchmark(port2, {"cache_prompt": True, "unique_prompt": False},
                           "vanilla cache on")
    if cached:
        cs = [s for s in (cached.get("samples") or []) if s.get("ok")]
        hits = [s.get("cache_hit_tokens") for s in cs]
        check(any((h or 0) > 0 for h in hits),
              "a real cache hit is visible in cache_n (control)",
              str(hits))
        warns = cached.get("data_warnings") or []
        check(any("cache" in w.lower() or "缓存" in w for w in warns),
              "and the run warns that prefill is not cold compute",
              "; ".join(warns)[:140] or "(none)")


# ---------------------------------------------------------------------- #
# D. phase 4: sampling options
# ---------------------------------------------------------------------- #
def layer_d() -> None:
    print("\n=== D. unique_prompt defeats caching at the content level ===")
    port = FLAVOUR_PORTS["llamacpp-vanilla"]

    # cache explicitly ON: a repeated prompt would normally be served from
    # cache, so only unique input can keep TTFT cold.  The "cached" run must
    # opt out of uniqueness explicitly - it defaults to ON now, and a test
    # cannot observe caching while uniqueness is doing its job.
    cached = run_benchmark(port, {"cache_prompt": True, "unique_prompt": False},
                           "cached")
    unique = run_benchmark(
        port, {"cache_prompt": True, "unique_prompt": True}, "unique input")
    if not cached or not unique:
        check(False, "unique_prompt comparison runs completed")
        return

    c_ttft = (cached["result"]["primary"] or {}).get("ttft_ms")
    u_ttft = (unique["result"]["primary"] or {}).get("ttft_ms")
    check(c_ttft is not None and u_ttft is not None,
          "both runs produced TTFT", f"cached={c_ttft} unique={u_ttft}")
    if c_ttft and u_ttft:
        check(u_ttft > c_ttft * 2,
              "unique input stays cold even with cache_prompt=true",
              f"{c_ttft:.1f}ms (cache hit) -> {u_ttft:.1f}ms (never equal)")
    check(all(s.get("unique_prompt") for s in unique["samples"]),
          "samples are flagged as using a unique prompt")

    # The mock caches by *prefix*, as real engines do.  A marker appended to
    # the end of a prompt would therefore still leave the body cacheable - a
    # mistake that was made for real and only caught by reading `cache_n`.
    # These two assertions are the regression test for it.
    c_hits = [s.get("cache_hit_tokens") for s in (cached.get("samples") or [])
              if s.get("ok")]
    u_hits = [s.get("cache_hit_tokens") for s in (unique.get("samples") or [])
              if s.get("ok")]
    check(any((h or 0) > 0 for h in c_hits),
          "repeating a prompt really is served from the prefix cache (control)",
          str(c_hits))
    check(all((h or 0) == 0 for h in u_hits),
          "the unique tag defeats the PREFIX cache, not just whole-prompt equality",
          str(u_hits))

    print("\n=== D. sampling options reach the request body ===")
    doc = run_benchmark(
        port,
        {"cache_prompt": False, "seed": 4242, "enable_thinking": False},
        "seed + thinking",
    )
    if not doc:
        check(False, "sampling options run completed")
        return
    opts = (doc.get("config") or {}).get("options") or {}
    check(opts.get("seed") == 4242, "seed forwarded as a top-level field",
          json.dumps(opts.get("seed")))
    nested = opts.get("chat_template_kwargs") or {}
    check(nested.get("enable_thinking") is False,
          "enable_thinking nested under chat_template_kwargs "
          "(top-level form is silently ignored by llama.cpp)",
          json.dumps(nested))
    check("enable_thinking" not in opts,
          "enable_thinking is NOT also sent top-level")
    check((doc.get("config") or {}).get("enable_thinking") is False,
          "enable_thinking recorded in the run config for the report")


# ---------------------------------------------------------------------- #
# F. vLLM-specific realities found on a live node
# ---------------------------------------------------------------------- #
def layer_f() -> None:
    """Two failures from a real vLLM 0.29 node, reproduced and pinned down.

    1. Reasoning arrives in ``delta.reasoning``.  Reading only
       ``reasoning_content`` made every chunk invisible, so a *successful*
       request was recorded as 0 output tokens and counted as a failure
       (success_rate 0) - the engine looked broken while working fine.
    2. ``vllm:cache_config_info`` states ``enable_prefix_caching=True`` and
       there is no API switch.  Without action, repeated inputs are partly
       served from cache and TTFT stops describing the engine.
    """
    print("\n=== F. vLLM reasoning field + un-switchable prefix caching ===")
    port = EXTRA_MOCKS["vllm-thinking"]["port"]

    det, met = asyncio.run(_probe_port(port))
    caps = det.capabilities
    check(det.family == "vllm", "thinking mock: family=vllm", det.family)
    check(caps.prompt_cache_enabled is True,
          "cache_config_info makes prompt caching explicit",
          str(caps.prompt_cache_enabled))
    check(not caps.cache_prompt_control,
          "and reports that no API switch exists to turn it off")
    check(any("enable_prefix_caching" in e for e in det.evidence),
          "the evidence quotes the engine's own declaration",
          "; ".join(det.evidence)[:150])
    check(any("kv_cache_size_tokens" in e or "KV cache" in e
              for e in det.evidence),
          "KV cache capacity is reported as evidence",
          "; ".join(det.evidence)[:150])
    check(any("prefix caching" in n for n in det.notes),
          "a note explains that prefill/TTFT will be optimistic",
          "; ".join(det.notes)[:150])

    norm = met["normalized"]
    check(abs((norm["kv_cache_usage_ratio"] or 0) - 0.37) < 1e-6,
          "renamed kv gauge read from a live adapter",
          str(norm["kv_cache_usage_ratio"]))
    check(abs((norm["prefix_cache_hit_ratio"] or 0) - 0.225) < 1e-6,
          "prefix cache hit ratio read from a live adapter",
          str(norm["prefix_cache_hit_ratio"]))
    check(norm["preemptions_total"] == 7.0, "preemptions read from a live adapter")

    # the engine-stated KV capacity must reach the read-only param block
    st, probe = fetch(f"{API}/api/probe", {"base_url": f"http://127.0.0.1:{port}"},
                      timeout=60)
    if st == 200:
        vals = (probe.get("model_param_values") or {}).get("mock-qwen3-flash") or {}
        check(vals.get("server_kv_pool") == 1103309,
              "KV pool size exposed as a read-only param", str(vals.get("server_kv_pool")))
        check(abs((vals.get("server_kv_concurrency") or 0) - 16.835164835164836) < 1e-6,
              "KV-derived concurrency ceiling exposed", str(vals.get("server_kv_concurrency")))

        # vLLM declares no thinking controls; the behavioural probe must find
        # them anyway, or `enable_thinking` gets stripped and a reasoning model
        # produces no visible answer at all.
        tp = probe.get("thinking_probe") or {}
        check(tp.get("supported") is True,
              "thinking control discovered by measurement, not declaration",
              json.dumps(tp, ensure_ascii=False)[:170])
        check((tp.get("baseline_reasoning_chars") or 0) > 0,
              "the baseline request really did produce reasoning",
              str(tp.get("baseline_reasoning_chars")))
        check((tp.get("off_reasoning_chars") or 0) == 0,
              "and disabling it removed the reasoning",
              str(tp.get("off_reasoning_chars")))
        check(probe.get("capabilities", {}).get("thinking_control") is True,
              "the measured capability reaches the API response")
        det_caps = (probe.get("detection") or {}).get("capabilities") or {}
        check(det_caps.get("thinking_control") is True,
              "and supersedes the earlier static guess in the detection block")
    else:
        check(False, "probe against the thinking mock", str(probe)[:160])

    # ---- the regression test for the invisible-reasoning bug ----------
    # max_tokens 16 < 24 thinking tokens: the whole budget goes to reasoning
    # and `content` stays empty, exactly like the live node.
    doc = run_benchmark(port, {"max_tokens": 16, "warmup": 0, "repeats": 1},
                        "reasoning-only", timeout_s=120)
    if not doc:
        check(False, "the reasoning-only run completed")
        return
    samples = doc.get("samples") or []
    check(bool(samples), "the run produced a sample")
    s = samples[0] if samples else {}
    check(s.get("ok") is True,
          "a reasoning-only response is a SUCCESS, not a failure",
          f"ok={s.get('ok')} error={s.get('error')!r}")
    check((s.get("output_tokens") or 0) == 16,
          "server-reported output tokens are kept", str(s.get("output_tokens")))
    check((s.get("reasoning_chars") or 0) > 0,
          "reasoning delivered via `delta.reasoning` is counted",
          f"{s.get('reasoning_chars')} chars")
    check((s.get("content_chars") or 0) == 0,
          "and there really was no visible content", str(s.get("content_chars")))
    check(s.get("finish_reason") == "length", "finish_reason=length recorded",
          str(s.get("finish_reason")))
    check((doc.get("result") or {}).get("samples_ok") == 1,
          "samples_ok counts it (success_rate must not read 0)",
          str((doc.get("result") or {}).get("samples_ok")))
    check((s.get("ttft_ms") or 0) > 0, "TTFT was measured", str(s.get("ttft_ms")))
    check(s.get("ttft_content_ms") is None,
          "ttft_content stays null when no content ever arrives")

    # ---- and a run with budget left over does produce content ---------
    doc2 = run_benchmark(port, {"max_tokens": 40, "warmup": 0, "repeats": 1},
                         "reasoning + content", timeout_s=120)
    if doc2:
        s2 = (doc2.get("samples") or [{}])[0]
        check((s2.get("content_chars") or 0) > 0,
              "with budget left over, visible content is counted",
              f"{s2.get('content_chars')} chars")
        check(s2.get("ttft_content_ms") is not None,
              "and ttft_content is measured separately from ttft",
              str(s2.get("ttft_content_ms")))
        check((s2.get("reasoning_chars") or 0) > 0,
              "reasoning is still counted alongside content")

    # run A must have warned *before* measuring that thinking would eat the
    # budget: finding that out from an empty content field is too late
    warns_a = doc.get("settings_warnings") or []
    check(any("思考链" in w for w in warns_a),
          "a pre-run warning flags the thinking budget",
          "; ".join(warns_a)[:180] or "(none)")

    # ---- the payoff: the switch must actually work end to end ----------
    doc3 = run_benchmark(port, {"max_tokens": 16, "warmup": 0, "repeats": 1,
                                "enable_thinking": False},
                         "thinking off", timeout_s=120)
    if doc3:
        s3 = (doc3.get("samples") or [{}])[0]
        check((s3.get("content_chars") or 0) > 0,
              "enable_thinking=false lets a small budget produce content",
              f"content={s3.get('content_chars')} reasoning={s3.get('reasoning_chars')}")
        check((s3.get("reasoning_chars") or 0) == 0,
              "and removes the reasoning entirely")
        opts3 = (doc3.get("config") or {}).get("options") or {}
        check((opts3.get("chat_template_kwargs") or {}).get("enable_thinking")
              is False,
              "the switch survived the capability gate",
              json.dumps(opts3, ensure_ascii=False)[:140])
        check(s3.get("ttft_content_ms") is not None,
              "so ttft_content is measurable even with a 16-token budget")

    # ---- prefix caching must trigger the automatic unique-input switch --
    # Opt out explicitly: the default is already unique, and the interesting
    # behaviour is that a *request* to reuse one input is refused on an engine
    # whose cache cannot be switched off.
    doc4 = run_benchmark(port, {"max_tokens": 16, "warmup": 0, "repeats": 2,
                                "unique_prompt": False},
                         "uniqueness refused", timeout_s=120)
    if doc4:
        check((doc4.get("config") or {}).get("unique_prompt") is True,
              "unique_prompt=false is overridden for an un-switchable cache",
              str((doc4.get("config") or {}).get("unique_prompt")))
        warns4 = doc4.get("settings_warnings") or []
        check(any("不重复输入" in w for w in warns4),
              "and the reason is stated in the run warnings",
              "; ".join(warns4)[:180] or "(none)")
        s4 = [s for s in (doc4.get("samples") or []) if s.get("ok")]
        hashes = [s.get("input_hash") for s in s4 if s.get("input_hash")]
        check(len(set(hashes)) == len(hashes),
              "so no two samples shared an input after all",
              f"{len(hashes)} samples, {len(set(hashes))} distinct")
    hits = [x.get("cache_hit_tokens") for x in samples]
    check(all((h or 0) == 0 for h in hits),
          "so the engine reports no cache hits after the first request",
          str(hits))


# ---------------------------------------------------------------------- #
def main() -> int:
    procs: list[subprocess.Popen] = []
    try:
        for flavor, port in FLAVOUR_PORTS.items():
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(HERE, "mock_engine.py"),
                 "--flavor", flavor, "--port", str(port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        for name, spec in EXTRA_MOCKS.items():
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(HERE, "mock_engine.py"),
                 "--port", str(spec["port"]), *spec["args"]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        # wait for readiness
        ready = {**{k: v for k, v in FLAVOUR_PORTS.items()},
                 **{k: v["port"] for k, v in EXTRA_MOCKS.items()}}
        for flavor, port in ready.items():
            for _ in range(50):
                st, _body = fetch(f"http://127.0.0.1:{port}/v1/models", timeout=3)
                if st == 200:
                    break
                time.sleep(0.2)
            else:
                check(False, f"mock {flavor} became ready")
                return 1
        print(f"started {len(procs)} mock engine(s)")

        layer_a()
        layer_b()
        layer_c()
        layer_d()
        layer_e()
        layer_f()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print("")
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL PHASE 3 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
