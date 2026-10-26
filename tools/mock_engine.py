"""Configurable fake inference engine, for adapter and detection tests.

A remote engine is not always reachable, and even when it is, we cannot make
it pretend to be vLLM.  This stdlib-only server can be launched in one of
four flavours so that engine fingerprinting and per-engine behaviour are
covered by deterministic tests:

    llamacpp-vanilla  /health {"status":"ok"}, /props, /slots, /tokenize,
                      /completion, llamacpp:* metrics
    llamacpp-fork     rich /health, **no** /props or /slots, llamacpp:* plus
                      fork-specific series  (this mirrors the real engine)
    vllm              empty-200 /health, vllm:* metrics with labels and
                      latency histograms, /tokenize, no /props or /slots
    generic           only /v1/models, /v1/chat/completions and /v1/completions

It also emulates prompt caching honestly: a repeated prompt with
``cache_prompt`` left enabled returns a near-zero TTFT, while
``cache_prompt: false`` always pays the cold prefill.  That is what lets the
test suite prove the cache-disable path actually works.

Usage:
    python tools/mock_engine.py --flavor llamacpp-fork --port 8731
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_ID = "mock-qwen3-flash"
CONTEXT = 32768
MAX_OUTPUT_CAP = 8192
DEFAULT_OUTPUT = 2048

TTFT_COLD_S = 0.20          # honest prefill cost
TTFT_CACHED_S = 0.01        # what a prompt-cache hit looks like
TOKEN_INTERVAL_S = 0.02

_seen_prompts: set[str] = set()
#: recent ``(token_count, text)`` pairs, so /detokenize can return the real
#: text a token count came from instead of a placeholder
_tokenized: list[tuple[int, str]] = []
_lock = threading.Lock()

# ---------------------------------------------------------------------- #
# context behaviour
# ---------------------------------------------------------------------- #
# The client builds inputs to an exact token count using /tokenize, so the
# mock's tokenizer defines what "N tokens" means here.  Same rule as the
# /tokenize handler below, so the two can never disagree.
CHARS_PER_TOKEN = 3

# Two distinct ways to say "too long", because a benchmark must not confuse
# them (see app/errors.classify_error):
#   LIMITS["enforce"]  the engine answers 400 -> a real capacity boundary
#   LIMITS["die"]      the engine drops the connection and the whole server
#                      stops, mirroring an OOM restart observed on real
#                      hardware under a 128K prefill
LIMITS: dict[str, int | None] = {"enforce": None, "die": None, "die_after": None}

# Some engines expose a tokenizer, some do not (the real halogen fork does
# not).  Flipping this off is what exercises the estimated/calibrated input
# path in the Context benchmark.
NO_TOKENIZE = False

#: how many leading output tokens are "thinking" rather than visible content
THINKING_TOKENS = 0
#: which delta field the thinking goes into.  Engines disagree: llama.cpp
#: forks use ``reasoning_content``, vLLM uses ``reasoning``.  Reading the wrong
#: one makes every chunk invisible, which turns a successful request into a
#: recorded failure - a real bug that this switch exists to reproduce.
THINKING_FIELD = "reasoning_content"
#: emit ``vllm:cache_config_info``: vLLM *states* its cache setup this way
#: (``enable_prefix_caching="True"``), unlike llama.cpp which nests it in
#: ``/health``.
PREFIX_CACHING = False

#: How many output tokens one SSE event carries.  This is deliberately a
#: separate knob from ``TOKEN_INTERVAL_S``: a real engine's *granularity* is
#: not its *speed*.  vLLM with speculative decoding produces ~2.3 tokens per
#: scheduler step, and a client that assumes "one stream event = one token"
#: derives both the rate and the per-token latency wrongly.
TOKENS_PER_CHUNK = 1

#: Compute the whole generation, then flush it as one burst (no sleep between
#: events).  This is what a real vLLM does for a *short* output: measured on a
#: live node, 8 tokens arrived with a first-to-last span of 0.036 ms while
#: ``usage`` still said 8 - which divided out to 196,078 tok/s.  Modelling it
#: is the only way to keep the harness honest, because the mock's default
#: (one token per event, evenly paced) can never expose that failure.
COALESCE = False

#: Drop the connection mid-prefill after this many seconds, *without* stopping
#: the work.  A real llama.cpp fork did exactly this: a 128K warmup died on
#: `ReadError` at 50 s while its prefill still needed ~85 s, so the engine kept
#: the slot busy and the *next* request sat behind it.  The client's TTFT came
#: out 41.8% above the engine's own `prompt_ms` (121,144 vs 85,410 ms), which
#: turned into a prefill rate of 1083 tok/s instead of 1536.
ABANDON_AFTER_S: float | None = None
#: Abandon only this many long requests, then behave normally - the real fork
#: was fine by the time the next sample arrived.  Abandoning *every* request
#: would produce no successful sample at all, which tests nothing.
ABANDON_TIMES = 1
#: Abandon only requests whose input is longer than this.  The real fork's
#: *small* steps were fine; it was the 128K request that died.  Without this the
#: single permitted abandonment gets eaten by a calibration probe and the test
#: never reaches the case it was written for.
ABANDON_ABOVE_TOKENS = 0
#: Override the cold-prefill cost so a test can create a queue wait long enough
#: to be measurable without waiting minutes (the default 0.2 s is nothing).
PREFILL_COLD_S = 0.20
_abandoned = 0

#: One request at a time: a new request waits until whatever is currently
#: running has finished.  Without this the mock serves requests in parallel
#: threads, so `requests_processing` would be a *decoration* that never costs
#: anyone any time - and the whole point of this mode is that a queued request
#: really does absorb the wait into its TTFT, the way a single compute queue
#: does on real hardware.
SERIALIZE = False
#: safety bound for the serialized wait, so a stuck test cannot hang forever
SERIALIZE_MAX_WAIT_S = 120.0

#: How many requests the engine runs at once.  0 = unlimited.  Real vLLM has
#: ``--max-num-seqs``; this knob exists because "the client kept 32 requests in
#: flight" and "the engine ran 16 and queued 16" are indistinguishable from the
#: client side - both look like 32 in flight - and only the engine's own gauges
#: can tell them apart.  Measured on a live node: nominal 32 showed
#: ``num_requests_running`` max **16** with ``num_requests_waiting`` max **16**,
#: so half of the reported parallelism was queueing.
MAX_RUNNING = 0

#: Deadlines of jobs whose *simulated work* has not finished.  A list rather
#: than a boolean because a client that disconnects does not stop an engine:
#: the work continues and the next request queues behind it.
_work_deadlines: list[float] = []
#: handlers attached right now, and handlers blocked waiting for a slot
_active = 0
_waiting = 0
_slot_cv = threading.Condition()


def _live_work() -> int:
    global _work_deadlines
    now = time.time()
    _work_deadlines = [d for d in _work_deadlines if d > now]
    return len(_work_deadlines)


def _engine_busy() -> bool:
    """Is the (simulated) engine still working on something?"""
    return _live_work() > 0


def _engine_running() -> int:
    """Requests the engine is working on, including abandoned ones.

    A client hanging up does not cancel a prefill, so a job whose handler is
    gone still occupies the engine.  Reporting plain ``_active`` would let a
    client observe an idle engine that is in fact busy - which is the exact
    failure `--abandon-after` was written to model.
    """
    return max(_active, _live_work())


def _acquire_slot() -> None:
    """Take a compute slot, counting the time spent waiting for one."""
    global _active, _waiting
    with _slot_cv:
        if MAX_RUNNING > 0 and _active >= MAX_RUNNING:
            _waiting += 1
            try:
                while _active >= MAX_RUNNING:
                    _slot_cv.wait(0.05)
            finally:
                _waiting = max(0, _waiting - 1)
        _active += 1


def _release_slot() -> None:
    global _active
    with _slot_cv:
        _active = max(0, _active - 1)
        _slot_cv.notify_all()

#: counts completed *generation* requests, for `--die-after`
_requests_served = 0


def _engine_busy() -> bool:
    """Is the (simulated) engine still working on something?"""
    return _live_work() > 0


#: Set MOCK_DEBUG=1 to trace the abandonment/cache decisions on stderr.  These
#: are exactly the branches a test asserts on, and "the test failed but every
#: input looks right" is otherwise unanswerable from the outside.
DEBUG = False


def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[mock] {msg}", file=sys.stderr, flush=True)


def _mark_busy(seconds: float) -> None:
    """Reserve the engine for ``seconds`` of simulated work.

    Deliberately independent of whether the client is still listening: an
    engine that has started a 128K prefill keeps working even after the socket
    dies, and that is what makes the next request queue.
    """
    _work_deadlines.append(time.time() + max(seconds, 0.0))


def _with_busy_gauge(body: str) -> str:
    """Rewrite the hard-coded gauges with the engine's real state.

    Both families are covered because both are *lies* otherwise, in a way that
    costs a test real time: vLLM's series shipped hard-coded at
    ``num_requests_running 2.0``, so a client waiting for "idle" polled for the
    full timeout instead of returning.  A gauge that cannot change is worse
    than a missing one.

    They must be **counts**, not a busy flag.  "Busy" cannot distinguish eight
    requests running from two running with six queued, and that distinction is
    the whole question a concurrency report has to answer.
    """
    running = _engine_running()
    waiting = _waiting
    out: list[str] = []
    for line in body.splitlines():
        if line.startswith("llamacpp:requests_processing "):
            out.append(f"llamacpp:requests_processing {running}")
        elif line.startswith("llamacpp:requests_deferred "):
            out.append(f"llamacpp:requests_deferred {waiting}")
        elif line.startswith("vllm:num_requests_running{"):
            head, _, _ = line.rpartition(" ")
            out.append(f"{head} {running}.0")
        elif line.startswith("vllm:num_requests_waiting{"):
            head, _, _ = line.rpartition(" ")
            out.append(f"{head} {waiting}.0")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def thinking_enabled(payload: dict[str, Any]) -> bool:
    """Honour the thinking switches, the way a real chat template does.

    Verified against a live vLLM node: ``chat_template_kwargs.enable_thinking
    = false`` and a top-level ``reasoning_effort: "none"`` both suppress the
    reasoning, while a *top-level* ``enable_thinking`` is silently ignored.
    Modelling that asymmetry is what makes the capability probe testable.
    """
    ctk = payload.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        if ctk.get("enable_thinking") is False:
            return False
        effort = str(ctk.get("reasoning_effort") or "").lower()
        if effort in ("none", "off"):
            return False
    effort = str(payload.get("reasoning_effort") or "").lower()
    if effort in ("none", "off"):
        return False
    return True


def common_prefix_tokens(a: str, b: str) -> int:
    """How many leading tokens two prompts share.

    Real engines cache a *prefix*, not a whole prompt, which is why appending
    a unique marker to the end of a prompt does not defeat the cache at all -
    everything before the marker still matches.  The mock has to model this or
    it cannot catch that mistake.
    """
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i // CHARS_PER_TOKEN


#: a prefix at least this long is served from cache instead of recomputed
CACHE_MIN_PREFIX_TOKENS = 32
#: real engines snapshot their cache on an alignment boundary (the halogen
#: fork reports `prompt_cache.snapshot_align: 64`), so a partial match shorter
#: than one block cannot be reused at all.  Modelling that is what keeps
#: `cache_n` meaningful: the chat template alone always shares a few tokens,
#: and without an alignment floor every unique prompt would look "cached".
CACHE_ALIGN_TOKENS = 16


# ---------------------------------------------------------------------- #
# metric payloads
# ---------------------------------------------------------------------- #
LLAMACPP_METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 12000
# HELP llamacpp:prompt_seconds_total Prompt process time.
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 24.0
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 54000
# HELP llamacpp:tokens_predicted_seconds_total Predict process time.
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 1000.0
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s. (since the last scrape)
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 500.0
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s. (since the last scrape)
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 54.0
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
# HELP llamacpp:kv_cache_tokens KV-cache tokens
# TYPE llamacpp:kv_cache_tokens gauge
llamacpp:kv_cache_tokens 2048
# HELP llamacpp:kv_cache_usage_ratio KV-cache usage. 1 means 100 percent usage.
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.25
"""

FORK_EXTRA_METRICS = """\
# HELP halogen:requests_total Requests completed.
# TYPE halogen:requests_total counter
halogen:requests_total 90
# HELP halogen:prompt_tokens_cached_total Prompt tokens the prompt cache covered
# TYPE halogen:prompt_tokens_cached_total counter
halogen:prompt_tokens_cached_total 6000
# HELP halogen:draft_tokens_total Tokens proposed by the draft head
# TYPE halogen:draft_tokens_total counter
halogen:draft_tokens_total 4000
# HELP halogen:draft_tokens_accepted_total Of those, accepted.
# TYPE halogen:draft_tokens_accepted_total counter
halogen:draft_tokens_accepted_total 3000
# HELP halogen:kv_pool_positions The KV pool, in positions.
# TYPE halogen:kv_pool_positions gauge
halogen:kv_pool_positions 32768
"""

VLLM_METRICS = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="mock-qwen3-flash"} 2.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="mock-qwen3-flash"} 3.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="mock-qwen3-flash"} 10000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="mock-qwen3-flash"} 20000.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="mock-qwen3-flash"} 5.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="mock-qwen3-flash"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="mock-qwen3-flash"} 3.5
vllm:time_to_first_token_seconds_count{model_name="mock-qwen3-flash"} 10.0
# HELP vllm:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{model_name="mock-qwen3-flash"} 1.2
vllm:time_per_output_token_seconds_count{model_name="mock-qwen3-flash"} 10.0
# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_sum{model_name="mock-qwen3-flash"} 25.0
vllm:e2e_request_latency_seconds_count{model_name="mock-qwen3-flash"} 10.0
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="mock-qwen3-flash"} 8.0
vllm:request_success_total{finished_reason="length",model_name="mock-qwen3-flash"} 2.0
"""

# vLLM renamed this gauge across releases.  A real engine publishes one or
# the other, never both, and the rename is exactly the kind of thing that
# silently yields `None` - so the two mock instances cover one name each:
# the plain `vllm` flavour uses the legacy name, `--prefix-caching` uses the
# new one.  The adapter must understand both.
VLLM_KV_GAUGE_LEGACY = """\
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="mock-qwen3-flash"} 0.42
"""

# Emitted only with `--prefix-caching`.  Real vLLM states its cache setup as
# an ``*_info`` metric, so the labels are the payload and the value is a
# constant 1.0 - which is exactly why a naive "sum the labels away" parse
# cannot see it.  The numbers mirror the node this was written against.
VLLM_PREFIX_CACHE_METRICS = """\
# HELP vllm:cache_config_info Information about the cache configuration.
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="800",cache_dtype="fp8",enable_prefix_caching="True",gpu_memory_utilization="0.97",kv_cache_max_concurrency="16.835164835164836",kv_cache_size_tokens="1103309",num_gpu_blocks="1532"} 1.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="mock-qwen3-flash"} 0.37
# HELP vllm:prefix_cache_queries_total Prefix cache queries.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="mock-qwen3-flash"} 400.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model_name="mock-qwen3-flash"} 90.0
# HELP vllm:num_preemptions_total Number of preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="mock-qwen3-flash"} 7.0
# HELP vllm:request_prefill_kv_computed_tokens Histogram of tokens computed in prefill.
# TYPE vllm:request_prefill_kv_computed_tokens histogram
vllm:request_prefill_kv_computed_tokens_sum{model_name="mock-qwen3-flash"} 4000.0
vllm:request_prefill_kv_computed_tokens_count{model_name="mock-qwen3-flash"} 8.0
"""


# ---------------------------------------------------------------------- #
# handler
# ---------------------------------------------------------------------- #
class MockEngine(BaseHTTPRequestHandler):
    flavor = "generic"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence
        return

    # ---- helpers -----------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message,
                                      "type": "not_found_error"}})

    # ---- routing -----------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        f = self.flavor

        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "owned_by": "mock",
                "created": 0, "max_model_len": CONTEXT,
                "context_length": CONTEXT,
                "meta": {"n_ctx_train": CONTEXT, "n_ctx": CONTEXT},
                "max_tokens_cap": MAX_OUTPUT_CAP,
                "max_tokens_default": DEFAULT_OUTPUT,
            }]})

        if path == "/health":
            if f in ("llamacpp-vanilla",):
                return self._json(200, {"status": "ok"})
            if f == "llamacpp-fork":
                return self._json(200, {
                    "status": "ok", "model": MODEL_ID,
                    "context": CONTEXT, "slots": 4, "slot_ctx": CONTEXT,
                    "kv_pool_positions": CONTEXT, "in_flight": 0, "queued": 0,
                    "version": {"api": "9.9.9", "engine": "9.9.9", "match": True},
                    "drafters_available": ["serial", "mtp"],
                    "drafter_default": "mtp", "drafter_weights_loaded": True,
                    "prompt_cache": {"enabled": True, "cap_mb": 110},
                    "chat_template": {"thinking_control": True, "probe": "passed"},
                    "tool_calls": {"streaming": True},
                    "vision": {"enabled": False},
                    "supported": ["reasoning_effort", "enable_thinking"],
                })
            if f == "vllm":
                # vLLM answers 200 with an EMPTY body - that is the fingerprint
                return self._send(200, b"", "application/json")
            return self._error(404, "Not Found")

        if path == "/metrics":
            if f == "vllm":
                if PREFIX_CACHING:
                    body = VLLM_METRICS + VLLM_PREFIX_CACHE_METRICS
                else:
                    body = VLLM_METRICS + VLLM_KV_GAUGE_LEGACY
                return self._send(200, _with_busy_gauge(body).encode(),
                                  "text/plain")
            if f == "llamacpp-vanilla":
                return self._send(200, _with_busy_gauge(LLAMACPP_METRICS).encode(),
                                  "text/plain")
            if f == "llamacpp-fork":
                body = _with_busy_gauge(LLAMACPP_METRICS + FORK_EXTRA_METRICS)
                return self._send(200, body.encode(), "text/plain")
            return self._error(404, "Not Found")

        if path == "/props":
            if f == "llamacpp-vanilla":
                # mirrors a real llama.cpp build: sampler params nested under
                # "params", chat-template capabilities in their own dict
                return self._json(200, {
                    "default_generation_settings": {
                        "n_ctx": 4096,
                        "params": {"seed": -1, "temperature": 0.8,
                                   "top_p": 0.95, "top_k": 40,
                                   "n_predict": -1, "cache_prompt": True},
                    },
                    "total_slots": 2,
                    "model_path": "/models/mock-qwen3-flash.gguf",
                    "model_alias": "/models/mock-qwen3-flash.gguf",
                    "build_info": "b4200-abcdef",
                    "modalities": {"vision": False, "video": False,
                                   "audio": False},
                    "chat_template_caps": {
                        "supports_reasoning_effort": True,
                        "supports_preserve_reasoning": True,
                        "supports_tool_calls": True,
                        "supports_system_role": True,
                    },
                    "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
                    "endpoint_props": True,
                    "endpoint_slots": True,
                    "endpoint_metrics": True,
                })
            return self._error(404, "Not Found")

        if path == "/slots":
            if f == "llamacpp-vanilla":
                return self._json(200, [
                    {"id": 0, "state": 1, "n_ctx": 4096, "is_processing": False,
                     "n_past": 0, "prompt_hash": 12345,
                     "prompt": "SHOULD BE TRIMMED AWAY"},
                    {"id": 1, "state": 0, "n_ctx": 4096, "is_processing": False,
                     "n_past": 0, "prompt_hash": 0, "prompt": ""},
                ])
            return self._error(404, "Not Found")

        # routes that exist but are POST-only -> 405 proves existence
        if path in ("/tokenize", "/detokenize", "/completion"):
            if f == "llamacpp-vanilla" and path in ("/tokenize", "/detokenize",
                                                    "/completion"):
                return self._error(405, "Method Not Allowed")
            if f == "vllm" and path == "/tokenize":
                return self._error(405, "Method Not Allowed")
            if f == "llamacpp-fork" and path in ("/v1/completions",):
                return self._error(405, "Method Not Allowed")
            return self._error(404, "Not Found")

        if path in ("/v1/completions", "/v1/chat/completions"):
            # llama.cpp style: GET on a POST-only route
            return self._error(405, "Method Not Allowed")

        return self._error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        f = self.flavor

        if path in ("/v1/chat/completions", "/v1/completions"):
            return self._complete()

        if path == "/completion":
            if f == "llamacpp-vanilla":
                return self._native_completion()
            return self._error(404, "Not Found")

        if path == "/tokenize":
            if not NO_TOKENIZE and f in ("llamacpp-vanilla", "llamacpp-fork",
                                         "vllm"):
                payload = self._read_json()
                text = str(payload.get("content") or payload.get("prompt") or "")
                n = estimate_tokens(text)
                with _lock:
                    _tokenized.append((n, text))
                    del _tokenized[:-32]          # keep a short history only
                return self._json(200, {"tokens": list(range(1, n + 1)),
                                        "count": n})
            return self._error(404, "Not Found")

        if path == "/detokenize":
            if not NO_TOKENIZE and f in ("llamacpp-vanilla", "llamacpp-fork",
                                         "vllm"):
                payload = self._read_json()
                ids = payload.get("tokens") or []
                k = len(ids)
                # Faithful-enough inverse: the ids stand for the first k
                # chunks of a text we were shown, so return that slice.  The
                # earlier version returned `"x" * 3k`, which re-tokenized to
                # the right *count* but threw the actual text away - including
                # the leading uniqueness marker that the whole prefix-cache
                # argument depends on.  A mock that loses the prefix would
                # hide exactly the bug it exists to catch.
                with _lock:
                    history = list(_tokenized)
                pick = next((t for c, t in history if c >= k), None)
                body = pick[:k * CHARS_PER_TOKEN] if pick is not None else \
                    "x" * (k * CHARS_PER_TOKEN)
                return self._json(200, {"content": body, "prompt": body})
            return self._error(404, "Not Found")

        if path == "/v1/completions":
            return self._complete()

        return self._error(404, "Not Found")

    # ---- generation --------------------------------------------------
    def _prompt_key(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if messages:
            return json.dumps(messages, sort_keys=True)
        return str(payload.get("prompt", ""))

    def _prompt_text(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if messages:
            return "".join(str(m.get("content") or "") for m in messages)
        return str(payload.get("prompt", ""))

    def _complete(self) -> None:
        global _requests_served
        payload = self._read_json()
        n_in = estimate_tokens(self._prompt_text(payload))

        die_after = LIMITS.get("die_after")
        if die_after is not None:
            with _lock:
                _requests_served += 1
                over = _requests_served > die_after
            if over:
                # the engine dies with load, not with length: used to test that
                # a concurrency sweep aborts instead of reporting throughput
                # for a server that is no longer there
                return self._crash(n_in, die_after)

        die = LIMITS.get("die")
        if die is not None and n_in > die:
            return self._crash(n_in, die)

        enforce = LIMITS.get("enforce")
        if enforce is not None and n_in > enforce:
            # a clean refusal: the engine is alive and says "no"
            return self._error(
                400,
                f"request exceeds the available context size: {n_in} > {enforce}",
            )

        if payload.get("stream") is False:
            # real engines honour `stream: false`; the capability probes use
            # the non-streaming path, so the mock must not always answer SSE
            return self._json(200, self._completion_json(payload))

        self._write_sse(self._stream_chunks(payload))

    def _completion_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming completion body, matching the streamed content."""
        key = self._prompt_key(payload)
        prompt_tokens = estimate_tokens(key)
        requested = payload.get("max_tokens")
        n_tokens = 32 if requested is None else max(0, min(int(requested), 48))
        n_think = (min(THINKING_TOKENS, n_tokens)
                   if THINKING_TOKENS > 0 and thinking_enabled(payload) else 0)
        n_content = n_tokens - n_think
        finish = "length" if (n_tokens and n_content == 0) else "stop"

        thinking = "".join(f"think{i} " for i in range(n_think))
        content = "".join(f"tok{i} " for i in range(n_content))
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if thinking:
            message[THINKING_FIELD] = thinking
        return {
            "id": "mock-1", "object": "chat.completion", "model": MODEL_ID,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n_tokens,
                "total_tokens": prompt_tokens + n_tokens,
                "completion_tokens_details": {"reasoning_tokens": n_think},
            },
        }

    def _crash(self, n_in: int, limit: int) -> None:
        """Model an OOM restart: drop mid-answer, then stop listening.

        The client must be able to tell this apart from a refusal, so the
        socket is closed on an *incomplete* chunked stream (no terminating
        chunk), and the listener is closed shortly after so the next request
        gets a connection error rather than hanging in the accept backlog.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        first = json.dumps({
            "id": "mock-1", "object": "chat.completion.chunk", "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": None,
                         "delta": {"role": "assistant", "content": ""}}],
        }).encode()
        try:
            self.wfile.write(f"{len(first):X}\r\n".encode() + first + b"\r\n")
            self.wfile.flush()
        except Exception:  # noqa: BLE001
            pass
        self.close_connection = True
        with contextlib.suppress(Exception):
            self.connection.close()

        def _die() -> None:
            time.sleep(0.1)
            with contextlib.suppress(Exception):
                self.server.shutdown()
            with contextlib.suppress(Exception):
                self.server.server_close()

        threading.Thread(target=_die, daemon=True).start()

    def _native_completion(self) -> None:
        """llama.cpp native /completion: honours cache_prompt and n_predict."""
        payload = self._read_json()
        if "n_predict" in payload and "max_tokens" not in payload:
            payload["max_tokens"] = payload["n_predict"]
        self._write_sse(self._stream_chunks(payload))

    def _write_sse(self, chunks) -> None:
        """Stream chunks as they are produced - never pre-materialise.

        Sleeping before the first byte would turn a streaming endpoint into a
        one-shot response and destroy every TTFT measurement taken against
        this mock, so the generator is consumed lazily here.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write(payload: bytes) -> None:
            self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

        try:
            for chunk in chunks:
                write(f"data: {json.dumps(chunk)}\n\n".encode())
            write(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _stream_chunks(self, payload: dict[str, Any]):
        """Hold a compute slot for the whole generation.

        Acquired before the prefill and released when the generator closes -
        which is also what happens when the client disconnects, mirroring a real
        engine: the request stops *running* while the work it already started
        still occupies the machine (that part lives in ``_mark_busy``).
        """
        _acquire_slot()
        try:
            yield from self._generate_chunks(payload)
        finally:
            _release_slot()

    def _generate_chunks(self, payload: dict[str, Any]):
        """Generate SSE chunks lazily, sleeping as a real engine would."""
        key = self._prompt_key(payload)
        cache_ok = payload.get("cache_prompt", True) is not False
        prompt_tokens = estimate_tokens(key)
        with _lock:
            matched = 0
            if cache_ok:
                for prev in _seen_prompts:
                    matched = max(matched, common_prefix_tokens(prev, key))
            _seen_prompts.add(key)
            # only whole alignment blocks can be reused
            matched = min(matched, prompt_tokens)
            cache_n = (matched // CACHE_ALIGN_TOKENS) * CACHE_ALIGN_TOKENS
            cached = cache_n >= CACHE_MIN_PREFIX_TOKENS

        ttft = TTFT_CACHED_S if cached else PREFILL_COLD_S
        requested = payload.get("max_tokens")
        # explicit 0 must stay 0: capability probing uses `n_predict: 0`
        n_tokens = 32 if requested is None else int(requested)
        n_tokens = max(0, min(n_tokens, 48))

        # thinking comes first and competes for the same output budget, exactly
        # as it does on a real reasoning model: with a small max_tokens the
        # whole budget can be consumed by reasoning and `content` stays empty
        n_think = (min(THINKING_TOKENS, n_tokens)
                   if THINKING_TOKENS > 0 and thinking_enabled(payload) else 0)
        n_content = n_tokens - n_think
        finish = "length" if (n_tokens and n_content == 0) else "stop"

        # The token stream is built first and *packaged* afterwards, because
        # packaging is an independent property: vLLM with speculative decoding
        # groups ~2.3 tokens per event, and a short output may be flushed in a
        # single burst.  Both change what a client can conclude from the
        # arrival times, so the mock has to be able to produce them.
        stream: list[dict[str, str]] = []
        for i in range(n_think):
            stream.append({THINKING_FIELD: f"think{i} "})
        for i in range(n_content):
            stream.append({"content": f"tok{i} "})

        # Book the slot for the whole simulated job, before any byte is sent.
        # A real engine cannot un-start a prefill because the client hung up.
        total_work = ttft + TOKEN_INTERVAL_S * len(stream)

        queued = 0.0
        if SERIALIZE:
            # One compute queue: this request waits for the current job,
            # including one whose client has already walked away.
            while _engine_busy() and queued < SERIALIZE_MAX_WAIT_S:
                time.sleep(0.05)
                queued += 0.05

        _mark_busy(total_work)

        if queued:
            # The wait is real to the *client* and must NOT be reported as
            # prefill time: this is precisely the distinction the real engine
            # makes (it answers with `prompt_ms` for the work it did), and the
            # whole bug being modelled here is a client that fails to notice.
            time.sleep(queued)

        if (ABANDON_AFTER_S is not None and _abandoned < ABANDON_TIMES
                and prompt_tokens > ABANDON_ABOVE_TOKENS
                and total_work > ABANDON_AFTER_S):
            # Die mid-prefill, the way the real fork did on a 128K request:
            # close the socket while the work continues in the background.
            globals()["_abandoned"] = _abandoned + 1
            time.sleep(ABANDON_AFTER_S)
            self.close_connection = True
            raise ConnectionResetError("mock: client abandoned mid-prefill")

        dbg(f"req prompt_tokens={prompt_tokens} cached={cached} cache_n={cache_n} "
            f"ttft={ttft:.3f} total_work={total_work:.2f} queued={queued:.2f} "
            f"abandon={'ARMD' if ABANDON_AFTER_S is not None else 'off'}"
            f"(n={_abandoned}/{ABANDON_TIMES},above={ABANDON_ABOVE_TOKENS})")

        if COALESCE:
            # the engine finishes generating, then flushes everything at once
            time.sleep(ttft + TOKEN_INTERVAL_S * len(stream))
        else:
            # the prefill happens *before* any byte leaves the server
            time.sleep(ttft)
        yield {
            "id": "mock-1", "object": "chat.completion.chunk", "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": None,
                         "delta": {"role": "assistant", "content": ""}}],
        }

        if COALESCE:
            # Everything - thinking and content - in one delta, emitted after
            # the full generation time.  Deterministic on purpose: the real
            # node's *tight multi-event* flush (0.036 ms across 4 events) is
            # timing-dependent on loopback, so that branch is pinned by a unit
            # test instead and this one reproduces the single-flush case end to
            # end without a flaky assertion.
            merged: dict[str, str] = {}
            for ev in stream:
                for key, val in ev.items():
                    merged[key] = merged.get(key, "") + val
            yield {
                "id": "mock-1", "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [{"index": 0, "finish_reason": None, "delta": merged}],
            }
        else:
            step = max(1, TOKENS_PER_CHUNK)
            for i in range(0, len(stream), step):
                # Sleeping per *token* rather than per event keeps the mock's
                # throughput constant, so turning granularity up does not
                # silently change every timing assertion in the suite.
                time.sleep(TOKEN_INTERVAL_S * len(stream[i:i + step]))
                merged = {}
                for ev in stream[i:i + step]:
                    for key, val in ev.items():
                        merged[key] = merged.get(key, "") + val
                yield {
                    "id": "mock-1", "object": "chat.completion.chunk",
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "finish_reason": None,
                                 "delta": merged}],
                }
        yield {
            "id": "mock-1", "object": "chat.completion.chunk", "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": finish, "delta": {}}],
            "timings": self._timings(cache_n, prompt_tokens, n_tokens, ttft),
        }
        # usage block, as requested via stream_options.include_usage
        yield {
            "id": "mock-1", "object": "chat.completion.chunk", "model": MODEL_ID,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n_tokens,
                "total_tokens": prompt_tokens + n_tokens,
                "completion_tokens_details": {"reasoning_tokens": n_think},
                "prompt_tokens_details": {"cached_tokens": cache_n},
                "_mock": {"ttft_ms": round(ttft * 1000, 2), "cached": cached,
                          "cache_n": cache_n, "thinking_field": THINKING_FIELD,
                          "reasoning_tokens": n_think},
            },
            "timings": self._timings(cache_n, prompt_tokens, n_tokens, ttft),
        }

    @staticmethod
    def _timings(cache_n: int, prompt_tokens: int, n_tokens: int,
                 ttft: float) -> dict[str, Any]:
        """The fork's per-response server-side timings.

        Reporting these honestly is the point: ``cache_n`` lets a client detect
        a prompt-cache hit *directly* instead of inferring it from a suspicious
        TTFT, and ``prompt_ms`` is the engine's own prefill time to compare
        against the client's measurement.
        """
        prompt_ms = round(ttft * 1000, 3)
        predicted_ms = round(n_tokens * TOKEN_INTERVAL_S * 1000, 3)
        return {
            "prompt_n": prompt_tokens,
            "predicted_n": n_tokens,
            "prompt_ms": prompt_ms,
            "predicted_ms": predicted_ms,
            "prompt_per_second": (round(prompt_tokens / ttft, 3)
                                  if ttft > 0 else 0.0),
            "predicted_per_second": round(1 / TOKEN_INTERVAL_S, 3),
            "cache_n": cache_n,
            "disk_restore_n": 0,
            "disk_restore_ms": 0.0,
            "prefix_n": cache_n,
            "draft_n": 0,
            "draft_n_accepted": 0,
        }


FLAVOURS = ("llamacpp-vanilla", "llamacpp-fork", "vllm", "generic")


def serve(flavor: str, port: int, host: str = "127.0.0.1") -> None:
    if flavor not in FLAVOURS:
        raise SystemExit(f"unknown flavour {flavor!r}; pick one of {FLAVOURS}")
    handler = type(f"Handler_{flavor.replace('-', '_')}", (MockEngine,),
                   {"flavor": flavor})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"mock engine [{flavor}] on http://{host}:{port}")
    if LIMITS.get("enforce") is not None:
        print(f"  refuses inputs above {LIMITS['enforce']} tokens (HTTP 400)")
    if LIMITS.get("die") is not None:
        print(f"  crashes above {LIMITS['die']} tokens (models an OOM restart)")
    if LIMITS.get("die_after") is not None:
        print(f"  crashes after {LIMITS['die_after']} requests")
    if THINKING_TOKENS:
        print(f"  emits {THINKING_TOKENS} reasoning tokens via "
              f"`delta.{THINKING_FIELD}` before any content")
    if PREFIX_CACHING:
        print("  reports enable_prefix_caching=True (no API switch to disable)")
    if TOKENS_PER_CHUNK != 1:
        print(f"  packs {TOKENS_PER_CHUNK} tokens into each SSE event "
              "(mirrors speculative decoding)")
    if COALESCE:
        print("  flushes the whole output in one burst after generating it "
              "(mirrors vLLM on a short max_tokens)")
    if ABANDON_AFTER_S is not None:
        print(f"  drops the connection after {ABANDON_AFTER_S}s on the first "
              f"{ABANDON_TIMES} long request(s), while KEEPING the slot busy "
              f"(mirrors a fork abandoning a 128K prefill)")
    if PREFILL_COLD_S != 0.20:
        print(f"  cold prefill takes {PREFILL_COLD_S}s")
    if SERIALIZE:
        print("  serves ONE request at a time: a new request waits for the "
              "current job (including one its client abandoned)")
    if MAX_RUNNING:
        print(f"  runs at most {MAX_RUNNING} generation(s) at once and QUEUES "
              f"the rest (gauges report real running/queued counts)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    global CONTEXT, NO_TOKENIZE, THINKING_TOKENS, THINKING_FIELD, PREFIX_CACHING
    global TOKENS_PER_CHUNK, COALESCE, ABANDON_AFTER_S, ABANDON_TIMES
    global PREFILL_COLD_S, ABANDON_ABOVE_TOKENS, SERIALIZE, DEBUG, MAX_RUNNING
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", default="llamacpp-fork", choices=FLAVOURS)
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--context", type=int, default=CONTEXT,
                    help="advertised context window, in tokens")
    ap.add_argument("--max-context", type=int, default=None,
                    help="refuse inputs longer than this (HTTP 400)")
    ap.add_argument("--die-above", type=int, default=None,
                    help="crash the whole server above this input length")
    ap.add_argument("--die-after", type=int, default=None,
                    help="crash the whole server after this many requests")
    ap.add_argument("--no-tokenize", action="store_true",
                    help="hide /tokenize, /detokenize and the native endpoints "
                         "(mirrors the halogen fork)")
    ap.add_argument("--thinking-tokens", type=int, default=0,
                    help="emit this many reasoning tokens before any content "
                         "(they consume the same max_tokens budget)")
    ap.add_argument("--thinking-field", default="reasoning_content",
                    choices=["reasoning_content", "reasoning"],
                    help="which delta field carries the reasoning. "
                         "llama.cpp forks use reasoning_content, vLLM uses "
                         "reasoning; the wrong one makes every chunk invisible")
    ap.add_argument("--prefix-caching", action="store_true",
                    help="emit vllm:cache_config_info with "
                         "enable_prefix_caching=True (no API switch to turn it "
                         "off, exactly like real vLLM)")
    ap.add_argument("--tokens-per-chunk", type=int, default=1,
                    help="how many tokens each SSE event carries. Real vLLM "
                         "with speculative decoding sends ~2.3; a client that "
                         "assumes one event per token gets both the rate and "
                         "the per-token latency wrong")
    ap.add_argument("--coalesce", action="store_true",
                    help="generate the whole output first, then flush it in one "
                         "burst with no per-token pacing. This is what a short "
                         "max_tokens does on real vLLM, and it is the case "
                         "where a client-side decode rate is not measurable")
    ap.add_argument("--abandon-after", type=float, default=None,
                    metavar="SECONDS",
                    help="on any request whose simulated work exceeds this, "
                         "close the connection mid-prefill while KEEPING the "
                         "slot busy. Models a fork abandoning a 128K prefill; "
                         "it is what makes the NEXT request queue and its TTFT "
                         "disagree with the engine's own prompt_ms")
    ap.add_argument("--abandon-times", type=int, default=1,
                    help="how many requests to abandon before behaving "
                         "normally (default 1; abandoning all of them would "
                         "leave nothing to measure)")
    ap.add_argument("--abandon-above", type=int, default=0, metavar="TOKENS",
                    help="only abandon requests whose input exceeds this many "
                         "tokens, leaving small calibration probes alone")
    ap.add_argument("--serialize", action="store_true",
                    help="serve one request at a time and make a new request "
                         "actually WAIT for the current job. Without this the "
                         "mock answers in parallel threads and its busy gauge "
                         "costs nobody any time")
    ap.add_argument("--max-running", type=int, default=0, metavar="N",
                    help="run at most N generations at once and queue the rest "
                         "(0 = unlimited). This is vLLM's --max-num-seqs: it is "
                         "the only way a client can tell '32 in flight' from "
                         "'16 running and 16 queued', because both look like 32 "
                         "from the client side")
    ap.add_argument("--prefill-seconds", type=float, default=0.20,
                    help="cold-prefill cost, so a test can create a queue wait "
                         "long enough to measure")
    args = ap.parse_args()

    CONTEXT = args.context
    LIMITS["enforce"] = args.max_context
    LIMITS["die"] = args.die_above
    LIMITS["die_after"] = args.die_after
    NO_TOKENIZE = args.no_tokenize
    THINKING_TOKENS = args.thinking_tokens
    THINKING_FIELD = args.thinking_field
    PREFIX_CACHING = args.prefix_caching
    TOKENS_PER_CHUNK = max(1, args.tokens_per_chunk)
    COALESCE = args.coalesce
    ABANDON_AFTER_S = args.abandon_after
    ABANDON_TIMES = max(0, args.abandon_times)
    ABANDON_ABOVE_TOKENS = max(0, args.abandon_above)
    SERIALIZE = args.serialize
    MAX_RUNNING = max(0, args.max_running)
    DEBUG = os.environ.get("MOCK_DEBUG") == "1"
    PREFILL_COLD_S = max(0.0, args.prefill_seconds)
    serve(args.flavor, args.port, args.host)


if __name__ == "__main__":
    main()
