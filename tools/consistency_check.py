"""Cross-task consistency check: do the three tasks mean the same thing?

The dashboard puts Performance, Context and Concurrency results side by side,
so a metric with the same name has to be the same metric in all three.  This
tool pins that down, and it exists because they *were* inconsistent:

  * the Context task gave every sample of a ladder step the **same input**, so
    on an engine that cannot switch off its prefix cache the repeats were
    served from cache - prefill inside one step was measured at both 252 and
    2000 tok/s (7.9x), and TTFT moved 8x (measured on a live node);
  * the Context task **counted warmup samples** in its statistics while the
    other two discarded them;
  * the Context task **reused one marker per step** instead of one per sample,
    which is the same thing as the first point.

Three layers:

  A. unit       - the shared helpers behave as documented
  B. end-to-end - all three tasks against one cache-hungry engine, asserting
                  that every scored sample used a distinct input and that the
                  within-group spread is small
  C. control    - with uniqueness deliberately off, the run must *say* the
                  samples were not independent instead of hiding it
  D. decode     - a whole output arriving in one burst must produce NO decode
                  rate (and a warning), while a multi-token-per-event stream
                  must still produce the *right* one.  This is the "decode
                  looks unstable" report: on a live vLLM the same engine gave a
                  steady 135-152 tok/s at max_tokens>=32 and an apparent
                  196,078 tok/s at max_tokens=8, where the whole reply arrived
                  inside 0.036 ms.
  E. prefill    - an engine that abandons a long request mid-prefill while
                  keeping its slot busy must not have the resulting queue wait
                  published as a prefill rate.  This is the "longer context is
                  faster" report: a live fork reported 1083 tok/s for a step
                  whose own accounting said 1536, because 35.7 s of the client
                  TTFT was spent queueing behind an abandoned 128K warmup.

Prerequisite: the backend must be running on 127.0.0.1:8000.

Usage:
    python tools/consistency_check.py
"""

from __future__ import annotations

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

# A vLLM-shaped node with prefix caching ON and no API switch to turn it off:
# the most cache-hungry engine we can simulate, i.e. the one that exposes any
# inconsistency in how the tasks build their inputs.
MOCK_PORT = 9200
MOCK_ARGS = [
    "mock_engine.py", "--flavor", "vllm", "--port", str(MOCK_PORT),
    "--prefix-caching",
]
TARGET = f"http://127.0.0.1:{MOCK_PORT}"

# A second engine whose cache *can* be switched off, for the control run: the
# capability gate only overrides `unique_prompt` when the cache cannot be
# disabled, so the "reused input is reported" path needs an engine where the
# user's choice actually stands.
OPEN_PORT = 9201
OPEN_TARGET = f"http://127.0.0.1:{OPEN_PORT}"

# An engine that computes a short output and then flushes it in one burst -
# measured on a live vLLM node, 8 tokens arrived with a first-to-last span of
# 0.036 ms.  No client-side decode rate can be derived from that, so the run
# must *say* it could not measure rather than publish 1e5 tok/s.
COALESCE_PORT = 9202
COALESCE_TARGET = f"http://127.0.0.1:{COALESCE_PORT}"

# An engine whose stream events carry several tokens each, the way vLLM does
# under speculative decoding (~2.3 tokens per scheduler step on the node this
# was written against).  The rate must still be right, which is the check that
# "one stream event = one token" has not crept back in.
BURST_PORT = 9203
BURST_TARGET = f"http://127.0.0.1:{BURST_PORT}"
BURST_TOKENS_PER_CHUNK = 3
#: the mock's own decode rate, so the measured one can be compared to it
MOCK_TOKENS_PER_SECOND = 1 / 0.02

# An engine that abandons a long request mid-prefill while KEEPING its slot
# busy - a real fork did exactly this on a 128K warmup (ReadError at 50 s, the
# prefill still needed ~85 s).  The next request then queues, and its TTFT
# absorbs the wait: measured at 121,144 ms against the engine's own 85,410 ms
# of prefill, which reported 1083 tok/s instead of 1536.
BUSY_PORT = 9204
BUSY_TARGET = f"http://127.0.0.1:{BUSY_PORT}"

# The same abandonment, but on an engine that exposes NO metrics endpoint at
# all - a plain OpenAI-compatible server.  A client that cannot see the engine's
# state cannot drain it, so the queue wait is unavoidable and must be
# *reported*, not published as a rate.
BUSY_BLIND_PORT = 9205
BUSY_BLIND_TARGET = f"http://127.0.0.1:{BUSY_BLIND_PORT}"
#: "generic" has no /metrics (and no /health), which is the point: the harness
#: can neither read a busy gauge nor fall back to one
BUSY_BLIND_FLAVOUR = "generic"

#: how long the mock's cold prefill takes in these runs, and how much of it
#: survives the abandonment as a busy slot
BUSY_PREFILL_S = 6.0
BUSY_ABANDON_AFTER_S = 0.5
#: calibration probes are 400/1600 tokens on this mock, so a threshold above
#: them leaves the abandonment for the ladder step it is meant to hit
BUSY_ABANDON_ABOVE_TOKENS = 4096

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def fetch(url: str, payload=None, timeout: float = 600.0):
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
def run_task(task: str, overrides: dict, target: str = TARGET,
             timeout: float = 600.0, top: dict | None = None) -> dict | None:
    """Run one task against a mock and return its sub-document.

    ``top`` adds run-level fields (``retries``, ``timeout_s``).  The retry knob
    matters for the prefill layer: the client retries a transport failure that
    produced no bytes, and a retried request is *the same text twice*, which an
    engine with a prompt cache then serves from cache.  Reproducing a real
    abandonment therefore needs ``retries: 0``, the way the live node behaved.
    """
    preset = {
        "performance": "quick",
        "context": "context",
        "concurrency": "concurrency",
    }[task]
    body = {
        "base_url": target,
        "preset": preset,
        "task_type": task,
        task: overrides,
    }
    body.update(top or {})
    st, run = fetch(f"{API}/api/runs", body)
    if st != 200:
        check(False, f"{task}: create run", str(run)[:200])
        return None
    rid = run["run_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        st, doc = fetch(f"{API}/api/runs/{rid}", timeout=60)
        if not isinstance(doc, dict):
            break
        if doc.get("status") not in ("queued", "running", "pending", "connect",
                                     "warmup", "testing"):
            return doc.get("result")
        time.sleep(1)
    check(False, f"{task}: run completed")
    return None


def groups(doc: dict, task: str) -> list[tuple[str, list[dict]]]:
    """The comparing unit per task: a ladder step, a level, or the run."""
    if task == "context":
        return [(f"{c['target_tokens']} tok", c.get("samples") or [])
                for c in (doc.get("contexts") or [])
                if c.get("samples")]
    if task == "concurrency":
        return [(f"C={l['concurrency']}", l.get("samples") or [])
                for l in (doc.get("levels") or [])]
    return [("run", doc.get("samples") or [])]


def spread(values: list[float]) -> float:
    vals = [v for v in values if v]
    if len(vals) < 2:
        return 1.0
    return max(vals) / max(min(vals), 1e-9)


# ---------------------------------------------------------------------- #
# A. the shared helpers
# ---------------------------------------------------------------------- #
def layer_a() -> None:
    print("\n=== A. shared helpers ===")
    from app.engine.probe import (
        MIN_STREAM_GAP_MS,
        MARKER_WIDTH,
        _r,  # noqa: F401  (imported to prove the module is importable)
        decode_coalesced_warning,
        input_fingerprint,
        input_identity_warning,
        input_length_warning,
        input_marker,
        scored_samples,
        summarize_samples,
    )

    m1, m2 = input_marker(True), input_marker(True)
    check(m1 != m2, "unique markers differ")
    check(len(m1) == len(m2) == MARKER_WIDTH,
          "every marker has the same width (so input length is stable)",
          f"{len(m1)} vs {MARKER_WIDTH}")
    placeholder = input_marker(False)
    check(placeholder != m1 and len(placeholder) == MARKER_WIDTH,
          "the placeholder has the same width as a real marker")
    check("bench" in placeholder.lower(), "marker is self-describing",
          placeholder.strip())

    a = input_fingerprint([{"role": "user", "content": "x"}], None)
    b = input_fingerprint([{"role": "user", "content": "y"}], None)
    check(a != b, "different content -> different fingerprint")
    check(a == input_fingerprint([{"content": "x", "role": "user"}], None),
          "key order does not change the fingerprint")

    samples = [
        {"ok": True, "warmup": True, "ttft_ms": 999.0, "input_hash": "w"},
        {"ok": True, "warmup": False, "ttft_ms": 100.0, "input_hash": "a"},
        {"ok": True, "warmup": False, "ttft_ms": 200.0, "input_hash": "b"},
    ]
    summary = summarize_samples(samples)
    check(summary["warmup_excluded"] == 1, "warmup is excluded from the summary",
          str(summary["warmup_excluded"]))
    check(summary["stats"]["ttft_ms"]["mean"] == 150.0,
          "and the mean is taken over scored samples only",
          str(summary["stats"]["ttft_ms"]["mean"]))
    check(len(scored_samples(samples)) == 2, "scored_samples drops warmups")

    dupes = [{"ok": True, "input_hash": "same"},
             {"ok": True, "input_hash": "same"}]
    distinct = [{"ok": True, "input_hash": "a"},
                {"ok": True, "input_hash": "b"}]
    check(input_identity_warning(dupes) is not None,
          "a reused input is reported")
    check(input_identity_warning(distinct) is None,
          "distinct inputs are not reported")
    check(input_identity_warning([{"ok": True, "input_hash": "a"}]) is None,
          "a single sample is not reported (nothing to compare)")

    spread_s = [{"ok": True, "input_tokens": 1000},
                {"ok": True, "input_tokens": 1300}]
    stable = [{"ok": True, "input_tokens": 1000},
              {"ok": True, "input_tokens": 1002}]
    check(input_length_warning(spread_s) is not None,
          "a large input-length spread is reported")
    check(input_length_warning(stable) is None,
          "a small input-length spread is not reported")

    # ---- decode measurability -------------------------------------------
    # The failure this guards against: a whole short output arriving in one
    # burst, so `e2e - ttft` collapses and the rate comes out at 1e5 tok/s.
    # A withheld number is correct; a published one is a lie.
    check(MIN_STREAM_GAP_MS > 0, "a minimum resolvable stream gap is defined",
          f"{MIN_STREAM_GAP_MS} ms")
    healthy = [{"ok": True, "output_tokens": 64, "decode_tps": 150.0,
                "decode_span_ms": 420.0, "stream_events": 29,
                "decode_unmeasurable": None}]
    check(decode_coalesced_warning(healthy) is None,
          "a properly paced stream is not reported")

    for reason, span, events in (("single_flush", 0.0, 1),
                                 ("coalesced_flush", 0.036, 4)):
        bad = [dict(healthy[0], decode_tps=None, decode_unmeasurable=reason,
                    decode_span_ms=span, stream_events=events)]
        msg = decode_coalesced_warning(bad)
        check(msg is not None, f"a '{reason}' sample is reported")
        if msg:
            check("无法测量" in msg and "max_tokens" in msg,
                  f"and the '{reason}' warning says what to change",
                  msg[:110])

    # The withheld sample must also vanish from the mean, or the report would
    # simply average the survivors and look clean.
    mixed = [
        {"ok": True, "output_tokens": 64, "decode_tps": 150.0,
         "decode_span_ms": 420.0, "decode_unmeasurable": None},
        {"ok": True, "output_tokens": 8, "decode_tps": None,
         "decode_span_ms": 0.036, "decode_unmeasurable": "coalesced_flush"},
    ]
    mean = summarize_samples(mixed)["stats"]["decode_tps"]["mean"]
    check(mean == 150.0,
          "a withheld decode_tps does not drag the mean",
          str(mean))

    # ---- prefill contamination ------------------------------------------
    # `prefill_tps = input/ttft` is only a throughput if TTFT is prefill.  When
    # the engine reports its own prefill time, the difference is queueing.
    from app.engine.probe import prefill_contamination_warning

    clean = [{"ok": True, "ttft_ms": 20000.0, "server_prompt_ms": 19800.0,
              "queue_wait_ms": 200.0, "prefill_contaminated": False}]
    check(prefill_contamination_warning(clean) is None,
          "a sample whose TTFT is prefill is not reported")

    # the real numbers: 121,144 ms of TTFT against 85,410 ms of engine prefill
    dirty = [{"ok": True, "ttft_ms": 121144.0, "server_prompt_ms": 85410.0,
              "queue_wait_ms": 35734.0, "prefill_contaminated": True}]
    msg = prefill_contamination_warning(dirty)
    check(msg is not None, "a queue-dominated TTFT is reported")
    if msg:
        check("等待" in msg and "prefill" in msg,
              "and the warning says what the number actually measured",
              msg[:110])

    # no evidence -> no claim
    unknown = [{"ok": True, "ttft_ms": 20000.0, "server_prompt_ms": None,
                "queue_wait_ms": None, "prefill_contaminated": None}]
    check(prefill_contamination_warning(unknown) is None,
          "without the engine's own prefill time nothing is claimed")

    # ---- the fixed-cost fit ---------------------------------------------
    # A prefill_tps column necessarily rises while a fixed per-request cost is
    # amortised; the fit is what stops that being read as "longer is faster".
    from app.engine.context import _prefill_fit

    fit = _prefill_fit([
        {"input_tokens": 2048, "ttft_ms": 1571.0, "prefill_contaminated": False},
        {"input_tokens": 8192, "ttft_ms": 5336.0, "prefill_contaminated": False},
        {"input_tokens": 16384, "ttft_ms": 9843.0, "prefill_contaminated": False},
        {"input_tokens": 32768, "ttft_ms": 20528.0, "prefill_contaminated": False},
    ])
    check(bool(fit), "a fixed-cost fit is produced from the ladder", str(fit))
    if fit:
        check(abs(fit["marginal_ms_per_token"] - 0.62) < 0.15,
              "and the marginal cost per token is about right",
              f"{fit['marginal_ms_per_token']} ms/token")
        check("固定开销" in fit["note"], "with a readable explanation",
              fit["note"][:90])

    # a contaminated point must not be allowed to bend the slope
    bent = _prefill_fit([
        {"input_tokens": 2048, "ttft_ms": 1571.0, "prefill_contaminated": False},
        {"input_tokens": 8192, "ttft_ms": 5336.0, "prefill_contaminated": False},
        {"input_tokens": 16384, "ttft_ms": 9843.0, "prefill_contaminated": False},
        {"input_tokens": 32768, "ttft_ms": 20528.0, "prefill_contaminated": False},
        {"input_tokens": 65536, "ttft_ms": 999999.0, "prefill_contaminated": True},
    ])
    check(bent.get("fixed_ms") == fit.get("fixed_ms"),
          "a contaminated point is excluded from the fit",
          f"{bent.get('fixed_ms')} vs {fit.get('fixed_ms')}")


# ---------------------------------------------------------------------- #
# B. all three tasks, one cache-hungry engine
# ---------------------------------------------------------------------- #
def layer_b() -> None:
    print("\n=== B. three tasks, one engine that cannot switch off its cache ===")
    common = {"warmup": 1, "repeats": 3, "max_tokens": 12}

    docs: dict[str, dict] = {}
    docs["performance"] = run_task("performance", dict(common)) or {}
    print(f"       performance: {len(docs['performance'].get('samples') or [])} "
          f"scored sample(s)")
    docs["context"] = run_task("context", {
        **common, "context_ladder": "1024,2048", "detect_max_context": False,
    }) or {}
    docs["concurrency"] = run_task("concurrency", {
        "warmup": 1, "max_tokens": 12,
        "concurrency_ladder": "1,2", "requests_per_level": 4,
    }) or {}

    for task, doc in docs.items():
        if not doc:
            check(False, f"{task}: produced a result")
            continue

        # ---- the engine really is cache-hungry ------------------------
        caps = doc.get("capabilities") or {}
        if task == "performance":
            check(caps.get("prompt_cache_enabled") is True,
                  "the mock engine reports prompt caching on (control)")
            check(not caps.get("cache_prompt_control"),
                  "and reports no switch to disable it (control)")

        # ---- every scored sample used a distinct input ---------------
        for label, samples in groups(doc, task):
            scored = [s for s in samples if not s.get("warmup") and s.get("ok")]
            hashes = [s.get("input_hash") for s in scored if s.get("input_hash")]
            check(all(h is not None for h in hashes),
                  f"{task}/{label}: every sample carries an input fingerprint",
                  f"{len(hashes)}/{len(scored)}")
            check(len(set(hashes)) == len(hashes),
                  f"{task}/{label}: no input was reused across samples",
                  f"{len(hashes)} samples, {len(set(hashes))} distinct")

            # ---- and therefore the within-group numbers are stable ----
            prefill = [s.get("prefill_tps") for s in scored]
            ttft = [s.get("ttft_ms") for s in scored]
            check(spread(prefill) < 1.6,
                  f"{task}/{label}: prefill is consistent within the group",
                  f"{spread(prefill):.2f}x  {[round(p or 0) for p in prefill]}")
            check(spread(ttft) < 1.6,
                  f"{task}/{label}: TTFT is consistent within the group",
                  f"{spread(ttft):.2f}x  {[round(t or 0) for t in ttft]}")

        # ---- warmup handling ----------------------------------------
        if task == "context":
            excluded = [c.get("warmup_excluded") for c in doc.get("contexts") or []
                        if c.get("samples")]
            check(all(e == 1 for e in excluded),
                  "context: every step excluded its warmup sample",
                  str(excluded))
            scored_counts = [c.get("scored_samples")
                             for c in doc.get("contexts") or [] if c.get("samples")]
            check(all(n == 3 for n in scored_counts),
                  "context: 3 scored samples per step (warmup not counted)",
                  str(scored_counts))

        # ---- nothing was hidden: no warnings on a clean run -----------
        warnings = doc.get("data_warnings") or []
        check(not any("重复使用了同一段输入" in w for w in warnings),
              f"{task}: no 'reused input' warning on a correctly built run",
              "; ".join(warnings)[:150] or "(none)")
        check(not any("输入长度不一致" in w for w in warnings),
              f"{task}: no input-length warning on a correctly built run",
              "; ".join(warnings)[:150] or "(none)")

    # ---- the definitions themselves are identical ---------------------
    print("\n=== B. the same formula behind every 'prefill' ===")
    for task, doc in docs.items():
        for label, samples in groups(doc, task):
            scored = [s for s in samples if not s.get("warmup") and s.get("ok")]
            if not scored:
                continue
            bad = []
            for s in scored:
                ttft_s = (s.get("ttft_ms") or 0) / 1000.0
                if not ttft_s or not s.get("input_tokens"):
                    continue
                expected = s["input_tokens"] / ttft_s
                got = s.get("prefill_tps") or 0
                if abs(got - expected) > max(1.0, expected * 0.02):
                    bad.append((s["input_tokens"], s["ttft_ms"], got, expected))
            check(not bad,
                  f"{task}/{label}: prefill == input_tokens / ttft",
                  f"{len(bad)} mismatch(es) {bad[:2]}")

    # ---- the input really is the length that was asked for ------------
    print("\n=== B. Context: requested length vs achieved length ===")
    ctx = docs.get("context") or {}
    for c in ctx.get("contexts") or []:
        if not c.get("samples"):
            continue
        built = c.get("built") or {}
        source = built.get("source")
        dev = built.get("deviation_pct")
        target = c["target_tokens"]
        check(source == "tokenize",
              f"context {target}: measured with the engine's tokenizer",
              str(source))
        check(dev is not None and abs(dev) <= 2.0,
              f"context {target}: built within 2% of the target",
              f"{dev}%")
        # the server's own count is what the request actually cost
        scored = [s for s in c["samples"] if not s.get("warmup")]
        got = [s.get("input_tokens") for s in scored if s.get("input_tokens")]
        if got:
            server_dev = (sum(got) / len(got) - target) / target * 100
            check(abs(server_dev) <= 5.0,
                  f"context {target}: server-side length within 5%",
                  f"{server_dev:+.1f}%")
            check(len(set(got)) == 1,
                  f"context {target}: every sample cost the same input length",
                  str(got))


# ---------------------------------------------------------------------- #
# C. controls
# ---------------------------------------------------------------------- #
def layer_c() -> None:
    print("\n=== C1. cache cannot be disabled: uniqueness is forced, and said ===")
    doc = run_task("context", {
        "warmup": 1, "repeats": 3, "max_tokens": 12,
        "context_ladder": "2048", "detect_max_context": False,
        "unique_prompt": False,
    })
    if not doc:
        check(False, "C1 run completed")
        return
    cfg = doc.get("config") or {}
    check(cfg.get("unique_prompt") is True,
          "unique_prompt=false is overridden on an engine with no cache switch",
          str(cfg.get("unique_prompt")))
    warnings = doc.get("settings_warnings") or []
    check(any("不重复输入" in w for w in warnings),
          "and the override is explained to the user",
          "; ".join(warnings)[:170] or "(none)")
    entry = next((c for c in doc.get("contexts") or [] if c.get("samples")), None)
    scored = [s for s in (entry or {}).get("samples", []) if not s.get("warmup")]
    hashes = [s.get("input_hash") for s in scored]
    check(len(set(hashes)) == len(hashes),
          "so the inputs really are distinct despite the request",
          f"{len(hashes)} samples, {len(set(hashes))} distinct")

    print("\n=== C2. cache off and uniqueness off: the run must warn ===")
    # A controllable engine: nothing overrides the user's choice here, so this
    # is the case where reusing one input is *possible* - and therefore the
    # case where silence would be dangerous.
    doc2 = run_task("context", {
        "warmup": 0, "repeats": 3, "max_tokens": 12,
        "context_ladder": "2048", "detect_max_context": False,
        "unique_prompt": False, "cache_prompt": True,
    }, target=OPEN_TARGET)
    if not doc2:
        check(False, "C2 run completed")
        return
    cfg2 = doc2.get("config") or {}
    check(cfg2.get("unique_prompt") is False,
          "with a controllable cache the user's choice is respected",
          str(cfg2.get("unique_prompt")))

    entry2 = next((c for c in doc2.get("contexts") or [] if c.get("samples")), None)
    if not entry2:
        check(False, "C2 run produced a step")
        return
    scored2 = [s for s in entry2["samples"] if not s.get("warmup")]
    hashes2 = [s.get("input_hash") for s in scored2]
    check(len(set(hashes2)) == 1,
          "every sample really did reuse one input",
          f"{len(hashes2)} samples, {len(set(hashes2))} distinct")

    warnings2 = doc2.get("data_warnings") or []
    check(any("重复使用了同一段输入" in w for w in warnings2),
          "and the result says so instead of quietly averaging",
          "; ".join(warnings2)[:170] or "(none)")

    # the consequence, for the record: the mean is a cache hit, not the engine
    prefill = [s.get("prefill_tps") for s in scored2 if s.get("prefill_tps")]
    ttft = [s.get("ttft_ms") for s in scored2 if s.get("ttft_ms")]
    print(f"       (observed with a shared input: prefill "
          f"{[round(p) for p in prefill]} tok/s, TTFT {[round(t) for t in ttft]} ms)")


# ---------------------------------------------------------------------- #
# D. decode: a rate is only published when it was actually measurable
# ---------------------------------------------------------------------- #
def layer_d() -> None:
    print("\n=== D. decode measurability (end to end) ===")

    # ---- D1. the engine flushes a whole short output in one burst --------
    doc = run_task("performance", {
        "warmup": 0, "repeats": 3, "max_tokens": 32,
    }, target=COALESCE_TARGET)
    if not doc:
        check(False, "coalescing run completed")
        return

    samples = [s for s in (doc.get("samples") or []) if s.get("ok")]
    check(bool(samples), "coalescing engine produced samples", str(len(samples)))
    if not samples:
        return

    reasons = [s.get("decode_unmeasurable") for s in samples]
    check(all(r for r in reasons),
          "every sample from a burst-flushed stream is marked unmeasurable",
          str(reasons))
    check(all(s.get("decode_tps") is None for s in samples),
          "and no decode_tps is published for any of them",
          str([s.get("decode_tps") for s in samples]))

    # the trap this replaces: usage still reported 32 tokens, so the old
    # formula happily divided an ~0 ms window by 31 and printed a huge number
    spawns = [s.get("decode_span_ms") for s in samples]
    print(f"       (first-to-last token span: {spawns} ms, "
          f"events: {[s.get('stream_events') for s in samples]}, "
          f"usage said {[s.get('output_tokens') for s in samples]} tokens)")

    # other metrics must survive: only decode was unmeasurable
    check(all(s.get("ttft_ms") is not None for s in samples),
          "TTFT is still reported (only decode was withheld)")
    check(all((s.get("output_tokens") or 0) > 0 for s in samples),
          "and the token accounting is intact")
    check((doc.get("stats") or {}).get("decode_tps", {}).get("count") == 0,
          "the summary has no decode_tps to average",
          str((doc.get("stats") or {}).get("decode_tps", {}).get("count")))

    warns = doc.get("data_warnings") or []
    check(any("decode 无法测量" in w for w in warns),
          "and the run warns about it instead of staying silent",
          "; ".join(warns)[:170] or "(none)")

    # ---- D2. many tokens per event must not change the rate ---------------
    burst = run_task("performance", {
        "warmup": 0, "repeats": 3, "max_tokens": 48,
    }, target=BURST_TARGET)
    if not burst:
        check(False, "burst run completed")
        return

    bs = [s for s in (burst.get("samples") or []) if s.get("ok")]
    if not bs:
        check(False, "burst engine produced samples")
        return

    rates = [s.get("decode_tps") for s in bs if s.get("decode_tps")]
    check(bool(rates), "a multi-token-per-event stream is still measurable",
          str(rates))
    if rates:
        # With 3 tokens packed per event the client sees ~1/3 as many events,
        # but the *rate* is unchanged.  A harness that counted events instead
        # of tokens would be 3x off here.
        lo = MOCK_TOKENS_PER_SECOND / 1.6
        hi = MOCK_TOKENS_PER_SECOND * 1.6
        check(all(lo <= r <= hi for r in rates),
              "and the rate matches the engine's real rate, not the event rate",
              f"measured {[round(r, 1) for r in rates]} vs "
              f"engine {MOCK_TOKENS_PER_SECOND:.0f} tok/s")

        per_event = [s.get("output_tokens", 0) / max(s.get("stream_events") or 1, 1)
                     for s in bs]
        check(all(p > 1.5 for p in per_event),
              "the event granularity really was >1 token per event",
              str([round(p, 2) for p in per_event]))

    check(not (burst.get("data_warnings") or []),
          "a properly paced multi-token stream raises no decode warning",
          "; ".join(burst.get("data_warnings") or [])[:150] or "(none)")


# ---------------------------------------------------------------------- #
# E. prefill: a queue wait must not be published as a prefill rate
# ---------------------------------------------------------------------- #
def layer_e() -> None:
    print("\n=== E. prefill rate vs queue wait (end to end) ===")

    overrides = {
        "warmup": 1, "repeats": 2, "max_tokens": 16,
        "context_ladder": "8192,16384", "detect_max_context": False,
    }
    # `retries: 0` reproduces the live node: the abandoned prefill was NOT
    # retried.  With the default retry budget the client re-sends the same
    # text, the engine's cache answers it, and the queue wait never reaches a
    # scored sample - the test would pass for the wrong reason.
    no_retry = {"retries": 0}

    # ---- E1. the engine exposes a busy gauge -> we can drain it ----------
    # The 6 s prefill is abandoned after 0.5 s, so ~5.5 s of work survives the
    # client giving up.  A harness that fires the next request immediately
    # charges that to the next sample's TTFT.
    doc = run_task("context", overrides, target=BUSY_TARGET, timeout=600,
                   top=no_retry)
    if not doc:
        check(False, "busy-engine run completed")
        return

    steps = [c for c in (doc.get("contexts") or []) if c.get("samples")]
    check(bool(steps), "the busy engine produced steps", str(len(steps)))
    if not steps:
        return

    waits = [c.get("queue_wait_ms") for c in steps]
    settled = any("引擎仍在处理它" in w for w in (doc.get("data_warnings") or []))
    print(f"       E1 drained: settle note={settled}  queue_wait_ms={waits}")
    check(settled,
          "after a failed request the harness waits for the engine to drain",
          "; ".join(doc.get("data_warnings") or [])[:150] or "(none)")
    check(not any(c.get("prefill_contaminated") for c in steps),
          "so no step ends up with a queue wait inside its prefill rate",
          str([c.get("prefill_contaminated") for c in steps]))

    # ---- E2. no busy gauge -> cannot drain -> must REPORT ----------------
    blind = run_task("context", overrides, target=BUSY_BLIND_TARGET,
                     timeout=600, top=no_retry)
    if not blind:
        check(False, "gauge-blind run completed")
        return

    bsteps = [c for c in (blind.get("contexts") or []) if c.get("samples")]
    bflags = [c for c in bsteps if c.get("prefill_contaminated")]
    bwaits = [c.get("queue_wait_ms") for c in bsteps]
    print(f"       E2 blind:  flagged={len(bflags)}/{len(bsteps)}  "
          f"queue_wait_ms={bwaits}")
    check(bool(bflags),
          "an engine that cannot be drained produces a flagged point",
          str(bwaits))
    if bflags:
        warn = "; ".join(blind.get("data_warnings") or [])
        check("非 prefill 的等待" in warn,
              "and the run warns instead of publishing the number",
              warn[:170] or "(none)")
        worst = max(bflags, key=lambda c: c.get("queue_wait_ms") or 0)
        check((worst.get("queue_wait_ms") or 0) > 1000,
              "with the wait quantified",
              f"{worst.get('queue_wait_ms')} ms")
        # the engine's own reading must survive as the usable number, and it
        # must NOT be the contaminated one
        srv = [c.get("server_prefill_tps") for c in bsteps
               if c.get("server_prefill_tps")]
        check(bool(srv),
              "and the engine's own prefill rate remains the usable reading",
              str([round(v, 1) for v in srv]))
        check(all((c.get("server_prefill_tps") or 0)
                  > (c.get("prefill_tps") or 0)
                  for c in bflags),
              "the engine's rate is higher than the queue-inflated one",
              str([(c.get("prefill_tps"), c.get("server_prefill_tps"))
                   for c in bflags]))

    # ---- E3. no engine-reported prefill time -> no claim -----------------
    # Claiming contamination without evidence would be its own kind of lie.
    for c in bsteps + steps:
        for s in c.get("samples") or []:
            if s.get("ok") and s.get("server_prompt_ms") is None:
                check(s.get("prefill_contaminated") in (None, False),
                      "no contamination is claimed without the engine's own "
                      "prefill time", str(s.get("prefill_contaminated")))


# ---------------------------------------------------------------------- #
def main() -> int:
    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, MOCK_ARGS[0]), *MOCK_ARGS[1:]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_engine.py"),
             "--flavor", "llamacpp-vanilla", "--port", str(OPEN_PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_engine.py"),
             "--flavor", "vllm", "--port", str(COALESCE_PORT), "--coalesce"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_engine.py"),
             "--flavor", "vllm", "--port", str(BURST_PORT),
             "--tokens-per-chunk", str(BURST_TOKENS_PER_CHUNK)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_engine.py"),
             "--flavor", "llamacpp-fork", "--port", str(BUSY_PORT),
             "--prefill-seconds", str(BUSY_PREFILL_S),
             "--abandon-after", str(BUSY_ABANDON_AFTER_S),
             "--abandon-above", str(BUSY_ABANDON_ABOVE_TOKENS),
             "--serialize"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        # same abandonment, but a flavour with NO metrics endpoint - the client
        # cannot tell it is busy, so the wait cannot be avoided
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_engine.py"),
             "--flavor", BUSY_BLIND_FLAVOUR, "--port", str(BUSY_BLIND_PORT),
             "--prefill-seconds", str(BUSY_PREFILL_S),
             "--abandon-after", str(BUSY_ABANDON_AFTER_S),
             "--abandon-above", str(BUSY_ABANDON_ABOVE_TOKENS),
             "--serialize"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for port in (MOCK_PORT, OPEN_PORT, COALESCE_PORT, BURST_PORT,
                     BUSY_PORT, BUSY_BLIND_PORT):
            for _ in range(60):
                st, _ = fetch(f"http://127.0.0.1:{port}/v1/models", timeout=3)
                if st == 200:
                    break
                time.sleep(0.2)
            else:
                check(False, f"mock on {port} became ready")
                return 1
        print(f"started mocks on {MOCK_PORT} (cache locked on), "
              f"{OPEN_PORT} (cache controllable), "
              f"{COALESCE_PORT} (flush-in-one-burst), "
              f"{BURST_PORT} ({BURST_TOKENS_PER_CHUNK} tokens/event) and "
              f"{BUSY_PORT} (abandons long requests while staying busy) and "
              f"{BUSY_BLIND_PORT} (same, but no metrics endpoint at all)")

        layer_a()
        layer_b()
        layer_c()
        layer_d()
        layer_e()
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
    print("ALL CONSISTENCY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
