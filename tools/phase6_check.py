"""Phase 6 integration check: concurrency sweep.

Three layers:

  A. in-process   - ladder parsing and the concurrency config clamps
  B. mock + load  - a real sweep against the mock, asserting that throughput
                    actually *scales with concurrency* (a harness that awaited
                    requests one by one would produce a flat line and pass
                    every other check)
  C. mock + crash - the engine dies mid-sweep, so the run must abort and say
                    so rather than reporting throughput for a dead server

Prerequisite: the backend must be running on 127.0.0.1:8000.

Usage:
    python tools/phase6_check.py
"""

from __future__ import annotations

import argparse
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

MOCK_PORT_LOAD = 9301
MOCK_PORT_CRASH = 9302
#: an engine that will only run 2 generations at once (vLLM's --max-num-seqs)
MOCK_PORT_QUEUE = 9303

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def fetch(url: str, payload=None, timeout: float = 900.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    try:
        with NO_PROXY.open(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw.decode()) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def start_mock(port: int, *extra: str) -> subprocess.Popen:
    cmd = [sys.executable, os.path.join(HERE, "mock_engine.py"),
           "--flavor", "llamacpp-fork", "--port", str(port), *extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    for _ in range(60):
        status, _body = fetch(f"http://127.0.0.1:{port}/v1/models", timeout=2)
        if status == 200:
            return proc
        time.sleep(0.25)
    proc.terminate()
    raise SystemExit(f"mock engine on {port} did not start")


def run_concurrency(base_url: str, cfg: dict) -> dict | None:
    status, run = fetch(f"{API}/api/runs", {
        "base_url": base_url, "preset": "concurrency", "concurrency": cfg,
    })
    if status != 200:
        check(False, "create concurrency run", str(run)[:200])
        return None
    rid = run["run_id"]
    deadline = time.time() + 900
    while time.time() < deadline:
        _st, doc = fetch(f"{API}/api/runs/{rid}", timeout=60)
        if isinstance(doc, dict) and doc.get("status") not in (
                "queued", "running", "warmup", "testing", "connect", "pending"):
            return doc
        time.sleep(2)
    check(False, "concurrency run finished", "timeout waiting for the run")
    return None


def unwrap(doc: dict) -> tuple[dict, dict, str]:
    inner = doc.get("result") or {}
    return inner, (inner.get("result") or {}), doc.get("status")


# ---------------------------------------------------------------------- #
# A. in-process
# ---------------------------------------------------------------------- #
def layer_a() -> None:
    print("\n=== A. ladder + config clamps ===")
    from app.config import HARD_MAX_CONCURRENCY
    from app.params import build_concurrency_config, parse_ladder

    ladder, warn = parse_ladder("1,2,4,8,16,32", label="concurrency_ladder")
    check(ladder == [1, 2, 4, 8, 16, 32], "the default ladder parses", str(ladder))
    check(warn == [], "and needs no warnings", str(warn))

    bad, warn = parse_ladder("1,2,oops,8", label="concurrency_ladder")
    check(bad == [1, 2, 8] and bool(warn),
          "garbage entries are dropped with a warning", f"{bad} {warn}")

    cfg, warn = build_concurrency_config("m", {"concurrency_ladder": "1,4,8"})
    check(cfg["concurrency_ladder"] == [1, 4, 8],
          "an explicit ladder is honoured", str(cfg["concurrency_ladder"]))
    check(cfg["requests_per_level"] == 12,
          "requests_per_level defaults to 12", str(cfg["requests_per_level"]))
    # the shipped default must not be the flattering one
    check((cfg["options"] or {}).get("cache_prompt") is not True,
          "prompt caching is not enabled by default",
          json.dumps(cfg["options"]))

    cfg, warn = build_concurrency_config(
        "m", {"concurrency_ladder": f"1,{HARD_MAX_CONCURRENCY + 5}"})
    check(all(c <= HARD_MAX_CONCURRENCY for c in cfg["concurrency_ladder"]),
          "levels above the hard cap are removed", str(cfg["concurrency_ladder"]))
    check(any("硬上限" in w for w in warn),
          "and the removal is explained", "; ".join(warn)[:120])

    cfg, warn = build_concurrency_config(
        "m", {"concurrency_ladder": "1,32"}, server_slots=4)
    check(cfg.get("queues_above_slots") is True,
          "a ladder past the engine's slots is flagged as queueing")
    check(any("排队" in w for w in warn),
          "and the report is told not to read it as parallel capacity",
          "; ".join(warn)[:140])

    cfg, warn = build_concurrency_config(
        "m", {"concurrency_ladder": "1,32"}, server_slots=32)
    check(not cfg.get("queues_above_slots"),
          "a ladder within the slot count is not flagged")

    cfg, warn = build_concurrency_config("m", {"context_ladder": "512"})
    check(any("无关" in w for w in warn),
          "a parameter from another task is reported, not silently applied",
          "; ".join(warn)[:100])
    cfg, warn = build_concurrency_config("m", {"nonsense_key": "x"})
    check(any("未知参数" in w for w in warn),
          "and a parameter the registry has never heard of is reported too",
          "; ".join(warn)[:100])

    # ---- the ladder must not out-reach the requests per level -----------
    # Effective concurrency cannot exceed the number of requests in flight, so
    # a level of 32 fired with 12 requests measures 12.  Measured on a real
    # node with the shipped defaults: effective_concurrency read 10.3 / 8.9 /
    # 11.2 / 11.1 at levels 16 / 32 / 64 / 128 - the top half of the table was
    # all the same measurement, which is what made the curve look flat and
    # non-monotonic.
    #
    # The fix must NOT be "silently raise the sample count".  Measured on the
    # same node in the same session: a pinned 3-sample level became a
    # 192-request run, 10.7x the cost (82 s vs ~8 s) - the user's next words
    # were "变慢了非常多".  A pinned value is honoured; sizing is only derived
    # when the caller left it at the default.
    cfg, warn = build_concurrency_config("m", {"concurrency_ladder": "1,64"})
    check(cfg["concurrency_ladder"] == [1, 64],
          "a ladder above the default request count keeps its levels",
          str(cfg["concurrency_ladder"]))
    check(cfg.get("requests_per_level_mode") == "waves",
          "an un-pinned request count switches to per-level sizing",
          str(cfg.get("requests_per_level_mode")))
    check(cfg["requests_per_level"] == 12,
          "and the displayed floor stays the default",
          str(cfg["requests_per_level"]))
    check(any("至少 2×并发数" in w for w in warn),
          "the derived per-level sizes are stated",
          "; ".join(warn)[:150])

    # a pinned value is never rewritten - that is the whole point
    cfg, warn = build_concurrency_config(
        "m", {"concurrency_ladder": "1,2,4,8,16,32", "requests_per_level": 3})
    check(cfg["requests_per_level"] == 3,
          "an explicitly pinned request count is left exactly alone",
          str(cfg["requests_per_level"]))
    check(cfg.get("requests_per_level_mode") == "flat",
          "and the run fires exactly that many per level",
          str(cfg.get("requests_per_level_mode")))
    check(any("采样不足" in w for w in warn),
          "the levels that cannot measure themselves are named up front",
          "; ".join(warn)[:150])

    # bounded by the per-run total: trim the ladder, do not inflate every level
    from app.config import HARD_MAX_REQUESTS
    big = ",".join(str(c) for c in (1, 2, 4, 8, 16, 32, 64, 128, 256))
    cfg, warn = build_concurrency_config("m", {"concurrency_ladder": big})
    eff_total = sum(max(cfg["requests_per_level"], 2 * c)
                    for c in cfg["concurrency_ladder"])
    check(eff_total <= HARD_MAX_REQUESTS,
          "the derived sizing respects the per-run total cap",
          f"{eff_total} vs {HARD_MAX_REQUESTS}")
    check(max(cfg["concurrency_ladder"]) < 256,
          "achieved by trimming the top of the ladder, not inflating every level",
          str(cfg["concurrency_ladder"]))
    check(any("已移除最高档" in w for w in warn),
          "and the dropped levels are stated", "; ".join(warn)[:150])

    # a ladder comfortably under the request count invites no complaint
    cfg, warn = build_concurrency_config(
        "m", {"concurrency_ladder": "1,2,4", "requests_per_level": 40})
    check(not any("并发" in w and "请求数" in w for w in warn),
          "a well-provisioned ladder raises no provisioning warning",
          "; ".join(warn)[:120] or "(none)")
    check(not any("自动确定" in w for w in warn),
          "and no derived-sizing note either, because the value was pinned",
          "; ".join(warn)[:120] or "(none)")


# ---------------------------------------------------------------------- #
def layer_analysis() -> None:
    """The analysis that separates the two causes of a flat/non-monotonic
    throughput curve: a harness that never overlapped, versus an engine that
    gets slower when batched.  They look identical in a throughput column and
    have opposite remedies."""
    print("\n=== D. batching profile + evidence-based diagnosis ===")
    from app.engine.concurrency import (
        _batching_profile,
        _concurrency_warnings,
        _engine_evidence,
    )

    def level(c, tpot, tok_tps, eff, req_tps=None, requests=None,
              success=1.0, engine_mean=None, engine_max=None, engine_wait=None):
        # Two waves, which is what the harness actually fires now: the batching
        # analysis refuses to describe a curve from a handful of samples.
        return {
            "concurrency": c,
            "requests": requests if requests is not None else max(8, 2 * c),
            "requests_ok": 0,
            "wall_s": 1.0,
            "request_throughput": req_tps if req_tps is not None else 1.0,
            "token_throughput": tok_tps,
            "input_throughput": 0.0,
            "effective_concurrency": eff,
            "engine_running_mean": engine_mean,
            "engine_running_max": engine_max,
            "engine_waiting_max": engine_wait,
            "engine_queued": bool(engine_wait and engine_wait > 0),
            "achievable_concurrency": min(c, requests if requests is not None else c),
            # mirrors the product: not enough requests fired, OR enough fired
            # but the wave shape still pulled the mean in-flight count down
            "underprovisioned": bool(
                (requests is not None and requests < c)
                or (eff is not None and eff < 0.8 * c)),
            "success_rate": success,
            "error_rate": 0.0 if success else 1.0,
            "errors": [],
            "failure_kind": None,
            "ttft_ms": {"mean": 200.0, "p50": 200.0, "p95": 300.0,
                        "p99": 400.0, "min": 150.0, "max": 500.0},
            "e2e_ms": {"mean": 2000.0, "p50": 2000.0, "p95": 2500.0,
                       "p99": 3000.0, "min": 1000.0, "max": 4000.0},
            "tpot_ms": tpot,
            "decode_tps": round(1000 / tpot, 2),
            "output_tokens_total": 64,
            "samples": [],
        }

    # the real shape: batch-1 is a fast path, batched levels are all alike and
    # the aggregate never beats the baseline (190 -> 72 -> 136 ... tok/s)
    real = [level(1, 5.29, 138.4, 1.00),
            level(2, 20.65, 81.2, 1.99),
            level(4, 24.97, 136.7, 3.91),
            level(8, 26.70, 198.1, 7.45)]
    b = _batching_profile(real)
    check(bool(b), "a batching profile is produced", str(sorted(b.keys())[:5]))
    check(abs(b["batching_tax_x"] - (24.97 / 5.29)) < 0.05,
          "the per-request cost of batching is measured against concurrency 1",
          str(b["batching_tax_x"]))
    check(b["batch1_fast_path"] is False or b["throughput_gain_x"] > 1.2,
          "this shape is NOT called a batch-1 fast path (it still gains)",
          f"gain={b['throughput_gain_x']} tax={b['batching_tax_x']}")

    # the case the user hit: no gain at all despite the tax
    flat = [level(1, 5.29, 138.4, 1.00),
            level(2, 20.65, 81.2, 1.99),
            level(4, 24.97, 136.7, 3.91),
            level(8, 26.70, 141.0, 7.45)]
    bf = _batching_profile(flat)
    check(bf["batch1_fast_path"] is True,
          "a tax with no throughput gain is flagged as a batch-1 fast path",
          f"gain={bf['throughput_gain_x']} tax={bf['batching_tax_x']}")
    msgs = _concurrency_warnings(flat, None, {}, bf)
    check(any("并发没有换来吞吐" in m for m in msgs),
          "and the report says so in words",
          "; ".join(msgs)[:170] or "(none)")

    # ---- under-provisioned levels must be named ------------------------
    prov = [level(1, 5.0, 190.0, 1.0),
            level(8, 26.0, 140.0, 6.0),
            level(32, 26.0, 150.0, 11.1, requests=12),
            level(128, 26.0, 151.0, 11.1, requests=12)]
    msgs = _concurrency_warnings(prov, None, {}, _batching_profile(prov))
    check(any("不可能超过它发出的请求数" in m for m in msgs),
          "levels that fired fewer requests than they claim are reported",
          "; ".join(msgs)[:170] or "(none)")
    check(any("128 档只发了 12 个请求" in m for m in msgs),
          "with the offending levels named", "; ".join(msgs)[:170])

    # ---- the engine saturates: above its ceiling throughput is flat --------
    # Real shape from a live vLLM node (--max-num-seqs=16): levels 32 and
    # 64 measured 484 / 473 tok/s while the engine ran at most 16 at once and
    # queued up to 48.  Past the ceiling extra requests only buy latency, so the
    # report must state that capacity is reached rather than imply that more
    # concurrency would help - and must not "correct" the capacity downwards
    # just because those levels queued.
    sat = [level(1, 5.28, 139.5, 1.00, requests=12, engine_mean=0.68,
                 engine_max=1, engine_wait=0),
           level(8, 26.0, 224.7, 7.53, requests=16, engine_mean=6.78,
                 engine_max=8, engine_wait=0),
           level(32, 28.0, 484.0, 27.43, requests=64, engine_mean=14.43,
                 engine_max=16, engine_wait=16),
           level(64, 28.0, 473.1, 51.21, requests=128, engine_mean=14.77,
                 engine_max=16, engine_wait=48)]
    msgs = _concurrency_warnings(sat, None, {}, _batching_profile(sat))
    joined = " | ".join(msgs)
    check(any("引擎已饱和" in m and "容量就是这个水平" in m for m in msgs),
          "a saturated engine is described by its capacity, not its scaling",
          joined[:240] or "(none)")
    check(any("--max-num-seqs" in m for m in msgs),
          "and the engine's own parallelism limit is named",
          joined[:240] or "(none)")
    check(any("引擎在并发 32 档就已到顶" in m for m in msgs),
          "the effective-concurrency column is disowned where queueing inflates it",
          joined[:240] or "(none)")
    # The quoted eff must be the real maximum over all levels, and the two
    # numbers must not be swapped - a live run printed "highest eff 64 (at level
    # 51.06)" because a (level, eff) tuple was unpacked the other way round.
    check(any("实测并发度最高 51.21（并发 64 档）" in m for m in msgs),
          "the highest measured concurrency is quoted with its own level",
          joined[:240] or "(none)")

    # ---- the exact shape the user's run produced (node .29.21, 12:30) -----
    # Two different problems in one table, and they must not be conflated:
    #   * level 32 fired exactly one wave of 32 requests, so its mean in-flight
    #     count only reached 22.69 (= 0.71*32) - not enough requests is NOT the
    #     explanation, the wave shape is;
    #   * the 1 -> 2 step FELL in aggregate (125.5 -> 76.0 tok/s) because this
    #     engine decodes at 5.3 ms/token alone and ~27 ms/token once a second
    #     sequence is in flight.  Reproduced outside the harness: 107.9 -> 61.8
    #     -> 105.4 tok/s at concurrency 1/2/4 with zero preemptions/waiting, so
    #     it is an engine property, not a measurement artifact.
    observed = [level(1, 5.28, 125.5, 1.00, req_tps=1.961, requests=32),
                level(2, 23.72, 76.0, 2.00, req_tps=1.188, requests=32),
                level(4, 26.74, 122.7, 3.89, req_tps=1.917, requests=32),
                level(8, 28.87, 219.1, 7.45, req_tps=3.424, requests=32),
                level(16, 27.37, 462.7, 14.99, req_tps=7.229, requests=32),
                level(32, 28.08, 466.0, 22.69, req_tps=7.282, requests=32)]
    ob = _batching_profile(observed)
    check((ob.get("batching_tax_x") or 0) >= 4.0,
          "the observed curve yields the batching tax",
          str(ob.get("batching_tax_x")))
    msgs = _concurrency_warnings(observed, None, {}, ob)
    joined = " | ".join(msgs)
    check(any("32 档实测 22.69" in m for m in msgs),
          "a single-wave level is reported by its measured shortfall",
          joined[:200] or "(none)")
    check(not any("不可能超过它发出的请求数" in m for m in msgs),
          "and is not mislabelled as an arithmetic shortfall (it had 32 requests)",
          joined[:200] or "(none)")
    check(any("1→2" in m and "先跌再涨" in m for m in msgs),
          "the 1->2 collapse is attributed to the batch-1 fast path",
          joined[:200] or "(none)")
    check(not any("不能归因于" in m for m in msgs),
          "so the vague 'the drop must have another cause' note is not added",
          joined[:200] or "(none)")

    # ---- a table that measured thinking, not answering -------------------
    # Every output token of the observed run was reasoning (64 of 64), because
    # the engine thinks by default and max_tokens was 64.
    thinking = [level(1, 5.28, 125.5, 1.00)]
    thinking[0]["samples"] = [{"ok": True, "output_tokens": 64,
                               "reasoning_tokens": 64} for _ in range(3)]
    msgs = _concurrency_warnings(thinking, None, {},
                                 _batching_profile(thinking))
    check(any("全部输出 token 都是思考链" in m for m in msgs),
          "an all-reasoning sweep says so instead of calling it decode speed",
          "; ".join(msgs)[:180] or "(none)")

    answering = [level(1, 5.28, 125.5, 1.00)]
    answering[0]["samples"] = [{"ok": True, "output_tokens": 64,
                                "reasoning_tokens": 0}]
    msgs = _concurrency_warnings(answering, None, {},
                                 _batching_profile(answering))
    check(not any("全部输出 token 都是思考链" in m for m in msgs),
          "and a normal sweep is not accused of it",
          "; ".join(msgs)[:180] or "(none)")

    # ---- a 3-sample ladder must not be read as a hardware ceiling --------
    # Real run 75ee8b8b (a pinned 3 requests per level) produced a confident
    # "this engine tops out at 2.9-way concurrency" and "concurrency never pays
    # off".  Both statements were the sample count talking, not the engine.
    thin = [level(1, 5.86, 124.3, 1.00, requests=3),
            level(2, 17.97, 85.0, 1.81, requests=3),
            level(4, 17.83, 68.0, 2.13, requests=3),
            level(8, 24.50, 105.8, 2.91, requests=3)]
    check(_batching_profile(thin) == {},
          "a ladder with too few samples per level yields no batching profile",
          str(_batching_profile(thin))[:140])
    msgs = _concurrency_warnings(thin, None, {}, _batching_profile(thin))
    joined = " | ".join(msgs)
    check("天花板" not in joined,
          "so no hardware concurrency ceiling is claimed from 3 samples",
          joined[:200] or "(none)")
    check("并发没有换来吞吐" not in joined,
          "and no batching verdict is drawn from it either",
          joined[:200] or "(none)")
    check(any("只发了 3 个请求" in m for m in msgs),
          "while the sampling shortfall itself is still reported",
          joined[:200] or "(none)")

    # ---- a level where everything failed is a failure, not dilution ------
    broken = [level(1, 5.0, 190.0, 1.0),
              level(2, 26.0, 0.0, 0.0, success=0.0, requests=8)]
    msgs = _concurrency_warnings(broken, None, {}, _batching_profile(broken))
    check(any("100% 请求失败" in m for m in msgs),
          "a fully-failing level is reported as a failure",
          "; ".join(msgs)[:180] or "(none)")
    check(not any("只发了一波" in m for m in msgs),
          "and is not also described as diluted sampling",
          "; ".join(msgs)[:180] or "(none)")

    # ---- the engine's own view: queueing is not parallelism ---------------
    # A live vLLM at nominal 32: eff 28.58, engine running max 16, queued max
    # 16.  Every client-side number looked like 32-way parallelism.
    capped = [level(1, 5.2, 138.0, 1.00, engine_mean=1.0, engine_max=1),
              level(16, 27.9, 383.0, 14.6, engine_mean=12.0, engine_max=16),
              level(32, 29.3, 461.0, 28.58, engine_mean=13.63, engine_max=16,
                    engine_wait=16)]
    msgs = _concurrency_warnings(capped, None, {}, _batching_profile(capped))
    joined = " | ".join(msgs)
    check(any("引擎自己在排队" in m for m in msgs),
          "a level the engine queued is reported as queueing, not parallelism",
          joined[:220] or "(none)")
    check("16" in " ".join(m for m in msgs if "引擎自己在排队" in m),
          "and the engine's own running ceiling is quoted",
          joined[:220] or "(none)")
    check(any("高于引擎" in m for m in msgs),
          "and eff above the engine's ceiling is called out as queueing",
          joined[:220] or "(none)")

    # the same shape with NO engine gauges must not invent the conclusion
    blind = [level(1, 5.2, 138.0, 1.00), level(32, 29.3, 461.0, 28.58)]
    msgs = _concurrency_warnings(blind, None, {}, _batching_profile(blind))
    check(not any("引擎自己在排队" in m for m in msgs),
          "without gauges no queueing is invented",
          "; ".join(msgs)[:180] or "(none)")

    # ---- in-load evidence outranks the boundary snapshot ------------------
    # The before/after snapshot is taken with the engine IDLE, so it reported a
    # reassuring zero for a run whose in-load gauges peaked at 16 queued.  The
    # in-load reading must win, and the difference must be recorded.
    idle_snapshot = {
        "before": {"normalized": {"preemptions_total": 0.0,
                                  "requests_waiting": 0.0,
                                  "requests_success_total": 100.0,
                                  "generation_tokens_total": 1000.0}},
        "after": {"normalized": {"preemptions_total": 0.0,
                                 "requests_waiting": 0.0,
                                 "requests_success_total": 172.0,
                                 "generation_tokens_total": 2000.0}},
    }
    ev = _engine_evidence(idle_snapshot)
    check(ev.get("measured_in_load") is False,
          "a boundary-only snapshot is marked as not measured in load",
          str(ev.get("measured_in_load")))
    check(ev.get("signs_of_overload") == [],
          "so it cannot rule queueing out on its own",
          str(ev.get("signs_of_overload")))
    ev2 = _engine_evidence(idle_snapshot, capped)
    check(ev2.get("measured_in_load") is True,
          "with in-load gauges the evidence is marked as such",
          str(ev2.get("measured_in_load")))
    check(ev2.get("peak_waiting") == 16,
          "the in-load queue peak is carried", str(ev2.get("peak_waiting")))
    check(ev2.get("boundary_waiting") == 0,
          "next to the idle snapshot's misleading zero",
          str(ev2.get("boundary_waiting")))
    check("requests_waiting" in (ev2.get("signs_of_overload") or []),
          "and queueing becomes a sign of overload",
          str(ev2.get("signs_of_overload")))
    falling = [level(1, 5.0, 190.0, 1.0, req_tps=2.1),
               level(2, 21.0, 82.0, 1.99, req_tps=1.2),
               level(4, 25.0, 137.0, 3.91, req_tps=2.1)]
    msgs = _concurrency_warnings(falling, None, ev, None)
    check(any("不足以保证负载中没有排队" in m for m in msgs),
          "a claim of 'no queueing' from boundary data alone is qualified",
          "; ".join(msgs)[:200] or "(none)")
    msgs = _concurrency_warnings(falling, None, ev2, None)
    check(not any("不足以保证负载中没有排队" in m for m in msgs),
          "and is stated plainly once the gauges were sampled under load",
          "; ".join(msgs)[:200] or "(none)")

    # ---- a drop with no engine evidence is NOT blamed on the server -----
    drop = [level(1, 5.0, 190.0, 1.0, req_tps=2.1),
            level(2, 21.0, 82.0, 1.99, req_tps=1.2),
            level(4, 25.0, 137.0, 3.91, req_tps=2.1)]
    quiet = _engine_evidence({
        "before": {"normalized": {"preemptions_total": 0.0,
                                  "requests_waiting": 0.0,
                                  "requests_success_total": 100.0}},
        "after": {"normalized": {"preemptions_total": 0.0,
                                 "requests_waiting": 0.0,
                                 "requests_success_total": 172.0}},
    })
    check(quiet.get("signs_of_overload") == [],
          "an engine reporting no overload produces no signs",
          str(quiet.get("signs_of_overload")))
    msgs = _concurrency_warnings(drop, None, quiet, _batching_profile(drop))
    check(not any("通常意味着服务端开始拒请求" in m for m in msgs),
          "a drop with a quiet engine is not blamed on the server",
          "; ".join(msgs)[:170] or "(none)")
    joined = " ".join(msgs)
    check("没有" in joined and ("排队" in joined or "抢占" in joined)
          and "不是测量误差" in joined,
          "and the report states plainly that it is not overload, "
          "citing the engine's own silence",
          "; ".join(msgs)[:200] or "(none)")

    # ---- with evidence, the server hypothesis is allowed back ----------
    loud = _engine_evidence({
        "before": {"normalized": {"preemptions_total": 0.0,
                                  "requests_waiting": 0.0}},
        "after": {"normalized": {"preemptions_total": 17.0,
                                 "requests_waiting": 40.0}},
    })
    check(loud.get("signs_of_overload") == ["preemptions", "requests_waiting"],
          "preemptions and waiting are picked up as signs of overload",
          str(loud.get("signs_of_overload")))
    msgs = _concurrency_warnings(drop, None, loud, _batching_profile(drop))
    check(any("--max-num-seqs" in m for m in msgs),
          "with evidence, the concurrency-limit advice is given",
          "; ".join(msgs)[:170] or "(none)")

    # a quiet engine whose drop is NOT explained by batching either: the report
    # must still refuse to blame the server, and point somewhere useful
    mild = [level(1, 5.0, 190.0, 1.0, req_tps=2.1),
            level(2, 6.0, 190.0, 1.99, req_tps=1.2),
            level(4, 6.2, 195.0, 3.91, req_tps=2.1)]
    msgs = _concurrency_warnings(mild, None, quiet, _batching_profile(mild))
    check(not any("通常意味着服务端开始拒请求" in m for m in msgs),
          "a drop without any overload evidence never blames the server",
          "; ".join(msgs)[:170] or "(none)")

    # ---- no engine metrics at all -> claim nothing ---------------------
    check(_engine_evidence({}) == {},
          "without a metrics snapshot no evidence is invented",
          str(_engine_evidence({})))

    # ---- a 1->2 cliff even when the ladder as a whole gains -------------
    # This is the shape that was reported as "non-linear": the first step
    # halves the throughput, then it climbs past the baseline.  Naming the
    # first step is what makes the curve readable.
    climbing = [level(1, 5.25, 132.8, 1.0),
                level(2, 23.34, 76.2, 2.0),
                level(4, 26.14, 128.9, 3.9),
                level(8, 28.10, 234.1, 7.7),
                level(16, 27.43, 463.1, 15.3)]
    bc = _batching_profile(climbing)
    check(bc["batch1_fast_path"] is False,
          "an overall gain is not called a flat curve",
          f"gain={bc['throughput_gain_x']}")
    check(bc["batching_tax_x"] >= 2.0,
          "but the per-request batch tax is still measured",
          str(bc["batching_tax_x"]))
    msgs = _concurrency_warnings(climbing, None, quiet, bc)
    check(any("1→2" in m and "先跌再涨" in m for m in msgs),
          "the 1->2 cliff is named, with the numbers and the recovery point",
          "; ".join(msgs)[:200] or "(none)")
    check(any("不是测量误差" in m for m in msgs),
          "and it is not attributed to the harness or to overload",
          "; ".join(msgs)[:200] or "(none)")


# ---------------------------------------------------------------------- #
# B. a real sweep against the mock
# ---------------------------------------------------------------------- #
def layer_b() -> None:
    print("\n=== B. throughput actually scales with concurrency ===")
    # the mock serves each request in ~ttft + n*20ms; with 8 requests per level
    # a *sequential* harness would take 8x longer at every level and show a
    # flat throughput line
    proc = start_mock(MOCK_PORT_LOAD, "--context", "65536")
    try:
        doc = run_concurrency(f"http://127.0.0.1:{MOCK_PORT_LOAD}", {
            "concurrency_ladder": "1,4,8",
            "requests_per_level": 8,
            "max_tokens": 8,
            "warmup": 0,
        })
    finally:
        proc.terminate()
    if not doc:
        return

    inner, res, status = unwrap(doc)
    check(status == "completed", "the sweep completed", str(status))
    levels = inner.get("levels") or []
    by_c = {l["concurrency"]: l for l in levels}
    check(set(by_c) == {1, 4, 8}, "every ladder level produced a batch",
          str(sorted(by_c)))

    for c, l in sorted(by_c.items()):
        print(f"       C={c:<2} wall={l['wall_s']:.2f}s "
              f"req/s={l['request_throughput']:.2f} "
              f"tok/s={l['token_throughput']:.1f} "
              f"ttft p50/p95={l['ttft_ms']['p50']:.0f}/{l['ttft_ms']['p95']:.0f}ms "
              f"ok={l['success_rate']:.2f}")

    check(all(l["requests"] == 8 for l in levels),
          "each level issued the requested number of requests")
    check(all(l["success_rate"] == 1.0 for l in levels),
          "every request succeeded on the mock")

    one, four, eight = by_c[1], by_c[4], by_c[8]
    check(four["request_throughput"] > one["request_throughput"] * 2,
          "4-way concurrency beats sequential by more than 2x",
          f"{one['request_throughput']:.2f} -> {four['request_throughput']:.2f} req/s")
    check(eight["request_throughput"] > one["request_throughput"] * 2.5,
          "8-way concurrency is still far above sequential",
          f"{one['request_throughput']:.2f} -> {eight['request_throughput']:.2f} req/s")

    # The assertion that actually matters, and the one a serialising harness
    # would fail: are requests genuinely in flight together?  Little's law on
    # the samples themselves - sum of per-request durations / wall clock.
    print("       effective concurrency (sum of request time / wall clock): "
          + ", ".join(f"C={c}:{l['effective_concurrency']}"
                      for c, l in sorted(by_c.items())))
    check(abs((one["effective_concurrency"] or 0) - 1.0) < 0.25,
          "at concurrency 1 exactly one request is in flight",
          str(one["effective_concurrency"]))
    check((four["effective_concurrency"] or 0) > 2.5,
          "at concurrency 4 the measured overlap is >2.5",
          str(four["effective_concurrency"]))
    # the mock itself saturates around 3-4 concurrent handlers, so this is a
    # floor rather than a monotonicity claim: what it rules out is a harness
    # that quietly serialises
    check((eight["effective_concurrency"] or 0) > 2.5,
          "and concurrency 8 is still genuinely overlapped",
          str(eight["effective_concurrency"]))

    # one batch of 8 at level 8 should take about as long as a single request
    single = one["wall_s"]
    check(eight["wall_s"] < single * 2.5,
          "8 parallel requests take about as long as one (not 8x)",
          f"1->{single:.2f}s  8->{eight['wall_s']:.2f}s")

    # throughput must be consistent with the tokens actually produced
    for c, l in sorted(by_c.items()):
        expected = l["request_throughput"] * l["output_tokens_total"] / l["requests"]
        check(abs(l["token_throughput"] - expected) < 0.05 * max(expected, 1),
              f"C={c}: token throughput matches request throughput x output tokens",
              f"{l['token_throughput']:.2f} vs {expected:.2f}")

    for c, l in sorted(by_c.items()):
        t = l["ttft_ms"]
        check(t["p50"] <= t["p95"] <= t["p99"],
              f"C={c}: TTFT percentiles are ordered",
              f"p50={t['p50']:.0f} p95={t['p95']:.0f} p99={t['p99']:.0f}")
        check(l["ttft_ms"]["p99"] is not None,
              f"C={c}: p99 exists (enough samples per level)", str(l["requests"]))

    check(res.get("peak_token_throughput") is not None,
          "peak token throughput reported", str(res.get("peak_token_throughput")))
    check(res.get("at_concurrency") == max(by_c, key=lambda c: by_c[c]["token_throughput"]),
          "the peak is attributed to the right level", str(res.get("at_concurrency")))
    check(bool(res.get("scaling")), "the scaling summary is present")

    # a unique marker per request means the engine reports no cache reuse
    hits = [s.get("cache_hit_tokens") for l in levels for s in l["samples"]
            if s.get("ok")]
    check(all((h or 0) == 0 for h in hits),
          "no prompt-cache reuse across concurrent requests",
          f"{len(hits)} samples, max hit {max([h or 0 for h in hits], default=0)}")


# ---------------------------------------------------------------------- #
# C. the engine dies mid-sweep
# ---------------------------------------------------------------------- #
def layer_c() -> None:
    print("\n=== C. engine dies mid-sweep -> abort, not a fake throughput ===")
    proc = start_mock(MOCK_PORT_CRASH, "--context", "65536", "--die-after", "6")
    try:
        doc = run_concurrency(f"http://127.0.0.1:{MOCK_PORT_CRASH}", {
            "concurrency_ladder": "1,2,4",
            "requests_per_level": 6,
            "max_tokens": 8,
            "warmup": 0,
            "retries": 0,
        })
    finally:
        proc.terminate()
    if not doc:
        return

    inner, res, status = unwrap(doc)
    check(status == "partial",
          "the sweep is reported partial, not completed", str(status))
    check(res.get("aborted") is True, "the result records the abort")
    check("失联" in str(res.get("abort_reason")),
          "and says the engine went away", str(res.get("abort_reason"))[:110])
    check(bool(inner.get("data_warnings")),
          "the report warns about the affected levels",
          "; ".join(inner.get("data_warnings") or [])[:130])

    levels = inner.get("levels") or []
    bad = [l for l in levels if l["error_rate"] > 0]
    check(bool(levels) and bool(bad),
          "at least one level shows the failures", f"{len(levels)} level(s)")
    if bad:
        check(bad[0].get("failure_kind") in ("unreachable", "too_long", "timeout"),
              "the failure is classified",
              f"C={bad[0]['concurrency']} kind={bad[0].get('failure_kind')}")
        check(bool(bad[0].get("errors")),
              "and the first error message is kept",
              str(bad[0]["errors"][0])[:90])


def layer_capacity() -> None:
    """The engine ran fewer requests at once than the client kept in flight.

    This is invisible from the client: 8 requests in flight with 2 running and
    6 queued looks exactly like 8-way parallelism in every timing we can take.
    Only the engine's own gauge, sampled *during* the level, can tell them
    apart - which is the whole reason the report carries it.  Reproduces, on a
    mock, what a live vLLM did at nominal 32 (running max 16, queued max 16).
    """
    print("\n=== F. engine-side parallelism: queueing is not parallelism ===")
    proc = start_mock(MOCK_PORT_QUEUE, "--max-running", "2")
    try:
        doc = run_concurrency(f"http://127.0.0.1:{MOCK_PORT_QUEUE}", {
            "concurrency_ladder": "8", "requests_per_level": 16,
            "max_tokens": 8, "warmup": 0,
        })
    finally:
        proc.terminate()
    if not doc:
        return

    inner, res, _status = unwrap(doc)
    levels = inner.get("levels") or []
    lv = levels[0] if levels else {}
    print(f"       C={lv.get('concurrency')} requests={lv.get('requests')} "
          f"eff={lv.get('effective_concurrency')} "
          f"engine mean/max={lv.get('engine_running_mean')}/"
          f"{lv.get('engine_running_max')} "
          f"queued_max={lv.get('engine_waiting_max')}")

    check(lv.get("engine_running_max") is not None,
          "the engine's own running count is recorded",
          str(lv.get("engine_running_max")))
    check((lv.get("engine_running_max") or 99) <= 2,
          "and it respects the engine's real capacity (2)",
          str(lv.get("engine_running_max")))
    check((lv.get("engine_waiting_max") or 0) > 0,
          "queueing inside the engine is visible",
          str(lv.get("engine_waiting_max")))
    check(lv.get("engine_queued") is True,
          "and the level is flagged as queueing",
          str(lv.get("engine_queued")))
    check((lv.get("effective_concurrency") or 0) > (lv.get("engine_running_max") or 0),
          "effective_concurrency overstates parallelism here - which is exactly "
          "why the engine's own number is published next to it",
          f"eff={lv.get('effective_concurrency')} vs "
          f"engine={lv.get('engine_running_max')}")
    warns = " ".join(inner.get("data_warnings") or [])
    check("引擎自己在排队" in warns,
          "the report says the level queued instead of parallelising",
          warns[:200] or "(none)")
    check(res.get("engine_parallelism_ceiling") == 2,
          "the engine's parallelism ceiling is reported",
          str(res.get("engine_parallelism_ceiling")))
    check(res.get("engine_queued_levels") == [8],
          "with the affected level listed",
          str(res.get("engine_queued_levels")))

    # ---- capacity is NOT filtered by queueing --------------------------
    # The level queued, but it also produced the most tokens per second, so it
    # IS the box's measured capacity.  Excluding it understated a live node by
    # half (headline 234.7 tok/s while levels 32/64 delivered 459/461).
    check(res.get("peak_token_throughput") == lv.get("token_throughput"),
          "the capacity figure is the best throughput at any offered load, "
          "queueing included",
          f"peak={res.get('peak_token_throughput')} "
          f"level={lv.get('token_throughput')}")
    check(res.get("engine_limited_capacity") is True,
          "and it is marked as having been reached through engine queueing",
          str(res.get("engine_limited_capacity")))
    check(res.get("parallel_peak_token_throughput") is None,
          "while the parallelism-only peak has nothing to report here",
          str(res.get("parallel_peak_token_throughput")))
    check(res.get("parallelism_levels") in ([], None),
          "and no level claims to demonstrate its own concurrency",
          str(res.get("parallelism_levels")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None,
                    help="also check a real engine, e.g. http://127.0.0.1:8080")
    ap.add_argument("--real-ladder", default="1,2,4")
    args = ap.parse_args()

    layer_a()
    layer_analysis()
    layer_b()
    layer_c()
    layer_capacity()
    if args.real:
        print(f"\n=== E. real engine {args.real} ===")
        doc = run_concurrency(args.real, {
            "concurrency_ladder": args.real_ladder,
            "requests_per_level": 4,
            "max_tokens": 32,
            "warmup": 0,
        })
        if doc:
            inner, res, status = unwrap(doc)
            for l in inner.get("levels") or []:
                print(f"       C={l['concurrency']:<2} wall={l['wall_s']:.2f}s "
                      f"req/s={l['request_throughput']:.2f} "
                      f"tok/s={l['token_throughput']:.1f} "
                      f"ttft p50/p95={l['ttft_ms']['p50'] or 0:.0f}/"
                      f"{l['ttft_ms']['p95'] or 0:.0f}ms "
                      f"ok={l['success_rate']:.2f}")
            check(bool(inner.get("levels")), "real engine produced levels",
                  f"status={status}")
            check(all(l["success_rate"] == 1.0 for l in inner.get("levels") or []),
                  "every real level succeeded")
            slots = res.get("server_slots")
            print(f"       server_slots={slots} "
                  f"queues_above_slots={res.get('queues_above_slots')}")

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED — {len(failures)} check(s): " + "; ".join(failures))
        sys.exit(1)
    print("ALL PHASE 6 CHECKS PASSED")


if __name__ == "__main__":
    main()
