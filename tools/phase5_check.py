"""Phase 5 integration check: Context benchmark + Maximum Verified Context.

Four layers:

  A. in-process  - ladder parsing, failure classification, ceiling arithmetic
  B. mock + clamp - a long run against a mock that *refuses* oversized inputs:
                    proves the ladder stops, the bracket is tight and the
                    binary search only probes inside it
  C. mock + crash - a run against a mock that *dies* under an oversized input
                    (the OOM-restart the real engine exhibited at 128K):
                    proves the harness aborts instead of inventing a ceiling
  D. real engine  - optional, only with ``--real <url>``

Prerequisite: the backend must be running on 127.0.0.1:8000.

Usage:
    python tools/phase5_check.py
    python tools/phase5_check.py --real http://127.0.0.1:8080
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

MOCK_PORT_REFUSE = 9201
MOCK_PORT_CRASH = 9202

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


# ---------------------------------------------------------------------- #
# mock engine lifecycle
# ---------------------------------------------------------------------- #
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


def run_context(base_url: str, payload_ctx: dict) -> dict | None:
    status, run = fetch(f"{API}/api/runs", {
        "base_url": base_url, "preset": "context", "context": payload_ctx,
    })
    if status != 200:
        check(False, "create context run", str(run)[:200])
        return None
    rid = run["run_id"]
    deadline = time.time() + 900
    while time.time() < deadline:
        _st, doc = fetch(f"{API}/api/runs/{rid}", timeout=60)
        if isinstance(doc, dict) and doc.get("status") not in (
                "queued", "running", "warmup", "testing", "connect", "pending"):
            return doc
        time.sleep(2)
    check(False, "context run finished", "timeout waiting for the run")
    return None


def unwrap(doc: dict) -> tuple[dict, dict, str]:
    """(run document, context result, run status)."""
    inner = doc.get("result") or {}
    return inner, (inner.get("result") or {}), doc.get("status")


def tested_entries(doc: dict) -> list[dict]:
    return [c for c in (doc.get("contexts") or [])
            if c.get("kind") == "context" and c.get("samples")]


# ---------------------------------------------------------------------- #
# A. in-process
# ---------------------------------------------------------------------- #
def layer_a() -> None:
    print("\n=== A. ladder parsing, failure classification, ceiling ===")
    from app.errors import (KIND_AUTH, KIND_TIMEOUT, KIND_TOO_LONG,
                            KIND_UNREACHABLE, classify_error)
    from app.errors import ApiError
    from app.params import parse_ladder

    ladder, warn = parse_ladder("512,1K,2K,4K")
    check(ladder == [512, 1024, 2048, 4096], "ladder parses 512,1K,2K,4K",
          f"{ladder} warn={warn}")
    ladder2, _ = parse_ladder("64K, 128K ,256K")
    check(ladder2 == [65536, 131072, 262144], "ladder parses 64K/128K/256K",
          str(ladder2))
    check(parse_ladder("nonsense")[0] == [],
          "ladder rejects garbage (caller falls back to the default)")

    check(classify_error("ConnectError: All connection attempts failed")
          == KIND_UNREACHABLE, "connection refused -> unreachable")
    check(classify_error("RemoteProtocolError: peer closed connection without "
                         "sending complete message body")
          == KIND_TOO_LONG, "dropped stream -> too_long (capacity)")
    check(classify_error("ReadTimeout: ") == KIND_TIMEOUT, "read timeout -> timeout")
    check(classify_error(ApiError("boom", status=400)) == KIND_TOO_LONG,
          "HTTP 400 -> too_long")
    check(classify_error(ApiError("boom", status=401)) == KIND_AUTH,
          "HTTP 401 -> auth")

    # the output budget competes with the input for the same window
    from app.config import HARD_MAX_CONTEXT
    from app.engine.context import _resolve_ceiling, scaled_timeout
    check(_resolve_ceiling({"server_context": 4096}) == 4096,
          "ceiling follows the engine's declared window")
    check(_resolve_ceiling({"server_context": HARD_MAX_CONTEXT * 4})
          == HARD_MAX_CONTEXT, "ceiling is capped by our hard limit")

    # a 64K step needs ~180s of prefill at the observed ~365 tok/s; a fixed
    # 120s timeout killed it mid-computation and took the engine down with it
    t, widened = scaled_timeout(65569, 64, 120.0)
    check(widened and t >= 400, "a 64K step widens the request timeout",
          f"{t:.0f}s widened={widened}")
    t, widened = scaled_timeout(2111, 64, 120.0)
    check(not widened and t == 120.0, "a 2K step keeps the default timeout",
          f"{t:.0f}s widened={widened}")
    t, widened = scaled_timeout(5_000_000, 64, 120.0)
    check(t <= 1800.0, "the widened timeout is still hard-capped",
          f"{t:.0f}s")
    t, widened = scaled_timeout(65569, 64, 900.0)
    check(t == 900.0 and not widened,
          "an explicit longer timeout is never shortened", f"{t:.0f}s")

    # ---- tokenizer calibration (for engines with no /tokenize) ---------
    from app.engine.context_builder import (
        _FILLER,
        _build_by_chars,
        calibrate,
    )
    # a simulated engine: 3.5 chars/token plus a 40-token chat template
    obs = [(1200, 40 + int(1200 / 3.5)), (4800, 40 + int(4800 / 3.5))]
    cal = calibrate(obs)
    check(abs(cal.chars_per_token - 3.5) < 0.05,
          "calibration recovers the simulated ratio",
          f"{cal.chars_per_token:.3f} chars/token")
    check(abs(cal.overhead_tokens - 40) < 3,
          "calibration separates the chat-template overhead",
          f"{cal.overhead_tokens:.1f} tokens")

    built = _build_by_chars(512, [_FILLER], cal)
    predicted = cal.overhead_tokens + len(built.text) / cal.chars_per_token
    check(abs(predicted - 512) <= 512 * 0.02,
          "a calibrated 512-token input is predicted within 2%",
          f"{len(built.text)} chars -> {predicted:.0f} tokens")
    check(built.source == "calibrated",
          "the built input declares itself calibrated", built.source)
    check(bool(built.notes) and "标定" in built.notes[0],
          "the calibration is explained in the result")

    # the old, cruder path still exists and still says so
    plain = _build_by_chars(512, [_FILLER], None)
    check(plain.source == "estimated" and len(plain.text) == 512 * 4,
          "without calibration it falls back to 4 chars/token and says so",
          f"{len(plain.text)} chars, source={plain.source}")

    degenerate = calibrate([])
    check(degenerate.chars_per_token > 0 and degenerate.overhead_tokens == 0.0,
          "calibration falls back safely with no observations",
          f"{degenerate.chars_per_token}")
    wild = calibrate([(100, 100), (200, 10000)])
    check(abs(wild.chars_per_token - 4.0) < 1e-9,
          "a nonsensical ratio is rejected in favour of the default",
          f"{wild.chars_per_token}")


# ---------------------------------------------------------------------- #
# B. a mock that refuses oversized inputs
# ---------------------------------------------------------------------- #
def layer_b() -> None:
    print("\n=== B. ladder + bracket + binary search (engine refuses) ===")
    # declared window 65536, but it refuses anything over 12000 tokens: the
    # benchmark must discover 12K, not trust the advertised number
    proc = start_mock(MOCK_PORT_REFUSE, "--context", "65536",
                      "--max-context", "12000")
    try:
        doc = run_context(f"http://127.0.0.1:{MOCK_PORT_REFUSE}", {
            "context_ladder": "1K,2K,4K,8K,16K,32K",
            "detect_max_context": True,
            "repeats": 1, "warmup": 0, "max_tokens": 8,
        })
    finally:
        proc.terminate()
    if not doc:
        return

    inner, res, status = unwrap(doc)
    check(status == "completed", "run completed (engine stayed alive)", str(status))

    entries = tested_entries(inner)
    by_target = {e["target_tokens"]: e for e in entries}
    print(f"       tested={sorted(by_target)} passed="
          f"{sorted(t for t, e in by_target.items() if e['ok'])}")

    # exact input construction
    exact = [e for e in entries if e["ok"]
             and (e.get("built") or {}).get("source") == "tokenize"]
    check(len(exact) == len([e for e in entries if e["ok"]]),
          "every successful step built its input with /tokenize",
          f"{len(exact)}/{len([e for e in entries if e['ok']])}")
    worst = max((abs((e["built"]["actual_tokens"] - e["target_tokens"])
                     / e["target_tokens"]) for e in exact), default=1.0)
    check(worst <= 0.02, "inputs hit the target token count within 2%",
          f"worst deviation {worst * 100:.2f}%")

    # the ladder must stop at the first refusal
    refusals = [e for e in entries if not e["ok"]]
    check(bool(refusals) and refusals[0]["target_tokens"] == 16384,
          "ladder refuses exactly at 16K",
          f"first failure target={refusals[0]['target_tokens'] if refusals else None}")
    check(all(e.get("failure_kind") == "too_long" for e in refusals),
          "refusal is classified as too_long, not unreachable",
          str([e.get("failure_kind") for e in refusals]))
    check(8192 in by_target and by_target[8192]["ok"],
          "8K passed (below the 12K real limit)")

    skipped = [c for c in (inner.get("contexts") or [])
               if c.get("skipped_reason")]
    check([c["target_tokens"] for c in skipped] == [32768],
          "un-attempted 32K step is reported as skipped, not dropped",
          str([c["target_tokens"] for c in skipped]))

    check(res.get("largest_passing") == 8192, "largest_passing = 8192",
          str(res.get("largest_passing")))
    check(res.get("smallest_failing") == 16384, "smallest_failing = 16384",
          str(res.get("smallest_failing")))

    mvc = res.get("max_verified_context") or {}
    print(f"       MVC value={mvc.get('value')} upper={mvc.get('upper_bound')} "
          f"steps={mvc.get('step_count')} verified={mvc.get('verified')}")
    check(mvc.get("verified") is True, "binary search converged (verified=True)")
    check(8192 <= (mvc.get("value") or 0) <= 12000,
          "verified ceiling lands inside the real 8K..12K band",
          str(mvc.get("value")))
    check(all(s["target_tokens"] <= 16384 for s in mvc.get("steps") or []),
          "binary search never probed above the known-bad bracket",
          str([s["target_tokens"] for s in mvc.get("steps") or []]))
    check(bool(mvc.get("bracket_from_ladder")),
          "detection reused the ladder bracket instead of re-probing it all")

    warnings = inner.get("data_warnings") or []
    check(any("容量边界" in w for w in warnings),
          "the refusal is explained as a capacity boundary", str(warnings[:2]))


# ---------------------------------------------------------------------- #
# C. a mock that dies under an oversized input
# ---------------------------------------------------------------------- #
def layer_c() -> None:
    print("\n=== C. engine dies mid-run -> abort, not a fake ceiling ===")
    proc = start_mock(MOCK_PORT_CRASH, "--context", "65536",
                      "--die-above", "12000")
    try:
        doc = run_context(f"http://127.0.0.1:{MOCK_PORT_CRASH}", {
            "context_ladder": "1K,2K,16K,32K",
            "detect_max_context": True,
            "repeats": 1, "warmup": 0, "max_tokens": 8, "retries": 0,
        })
    finally:
        proc.terminate()
    if not doc:
        return

    inner, res, status = unwrap(doc)
    check(status == "partial",
          "run is reported partial (data is real but incomplete)", str(status))
    check(res.get("aborted") is True, "the result records that it aborted")
    check("失联" in str(res.get("abort_reason")),
          "abort reason explains the engine went away",
          str(res.get("abort_reason"))[:110])

    entries = tested_entries(inner)
    by_target = {e["target_tokens"]: e for e in entries}
    check(by_target.get(1024, {}).get("ok") is True, "1K measured before the crash")
    check(by_target.get(2048, {}).get("ok") is True, "2K measured before the crash")

    skipped = [c for c in (inner.get("contexts") or []) if c.get("skipped_reason")]
    check([c["target_tokens"] for c in skipped] == [32768],
          "the 32K step is reported as skipped after the crash",
          str([(c["target_tokens"], c["skipped_reason"]) for c in skipped]))

    mvc = res.get("max_verified_context") or {}
    check(mvc.get("aborted") is True,
          "Maximum Verified Context is marked aborted, not presented as a result")
    check(mvc.get("verified") is False, "aborted detection is not 'verified'")
    check(mvc.get("value") == 2048,
          "value is the last length actually verified, not a guess",
          str(mvc.get("value")))
    check(mvc.get("upper_bound") == 16384,
          "upper_bound is the first length observed to fail",
          str(mvc.get("upper_bound")))
    check("重跑" in str(mvc.get("disclaimer")),
          "the disclaimer tells the user to re-run rather than trust the bracket")


# ---------------------------------------------------------------------- #
# D. an engine with no tokenizer (mirrors the halogen fork)
# ---------------------------------------------------------------------- #
def layer_notoken() -> None:
    print("\n=== E. no /tokenize: calibrated input length ===")
    port = 9203
    proc = start_mock(port, "--context", "65536", "--no-tokenize")
    try:
        doc = run_context(f"http://127.0.0.1:{port}", {
            "context_ladder": "2K,16K",
            "detect_max_context": False,
            "repeats": 1, "warmup": 0, "max_tokens": 8,
        })
    finally:
        proc.terminate()
    if not doc:
        return

    inner, res, status = unwrap(doc)
    entries = tested_entries(inner)
    by_target = {e["target_tokens"]: e for e in entries}
    check(status in ("completed", "partial"),
          "the run completed without a tokenizer", str(status))

    srcs = {(e.get("built") or {}).get("source") for e in entries}
    check("tokenize" not in srcs,
          "no step claims an exact length it could not measure", str(srcs))
    check(srcs <= {"calibrated", "estimated"},
          "the source is a declared approximation", str(srcs))

    cal = res.get("calibration") or {}
    check(len(cal.get("observations") or []) >= 2,
          "two calibration probes were taken",
          json.dumps(cal.get("observations"), ensure_ascii=False))
    check(bool(cal.get("chars_per_token")),
          "a ratio was fitted", str(cal.get("chars_per_token")))

    # the mock tokenizes at 3 chars/token; a calibrated build should land
    # close to the target instead of the ~10-19% error a hard-coded 4 gives
    devs = []
    for target, e in by_target.items():
        got = e.get("input_tokens")
        if got:
            devs.append(abs(got - target) / target)
    check(bool(devs), "steps reported their real input length",
          str([by_target[t].get("input_tokens") for t in by_target]))
    check(max(devs, default=1.0) <= 0.08,
          "calibrated inputs land within 8% of the target",
          f"worst {max(devs, default=1.0) * 100:.1f}%")

    notes = " ".join((e.get("built") or {}).get("notes", [""])[0]
                     for e in entries).lower()
    check("标定" in notes or "calibrat" in notes,
          "the report explains that the ratio was measured, not assumed")


# ---------------------------------------------------------------------- #
# D. real engine (optional)
# ---------------------------------------------------------------------- #
def layer_d(base_url: str) -> None:
    print(f"\n=== D. real engine {base_url} ===")
    doc = run_context(base_url, {
        "context_ladder": "512,2K,8K",
        "detect_max_context": False,
        "repeats": 1, "warmup": 0, "max_tokens": 16,
    })
    if not doc:
        return
    inner, res, status = unwrap(doc)
    entries = tested_entries(inner)
    for e in entries:
        b = e.get("built") or {}
        print(f"       {e['target_tokens']:>7} -> actual={b.get('actual_tokens')} "
              f"src={b.get('source')} ok={e['ok']} ttft={e.get('ttft_ms')} "
              f"prefill={e.get('prefill_tps')} in={e.get('input_tokens')}")
    check(bool(entries), "real engine produced context points")
    check(all(e["ok"] for e in entries),
          "all three ladder steps succeeded", f"status={status}")
    check(any((e.get("built") or {}).get("source") == "tokenize" for e in entries),
          "real engine's /tokenize was used for exact input length")
    ttf = [e.get("ttft_ms") for e in entries if e.get("ok")]
    if len(ttf) >= 2:
        check(ttf[-1] > ttf[0],
              "TTFT grows with input length (no prefix-cache artefact)",
              f"{ttf[0]:.0f}ms -> {ttf[-1]:.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None,
                    help="also check a real engine, e.g. http://127.0.0.1:8080")
    args = ap.parse_args()

    layer_a()
    layer_b()
    layer_c()
    layer_notoken()
    if args.real:
        layer_d(args.real)

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED — {len(failures)} check(s): " + "; ".join(failures))
        sys.exit(1)
    print("ALL PHASE 5 CHECKS PASSED")


if __name__ == "__main__":
    main()
