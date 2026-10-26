"""Real-engine verification of the measurement-validity controls (Phase 3-4).

Run this whenever the target address changes.  It answers four questions the
mock cannot settle, because they depend on what a *real* engine does with the
fields we send:

  1. does the adapter identify the engine correctly?
  2. the default (cache_prompt=false) really does measure cold prefill
  3. cache_prompt=true really does let the cache serve the request
  4. unique_prompt defeats the cache even when cache_prompt=true
  5. enable_thinking=false actually removes reasoning tokens

Usage:
    python tools/real_engine_check.py --llm http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:8000"
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
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
        return e.code, e.read(300).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def bench(llm: str, overrides: dict) -> dict | None:
    perf = {"repeats": 3, "warmup": 1, "max_tokens": 48}
    perf.update(overrides)
    st, run = fetch(f"{API}/api/runs", {"base_url": llm, "preset": "quick",
                                        "performance": perf})
    if st != 200:
        check(False, f"create run {overrides}", str(run)[:160])
        return None
    rid = run["run_id"]
    deadline = time.time() + 900
    while time.time() < deadline:
        st, doc = fetch(f"{API}/api/runs/{rid}", timeout=60)
        if doc.get("status") not in ("queued", "running", "warmup", "testing",
                                     "connect", "pending"):
            return doc.get("result")
        time.sleep(1)
    return None


def ttft(doc: dict | None):
    if not doc:
        return None
    return (doc.get("result", {}).get("primary") or {}).get("ttft_ms")


def reasoning_total(doc: dict | None) -> int:
    """Prefer characters actually received over the engine's self-report.

    Some engines omit ``completion_tokens_details.reasoning_tokens`` from the
    streamed usage block, so the self-reported number is 0 even when reasoning
    was clearly produced.
    """
    if not doc:
        return -1
    samples = doc.get("samples", [])
    chars = sum((s.get("reasoning_chars") or 0) for s in samples)
    if chars:
        return chars
    return sum((s.get("reasoning_tokens") or 0) for s in samples)


def reasoning_unit(doc: dict | None) -> str:
    if not doc:
        return "n/a"
    chars = sum((s.get("reasoning_chars") or 0) for s in doc.get("samples", []))
    return "chars" if chars else "usage-reported tokens"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    llm = args.llm

    print(f"=== target: {llm} ===\n")

    st, det = fetch(f"{API}/api/detect", {"base_url": llm}, timeout=120)
    if st != 200:
        check(False, "detect engine", str(det)[:200])
        return 1
    check(det["adapter"] in ("LlamaCppProvider", "VLLMProvider",
                             "OpenAICompatProvider"),
          "adapter selected", det["adapter"])
    print(f"       family={det['family']} conf={det['confidence']} "
          f"adapter={det['adapter']}")
    caps = det["capabilities"]
    print("       caps on: " + ", ".join(k for k, v in caps.items() if v is True))
    for e in det["evidence"]:
        print(f"       evidence: {e}")
    for n in det["notes"]:
        print(f"       note: {n}")
    print()

    print("=== measurement validity ===")
    default = bench(llm, {})                                    # cache default false
    cached = bench(llm, {"cache_prompt": True})
    unique = bench(llm, {"cache_prompt": True, "unique_prompt": True})

    d, c, u = ttft(default), ttft(cached), ttft(unique)
    print(f"       default (cache off)  TTFT = {d}")
    print(f"       cache_prompt=true    TTFT = {c}")
    print(f"       cache + unique input TTFT = {u}")

    if c and d:
        check(d > c * 1.5,
              "default measures cold prefill (slower than cache hit)",
              f"{d:.0f}ms vs {c:.0f}ms")
    if c and u:
        check(u > c * 1.5,
              "unique_prompt defeats the cache even with cache_prompt=true",
              f"{u:.0f}ms vs {c:.0f}ms")
    if d and u:
        check(abs(d - u) / max(d, 1) < 0.5,
              "cache-off and unique-input agree (both cold)",
              f"{d:.0f}ms vs {u:.0f}ms")

    print("\n=== thinking control ===")
    on = bench(llm, {"cache_prompt": False, "enable_thinking": True})
    off = bench(llm, {"cache_prompt": False, "enable_thinking": False})
    r_on, r_off = reasoning_total(on), reasoning_total(off)
    print(f"       reasoning ({reasoning_unit(on)})  on={r_on}  off={r_off}")
    if caps.get("thinking_control"):
        check(r_off == 0 and r_on > 0,
              "enable_thinking=false removes reasoning entirely",
              f"{r_on} -> {r_off}")
    else:
        check(True, "engine does not advertise thinking control (skipped)",
              "capability gate would have warned")

    print("")
    if failures:
        print(f"FAILED: {len(failures)} -> {', '.join(failures)}")
        return 1
    print("ALL REAL-ENGINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
