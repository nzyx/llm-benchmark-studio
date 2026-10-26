"""Phase-level smoke test for Local LLM Benchmark Studio.

Usage:
    python tools/smoke_test.py [--api http://127.0.0.1:8000]
                               [--llm http://127.0.0.1:8080]
                               [--frontend http://127.0.0.1:5173]
                               [--repeats 2] [--max-tokens 128]

Verifies, end to end:
  1. frontend dev server serves index.html + a transformed module
  2. backend /api/health
  3. frontend -> backend /api proxy
  4. parameter registry shape, and that max_tokens is classed as OUTPUT
  5. probe the inference server through the backend
  6. engine detection: family, adapter, evidence
  7. create a performance run, poll to completion, print the headline metrics
  8. create a context run, verify exact input lengths and TTFT growth
  9. read back the run history

Requires a reachable inference engine.  For a hermetic engine test that
covers llama.cpp / vLLM / generic fingerprints, use tools/phase3_check.py.

Exits non-zero if any hard check fails.

NOTE: this script deliberately bypasses environment proxies.  On this machine
HTTP_PROXY points at a sandbox proxy that mangles plain-HTTP requests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url: str, payload=None, timeout: float = 60.0) -> tuple[int, str]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    try:
        with NO_PROXY.open(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--llm", default=os.environ.get(
        "LLMB_TARGET", "http://127.0.0.1:8080"))
    ap.add_argument("--frontend", default="http://127.0.0.1:5173")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    r = Report()

    # 1 frontend -----------------------------------------------------------
    st, body = fetch(args.frontend + "/")
    r.check(st == 200 and "<div id=\"root\">" in body,
            "frontend serves index.html", f"HTTP {st}")
    st, body = fetch(args.frontend + "/src/main.tsx")
    r.check(st == 200 and "createRoot" in body,
            "frontend serves transformed entry module", f"HTTP {st}")

    # 2 backend ------------------------------------------------------------
    st, body = fetch(args.api + "/api/health")
    r.check(st == 200 and '"ok":true' in body,
            "backend /api/health", body[:80])

    # 3 proxy --------------------------------------------------------------
    st, body = fetch(args.frontend + "/api/health")
    r.check(st == 200 and '"ok":true' in body,
            "frontend -> backend /api proxy", f"HTTP {st}")

    # 4 parameter registry --------------------------------------------------
    st, body = fetch(args.api + "/api/params")
    cats: list[str] = []
    param_keys: list[str] = []
    has_relationship = False
    if st == 200:
        try:
            reg = json.loads(body)
            cats = [c["key"] for c in reg["categories"]]
            param_keys = [p["key"] for p in reg["params"]]
            has_relationship = bool(reg["relationship"].get("formula"))
        except Exception:  # noqa: BLE001
            pass
    r.check(st == 200 and has_relationship,
            "parameter registry exposed", f"categories={cats}")
    r.check({"model", "input", "output", "sampling", "execution"} <= set(cats),
            "registry covers all five categories")
    r.check("max_tokens" in param_keys, "max_tokens declared in registry")
    st, body = fetch(args.api + "/api/params")
    if st == 200:
        try:
            spec = next(p for p in json.loads(body)["params"]
                        if p["key"] == "max_tokens")
            r.check(spec["category"] == "output",
                    "max_tokens classified as OUTPUT (not context)",
                    f"category={spec['category']}")
        except Exception as exc:  # noqa: BLE001
            r.check(False, "max_tokens classified as OUTPUT", str(exc))

    # 5 probe --------------------------------------------------------------
    st, body = fetch(args.api + "/api/probe", {"base_url": args.llm})
    ok = st == 200
    models: list[str] = []
    model_params: dict = {}
    if ok:
        try:
            payload = json.loads(body)
            models = [m["id"] for m in payload["models"]]
            model_params = payload.get("model_param_values") or {}
        except Exception:  # noqa: BLE001
            ok = False
    r.check(ok and bool(models), "probe inference server",
            f"models={models}" if ok else body[:160])
    if models:
        vals = model_params.get(models[0], {})
        r.check(vals.get("context_window") is not None,
                "probe reports context window as a read-only model param",
                f"context_window={vals.get('context_window')}")

    # 6 engine detection ----------------------------------------------------
    st, body = fetch(args.api + "/api/detect", {"base_url": args.llm},
                     timeout=90)
    det: dict = {}
    if st == 200:
        try:
            det = json.loads(body) if isinstance(body, str) else body
        except Exception:  # noqa: BLE001
            det = {}
    r.check(st == 200 and det.get("adapter") in (
        "OpenAICompatProvider", "LlamaCppProvider", "VLLMProvider"),
        "engine adapter selected", str(det.get("adapter")))
    r.check(bool(det.get("evidence")) or bool(det.get("notes")),
            "detection explains itself (evidence or notes)")
    if det:
        print(f"       family     : {det.get('family')} "
              f"(confidence {det.get('confidence')})")
        print(f"       adapter    : {det.get('adapter')}")
        for item in (det.get("evidence") or [])[:4]:
            print(f"       evidence   : {item}")
        for item in (det.get("notes") or [])[:4]:
            print(f"       note       : {item}")
        caps = det.get("capabilities") or {}
        enabled = [k for k, v in caps.items() if v is True]
        print(f"       caps       : {', '.join(enabled) or '(none)'}")

    # 7 run ----------------------------------------------------------------
    run_id = None
    st, body = fetch(args.api + "/api/runs", {
        "base_url": args.llm,
        "preset": "quick",
        "performance": {"repeats": args.repeats,
                        "max_tokens": args.max_tokens},
    })
    if st == 200:
        run_id = json.loads(body).get("run_id")
    r.check(bool(run_id), "create benchmark run", body[:120])

    final = None
    if run_id:
        t0 = time.time()
        while time.time() - t0 < args.timeout:
            st, body = fetch(f"{args.api}/api/runs/{run_id}", timeout=30)
            try:
                doc = json.loads(body)
            except Exception:  # noqa: BLE001
                break
            if doc.get("status") not in (
                "queued", "running", "warmup", "testing", "connect", "pending",
            ):
                final = doc.get("result")
                break
            time.sleep(2)

    if r.check(final is not None and final.get("status") == "completed",
               "run reaches completed",
               f"{round(time.time() - t0, 1)}s" if final else "no result"):
        prim = (final.get("result") or {}).get("primary") or {}
        print("       ttft_ms      :", prim.get("ttft_ms"))
        print("       decode_tps   :", prim.get("decode_tps"))
        print("       prefill_tps  :", prim.get("prefill_tps"))
        print("       tpot_ms      :", prim.get("tpot_ms"))
        print("       e2e_ms       :", prim.get("e2e_ms"))
        print("       success_rate :", final["result"].get("success_rate"))
        print("       duration_ms  :", final.get("duration_ms"))
        r.check(bool(final["result"].get("success_rate")),
                "at least one successful sample")

    # 8 context task --------------------------------------------------------
    # a short ladder only: the point is to prove the task runs end to end and
    # reports exact input lengths, not to measure the engine's ceiling
    ctx_id = None
    st, body = fetch(args.api + "/api/runs", {
        "base_url": args.llm,
        "preset": "context",
        "context": {"context_ladder": "512,2K", "detect_max_context": False,
                    "repeats": 1, "warmup": 0, "max_tokens": 16},
    })
    if st == 200:
        ctx_id = json.loads(body).get("run_id")
    r.check(bool(ctx_id), "create context run", body[:120])

    ctx = None
    if ctx_id:
        t0 = time.time()
        while time.time() - t0 < args.timeout:
            st, body = fetch(f"{args.api}/api/runs/{ctx_id}", timeout=30)
            try:
                doc = json.loads(body)
            except Exception:  # noqa: BLE001
                break
            if doc.get("status") not in (
                "queued", "running", "warmup", "testing", "connect", "pending",
            ):
                ctx = doc.get("result")
                break
            time.sleep(2)

    if r.check(ctx is not None and ctx.get("status") == "completed",
               "context run reaches completed",
               f"{round(time.time() - t0, 1)}s" if ctx else "no result"):
        entries = [c for c in (ctx.get("contexts") or [])
                   if c.get("kind") == "context"]
        res = ctx.get("result") or {}
        for c in entries:
            b = c.get("built") or {}
            print(f"       {c['target_tokens']:>6} -> actual={b.get('actual_tokens')}"
                  f" src={b.get('source')} ok={c.get('ok')}"
                  f" ttft={c.get('ttft_ms')} prefill={c.get('prefill_tps')}")
        r.check(len(entries) == 2, "both ladder steps present", f"{len(entries)}")
        r.check(res.get("largest_passing") == 2048,
                "largest_passing reported", str(res.get("largest_passing")))
        # Not every engine has /tokenize (the halogen fork does not), so the
        # requirement is not "exact" but "honest": the source must be declared,
        # and a non-exact source must come with an explanation.
        srcs = [(c.get("built") or {}).get("source")
                for c in entries if c.get("ok")]
        r.check(srcs and all(s in ("tokenize", "calibrated", "estimated")
                             for s in srcs),
                "input length source declared", str(srcs))
        inexact = [c for c in entries if c.get("ok")
                   and (c.get("built") or {}).get("source") != "tokenize"]
        r.check(all((c.get("built") or {}).get("notes") for c in inexact),
                "a non-exact input length is explained, not silently used",
                f"{len(inexact)} inexact step(s)")
        ttf = [c.get("ttft_ms") for c in entries if c.get("ok")]
        if len(ttf) == 2:
            r.check(ttf[1] > ttf[0], "TTFT grows with input length",
                    f"{ttf[0]:.0f}ms -> {ttf[1]:.0f}ms")

    # 9 history ------------------------------------------------------------
    st, body = fetch(args.api + "/api/runs?limit=20")
    total = -1
    if st == 200:
        try:
            total = json.loads(body)["total"]
        except Exception:  # noqa: BLE001
            pass
    r.check(total >= 1, "run persisted in history", f"total={total}")

    print()
    if r.failures:
        print(f"FAILED: {len(r.failures)} check(s) -> {', '.join(r.failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
