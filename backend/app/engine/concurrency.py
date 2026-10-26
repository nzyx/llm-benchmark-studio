"""ConcurrencyTask - throughput and latency as a function of parallel load.

The measurement itself is the same streaming request the other tasks use
(:mod:`app.engine.probe`), so a "TTFT" here means exactly what it means in the
performance and context results.  What is new is the *shape*: N requests are
issued **at the same time** per level, and the report describes the level as a
whole rather than an individual request.

Two things make this honest:

1. **Throughput is measured from the wall clock of the level**, not summed
   from per-request timings.  Summed timings would happily report "throughput"
   for requests that actually ran one after another.

2. **Every request gets a unique input.**  At concurrency 1 a repeated prompt
   is cache-friendly and flattering; at concurrency 32 the picture is murkier
   still, because some of the parallel requests could share a cache hit.  A
   unique marker at the front of each prompt removes the question entirely -
   the same lesson as the prefix-cache bug in the performance task.

3. **The engine's own gauges are sampled *during* each level, not only at its
   boundaries.**  Client-side Little's law cannot tell "32 requests in flight"
   from "16 running and 16 queued in the engine" - both look like 32 from here.
   Measured on a live vLLM node at nominal 32: `num_requests_running` max 16,
   `num_requests_waiting` max 16, i.e. half of the reported parallelism was
   queueing, while a before/after snapshot (taken with the engine idle) showed
   a reassuring zero.  Sampling a gauge before and after a load test measures
   the idle engine, not the load.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..config import HARD_MAX_CONCURRENCY
from .base import BenchmarkTask, ProgressEvent
from .probe import (
    cache_warning,
    decode_coalesced_warning,
    input_identity_warning,
    input_length_warning,
    input_marker,
    measure_once,
    prefill_contamination_warning,
    summarize_samples,
    worst_kind,
)

#: How often to poll the engine's own gauges while a level runs.  ``/metrics``
#: is a different endpoint from the generation path and the payload is a few KB,
#: so this does not meaningfully perturb the level - whereas a gauge read only
#: at the level's boundaries measures an idle engine and cannot see queueing.
GAUGE_POLL_S = 0.2

DEFAULT_PROMPT = (
    "Explain, in about 120 words, how continuous batching keeps a GPU busy "
    "while requests arrive at different times. Be concrete."
)


class ConcurrencyTask(BenchmarkTask):
    task_type = "concurrency"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._levels: list[dict[str, Any]] = []
        self._abort_reason: str | None = None
        self._notes: list[str] = []
        #: engine-side gauges seen during the level currently being measured
        self._gauge_running: list[float] = []
        self._gauge_waiting: list[float] = []

    # ------------------------------------------------------------------ #
    async def prepare(self) -> None:
        try:
            self.doc["metrics_snapshot"]["before"] = await self.provider.get_metrics()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        cfg = self.config
        ladder: list[int] = list(cfg.get("concurrency_ladder") or [1])
        # defence in depth: the registry already clamps, but a task must never
        # be the thing that trusts its own config with a machine's safety
        over = [c for c in ladder if c > HARD_MAX_CONCURRENCY]
        if over:
            ladder = [c for c in ladder if c <= HARD_MAX_CONCURRENCY]
            self._notes.append(
                f"并发档位 {over} 超过硬上限 {HARD_MAX_CONCURRENCY}，已跳过"
            )
        if not ladder:
            ladder = [1]
        per_level = max(1, int(cfg.get("requests_per_level", 12)))
        # "waves" sizes each level to at least two waves of its own concurrency
        # (the sampled default); "flat" means the caller pinned the count and
        # we fire exactly that, however unmeasurable that makes a level.
        mode = cfg.get("requests_per_level_mode") or "flat"
        warmup = max(0, int(cfg.get("warmup", 0)))

        self._levels = []
        total = len(ladder)

        for i, level in enumerate(ladder):
            if self._abort_reason:
                break
            await self.emit(ProgressEvent(
                status="testing", phase="concurrency", done=i, total=total,
                message=f"并发 {level}（{i + 1}/{total}）",
                extra={"concurrency": level},
            ))
            entry = await self._measure_level(
                level,
                max(per_level, 2 * level) if mode == "waves" else per_level,
                warmup)
            self._levels.append(entry)
            self.doc["_done"], self.doc["_total"] = i + 1, total
            await self.emit(ProgressEvent(
                status="testing", phase="concurrency", done=i + 1, total=total,
                message=(f"并发 {level} → "
                         f"{entry['request_throughput'] or 0:.2f} req/s，"
                         f"TTFT p95 {entry['ttft_ms']['p95'] or 0:.0f}ms"),
                extra={
                    "concurrency": level,
                    "last": {
                        "ok": entry["success_rate"] > 0,
                        "ttft_ms": entry["ttft_ms"]["mean"],
                        "request_throughput": entry["request_throughput"],
                    },
                },
            ))

        if self._abort_reason:
            self._notes.append(self._abort_reason)

    # ------------------------------------------------------------------ #
    async def _measure_level(self, level: int, count: int,
                             warmup: int) -> dict[str, Any]:
        """Fire ``count`` requests at once and describe the batch.

        The semaphore is what makes this a *concurrency* test: without it the
        tasks would still all be created at once, but the measurement would be
        of our own queueing rather than the engine's.
        """
        gate = asyncio.Semaphore(level)

        async def one(idx: int) -> dict[str, Any]:
            async with gate:
                return await self._one_request(idx)

        for _ in range(warmup):
            await one(-1)

        # Sample the engine's own gauges while the level is in flight.  Doing
        # this only before/after the level would measure an idle engine.
        stop = asyncio.Event()
        sampler = asyncio.create_task(self._sample_engine_gauges(stop))

        t0 = time.perf_counter()
        try:
            samples = await asyncio.gather(*(one(i) for i in range(count)))
            wall_s = max(time.perf_counter() - t0, 1e-6)
        finally:
            stop.set()
            # never let the sampler outlive the level (or be left pending)
            try:
                await asyncio.wait_for(sampler, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                sampler.cancel()

        engine_running = self._gauge_running
        engine_waiting = self._gauge_waiting
        eng_running_max = max(engine_running) if engine_running else None
        eng_running_mean = (round(sum(engine_running) / len(engine_running), 2)
                            if engine_running else None)
        eng_waiting_max = max(engine_waiting) if engine_waiting else None

        ok = [s for s in samples if s.get("ok")]
        failed = [s for s in samples if not s.get("ok")]
        summary = summarize_samples(list(samples))
        stats = summary["stats"]

        out_tokens = sum(s.get("output_tokens") or 0 for s in ok)
        in_tokens = sum(s.get("input_tokens") or 0 for s in ok)
        kind = worst_kind(list(samples))
        if failed:
            await self._handle_failure(kind, level, failed[0].get("error"))

        ttft = _percentiles(stats["ttft_ms"])
        e2e = _percentiles(stats["e2e_ms"])

        # Little's law: the average number of requests actually in flight.
        # If a "concurrency 8" level reports ~1 here, the harness serialised
        # the requests and the level is not measuring concurrency at all -
        # the single most useful number for trusting this whole table.
        busy_s = sum((s.get("e2e_ms") or 0) for s in ok) / 1000.0
        effective = round(busy_s / wall_s, 2) if wall_s > 0 else None

        # `effective` counts a request as in flight for its whole lifetime,
        # which *includes* time spent queued inside the engine - correct, but
        # easy to misread as achieved parallelism.  The engine's own running
        # count is the parallelism; the difference is queueing.  Measured live:
        # nominal 32 -> eff 28.58 while the engine ran at most 16.
        engine_parallel = eng_running_mean
        queue_share = None
        if effective and engine_parallel:
            queue_share = round(max(0.0, 1.0 - engine_parallel / effective), 3)
        engine_queued = bool(eng_waiting_max and eng_waiting_max > 0)

        # How much of a request's life was spent *before* its first token.
        # Bounded by construction (over the requests that succeeded), unlike
        # `Σ ttft / wall`, which sums N overlapping waits and so grows with
        # concurrency - measured at 3688% on a 64-way level, i.e. nonsense.
        # It is a wait share, not a prefill share: at high concurrency most of
        # it is engine queueing, which the gauge fields above identify directly.
        ttft_share = None
        busy_ms = sum((s.get("e2e_ms") or 0) for s in ok)
        if busy_ms > 0:
            ttft_share = round(
                sum((s.get("ttft_ms") or 0) for s in ok) / busy_ms, 3)

        return {
            "concurrency": level,
            "requests": count,
            # The most requests that could ever be in flight at this level.
            # A level of 32 fired with only 12 requests can never exceed 12, so
            # its throughput is not a concurrency-32 reading - and without this
            # figure the table looks like the engine topped out.
            "achievable_concurrency": min(level, count),
            # Two ways a level fails to demonstrate its own concurrency, and
            # both must be visible: (a) arithmetic - it fired fewer requests
            # than the level claims, so N in flight is impossible; (b) measured
            # - it fired enough requests but the wave shape still pulled the
            # mean in-flight count well below N (a single wave of N reads
            # ~0.7*N; two waves read ~0.94*N).  A third way lives in the
            # engine-side fields below: the engine simply would not run that
            # many at once and queued the rest.
            "underprovisioned": bool(
                count < level
                or (effective is not None and effective < 0.8 * level)),
            # wall-clock based: this is the only honest source of throughput
            "wall_s": round(wall_s, 3),
            "request_throughput": round(len(ok) / wall_s, 3),
            "token_throughput": round(out_tokens / wall_s, 3),
            "input_throughput": round(in_tokens / wall_s, 3),
            "effective_concurrency": effective,
            # ---- the engine's own view, sampled while the level ran --------
            #: how many requests the ENGINE says were running at once
            "engine_running_max": eng_running_max,
            "engine_running_mean": eng_running_mean,
            #: how many it queued internally.  Non-zero means this level was
            #: partly a queueing measurement, whatever the client's eff says.
            "engine_waiting_max": eng_waiting_max,
            "engine_queued": engine_queued,
            #: the share of `effective_concurrency` that was engine queueing
            "queue_share": queue_share,
            #: share of a request's lifetime spent before its first token.
            #: `token_throughput` is end-to-end, so this wait sits in its
            #: denominator: a level with a big share here understates decode.
            "ttft_share": ttft_share,
            "success_rate": round(len(ok) / count, 4) if count else 0.0,
            "error_rate": round(len(failed) / count, 4) if count else 0.0,
            "errors": [s.get("error") for s in failed][:5],
            "failure_kind": kind,
            "ttft_ms": ttft,
            "tpot_ms": stats["tpot_ms"]["mean"],
            "e2e_ms": e2e,
            "decode_tps": stats["decode_tps"]["mean"],
            "output_tokens_total": out_tokens,
            "samples": list(samples),
            "stats": stats,
        }

    async def _sample_engine_gauges(self, stop: asyncio.Event) -> None:
        """Record the engine's own running/queued counts while a level runs.

        A client cannot distinguish "the engine ran 32 at once" from "it ran 16
        and queued 16": both look like 32 requests in flight from here, and
        Little's law returns the same number for either.  The engine's gauges
        are the only independent witness - and they have to be read *during* the
        load, because a snapshot taken before or after sees an idle engine.

        Polling stops with the level; failures are ignored rather than fatal, so
        an engine without ``/metrics`` simply produces no engine-side fields.
        """
        self._gauge_running = []
        self._gauge_waiting = []
        while not stop.is_set():
            try:
                metrics = await self.provider.get_metrics()
                norm = (metrics or {}).get("normalized") or {}
                running = norm.get("requests_running")
                waiting = norm.get("requests_waiting")
                if running is not None:
                    self._gauge_running.append(float(running))
                if waiting is not None:
                    self._gauge_waiting.append(float(waiting))
            except Exception:  # noqa: BLE001 - gauges are optional evidence
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=GAUGE_POLL_S)
            except asyncio.TimeoutError:
                continue

    async def _one_request(self, idx: int) -> dict[str, Any]:
        cfg = self.config
        prompt_text = cfg.get("prompt") or DEFAULT_PROMPT
        if cfg.get("unique_prompt", True):
            # shared with the other two tasks: a leading tag, but in a form
            # written for concurrent traffic (which sample, which request)
            prompt_text = input_marker(True) + prompt_text
        use_chat = bool(cfg.get("use_chat", True))
        sample = await measure_once(
            self.provider,
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt_text}] if use_chat else None,
            prompt=None if use_chat else prompt_text,
            max_tokens=cfg.get("max_tokens", 128),
            temperature=cfg.get("temperature", 0.0),
            options=cfg.get("options"),
            timeout_s=cfg.get("request_timeout_s"),
        )
        sample["index"] = idx
        sample["warmup"] = idx < 0
        return sample

    async def _handle_failure(self, kind: str | None, level: int,
                              detail: str | None) -> None:
        """A dying engine makes a concurrency sweep meaningless, not slow."""
        if not await self.engine_alive():
            if await self.wait_for_engine():
                return
            self._abort_reason = (
                f"引擎在并发 {level} 档失联"
                f"（{kind or 'unknown'}: {detail or 'no detail'}），"
                "后续档位已中止 —— 结果只覆盖失联之前的档位"
            )

    # ------------------------------------------------------------------ #
    async def collect(self) -> None:
        try:
            self.doc["metrics_snapshot"]["after"] = await self.provider.get_metrics()
        except Exception:  # noqa: BLE001
            pass

    async def calculate(self) -> None:
        levels = self._levels
        ok_levels = [l for l in levels if l["success_rate"] > 0]
        cfg = self.config
        slots = cfg.get("server_slots")

        self.doc["levels"] = levels
        evidence = _engine_evidence(self.doc.get("metrics_snapshot") or {}, levels)
        batching = _batching_profile(levels)
        self.doc["stats"] = {
            "request_throughput": _series(levels, "request_throughput"),
            "token_throughput": _series(levels, "token_throughput"),
            "effective_concurrency": _series(levels, "effective_concurrency"),
            #: the engine's own answer to the same question; None when it does
            #: not expose gauges, which is not evidence either way
            "engine_running": _series(levels, "engine_running_mean"),
            "ttft_p95": _series(levels, "ttft_p95"),
            "ttft_p99": _series(levels, "ttft_p99"),
            "e2e_p95": _series(levels, "e2e_p95"),
        }

        # Independent cross-check of the headline number: the engine counts the
        # tokens it produced, so our total must not be much smaller.  A large
        # shortfall means our own accounting is losing tokens - typically an
        # engine that reports no `usage` in a stream, where we fall back to
        # counting one token per event on an engine whose events carry several.
        counted_tokens = sum(l["output_tokens_total"] for l in ok_levels)
        engine_tokens = evidence.get("generation_tokens")
        token_coverage = (round(counted_tokens / engine_tokens, 3)
                          if engine_tokens and counted_tokens else None)

        all_samples = [s for l in levels for s in l["samples"]]
        cached = cache_warning(all_samples)
        coverage_warnings: list[str] = []
        if token_coverage is not None and token_coverage < 0.8:
            coverage_warnings.append(
                f"本次我们统计的输出 token 只有引擎自报的 "
                f"{token_coverage * 100:.0f}% —— 引擎产出的 token 数多于我们账面上"
                "的，最可能的原因是该引擎的流里没有 `usage`，我们退化成"
                "「一个流事件算一个 token」，而它的事件一次携带多个 token。"
                "**token 吞吐会被低估**，请以 request 吞吐和延迟为准"
            )
        # Same comparability rules as the other two tasks, applied per level.
        # A concurrency level is a *batch*, so reusing one prompt would let
        # parallel requests share a cache hit and inflate the throughput.
        comparability: list[str] = []
        wait_shares: list[tuple[int, float, bool]] = []
        for l in levels:
            scope = f"并发 {l['concurrency']} 档"
            checks = [input_identity_warning(l.get("samples") or [], scope),
                      input_length_warning(l.get("samples") or [], scope),
                      decode_coalesced_warning(l.get("samples") or [], scope)]
            # When we *know* the engine queued this level, that is the better
            # explanation than the generic "TTFT contained a wait" one - and we
            # only know it because the gauges were sampled under load.
            if not l.get("engine_queued"):
                checks.append(
                    prefill_contamination_warning(l.get("samples") or [], scope))
            for msg in checks:
                if msg and msg not in comparability:
                    comparability.append(msg)
            if l.get("ttft_share") is not None:
                wait_shares.append((l["concurrency"], l["ttft_share"],
                                    bool(l.get("engine_queued"))))

        # ---- is the quoted token throughput comparable across levels? -------
        # `token_throughput` is end-to-end tokens per wall-clock second, so
        # "time before the first token" sits in its denominator.  That wait
        # overlaps away as concurrency rises, which makes the curve climb for a
        # reason that has nothing to do with decode speed.  Measured on a real
        # node: 32% of a request's life at concurrency 1 was pre-first-token,
        # 73% at concurrency 64.
        #
        # At the high end the cause is usually *already measured*: the engine
        # gauge said it queued 48 of them.  Then this must point at that, not
        # offer a second, vaguer explanation for the same number.
        share_warnings: list[str] = []
        if len(wait_shares) >= 2:
            lo = min(wait_shares, key=lambda p: p[1])
            hi = max(wait_shares, key=lambda p: p[1])
            if hi[1] >= 0.35 and hi[1] - lo[1] >= 0.15:
                if hi[2]:
                    share_warnings.append(
                        f"并发 {hi[0]} 档有 {hi[1] * 100:.0f}% 的请求耗时花在"
                        f"「引擎开始算之前」，而并发 {lo[0]} 档只有 "
                        f"{lo[1] * 100:.0f}% —— 这正是上面「引擎自己在排队」的"
                        "量化结果，不是另一种现象。它也说明该档的 TTFT 主要由"
                        "排队构成，不能读作解码能力"
                    )
                else:
                    share_warnings.append(
                        f"各档「首 token 之前的等待」占请求总耗时的比例差别很大"
                        f"（并发 {lo[0]} 档 {lo[1] * 100:.0f}% → 并发 {hi[0]} 档 "
                        f"{hi[1] * 100:.0f}%）：`Token 吞吐` 是端到端算的，这段"
                        "等待也在它的分母里，而并发越高越被重叠掉 —— 所以曲线"
                        "前段的上升有一部分只是「等待被并发掩盖」，不代表解码"
                        "变快。比解码速度请用 TPOT / decode。"
                    )
        self.doc["data_warnings"] = (
            comparability
            + _concurrency_warnings(levels, slots, evidence, batching)
            + coverage_warnings
            + share_warnings
            + self._notes
            + ([cached] if cached else [])
        )

        # Two different questions, two different sets of levels.
        #
        #  * CAPACITY - "how many tok/s can this box produce?"  That is the
        #    highest aggregate throughput *observed at any offered load*,
        #    including levels where the engine queued.  Queueing inside the
        #    engine is not a measurement error: it is exactly how vLLM keeps
        #    its batch full, and it is how the box reaches its throughput.
        #    Measured live: excluding queued levels reported a "peak" of 234.7
        #    tok/s while levels 32/64 actually delivered 459/461 tok/s - the
        #    headline understated the machine by half.
        #  * PARALLELISM - "how many requests does it really run at once?"  A
        #    level can only support that claim if it fired enough requests AND
        #    the engine did not have to queue.  This is what the per-level
        #    `effective_concurrency` column must be read with.
        capacity_level = (max(ok_levels, key=lambda l: l["token_throughput"])
                          if ok_levels else None)
        # No fallback here.  A fallback would have to re-admit a level the
        # engine queued, and for *that* level we hold direct evidence it did
        # not parallelise - so it can never support a parallelism claim, even
        # when nothing better is available.  An empty list is the honest answer,
        # and the capacity figure above is unaffected because it draws on every
        # level.
        parallel = [l for l in ok_levels
                    if not l.get("underprovisioned")
                    and not l.get("engine_queued")]

        self.doc["result"] = {
            "ladder": [l["concurrency"] for l in levels],
            "tested": len(levels),
            "requests_per_level": int(cfg.get("requests_per_level", 0)),
            "requests_per_level_mode": cfg.get("requests_per_level_mode"),
            "per_level_requests": {str(l["concurrency"]): l["requests"]
                                   for l in levels},
            "total_requests": sum(l["requests"] for l in levels),
            "server_slots": slots,
            "queues_above_slots": bool(cfg.get("queues_above_slots")),
            #: throughput at the offered load that produced the most tokens/s.
            #: This is the box's measured capacity.
            "peak_request_throughput": max(
                (l["request_throughput"] for l in ok_levels), default=None),
            "peak_token_throughput": (
                capacity_level["token_throughput"] if capacity_level else None),
            "at_concurrency": (
                capacity_level["concurrency"] if capacity_level else None),
            #: True when the capacity figure came from a level the engine had
            #: to queue: the number is real, but reaching it needs more offered
            #: load than the engine can actually run in parallel.
            "engine_limited_capacity": bool(
                capacity_level and capacity_level.get("engine_queued")),
            #: levels that can support a "this is N-way parallel" claim
            "parallelism_levels": [l["concurrency"] for l in parallel],
            #: highest throughput among levels that demonstrated their own
            #: concurrency - i.e. scaling with parallelism only, no queueing
            "parallel_peak_token_throughput": max(
                (l["token_throughput"] for l in parallel), default=None),
            "scaling": _scaling(ok_levels),
            "batching": batching or None,
            "engine_evidence": evidence or None,
            #: the engine's own parallelism ceiling, when its gauges were
            #: sampled.  The client's `effective_concurrency` cannot express
            #: this: queueing inflates it.  `None` = the engine exposed nothing.
            "engine_parallelism_ceiling": (
                max((l.get("engine_running_max") or 0) for l in levels)
                if any(l.get("engine_running_max") is not None for l in levels)
                else None),
            "engine_queued_levels": [
                l["concurrency"] for l in levels if l.get("engine_queued")],
            #: our counted output tokens / the engine's own generation count.
            #: Well below 1 means our token accounting is losing tokens.
            "token_coverage": token_coverage,
            "aborted": bool(self._abort_reason),
            "abort_reason": self._abort_reason,
        }
        if self._abort_reason:
            self.doc["status"] = "partial"
            self._record_error(self._abort_reason)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _engine_evidence(snapshot: dict[str, Any],
                     levels: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """What did the engine itself report over the run?

    A falling throughput curve has several possible causes and they need very
    different responses: rejection needs a bigger server limit, queueing needs
    fewer concurrent users, and a batch-1 fast path needs nothing at all - it
    is just how the engine behaves.  Collecting the engine's own counters
    (preemptions, waiting, successes) is what lets the report pick the right
    one instead of guessing.

    The before/after snapshot alone is weak evidence about *load*: it is taken
    with the engine idle, so queueing that happened during a level is invisible
    in it.  Verified on a live node - the boundary snapshot read a reassuring
    ``requests_waiting: 0`` for a run whose in-load gauges peaked at 16 queued.
    ``levels`` carries those in-load readings, and they take precedence; the
    two sources are reported separately so a warning can be honest about which
    one it is quoting.
    """
    before = (snapshot.get("before") or {}).get("normalized") or {}
    after = (snapshot.get("after") or {}).get("normalized") or {}

    # in-load evidence first: it is the only kind that can see queueing
    in_load_running = [l.get("engine_running_max") for l in (levels or [])
                       if l.get("engine_running_max") is not None]
    in_load_waiting = [l.get("engine_waiting_max") for l in (levels or [])
                       if l.get("engine_waiting_max") is not None]
    peak_running = max(in_load_running) if in_load_running else None
    peak_waiting = max(in_load_waiting) if in_load_waiting else None

    if not before or not after:
        return ({"available": False, "measured_in_load": bool(in_load_waiting),
                 "peak_running": peak_running, "peak_waiting": peak_waiting}
                if in_load_waiting else {})

    def delta(key: str) -> float | None:
        a, b = before.get(key), after.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(b - a, 4)
        return None

    out = {
        "available": True,
        "preemptions": delta("preemptions_total"),
        "requests_success": delta("requests_success_total"),
        "generation_tokens": delta("generation_tokens_total"),
        "prompt_tokens": delta("prompt_tokens_total"),
        #: the engine's own view of mean TTFT over the sampled window - a
        #: cross-check on our client-side percentiles, not a per-level figure
        "ttft_seconds_mean": after.get("ttft_seconds_mean"),
        "e2e_seconds_mean": after.get("e2e_seconds_mean"),
        #: gauges sampled *while* the levels were running (None = not sampled)
        "measured_in_load": bool(in_load_waiting or in_load_running),
        "peak_running": peak_running,
        "peak_waiting": peak_waiting,
        #: what a before/after snapshot saw; kept for comparison, because a
        #: zero here next to a non-zero peak above is itself the finding
        "boundary_waiting": delta("requests_waiting"),
    }
    # An engine that was preempting or queueing *told* us so.  Its silence is
    # evidence too - but only the in-load reading is evidence about load.
    out["requests_waiting"] = peak_waiting or out["boundary_waiting"]
    out["signs_of_overload"] = [
        name for name, value in
        (("preemptions", out.get("preemptions")),
         ("requests_waiting", out.get("requests_waiting")))
        if value
    ]
    return out


def _batching_profile(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate "the engine gets slower when batched" from "the harness never
    actually overlapped anything" - the two look identical in a throughput
    column and have opposite remedies.

    Measured on a real vLLM node (`Qwen3.8-Flash-Next`): concurrency 1 decoded
    at ~190 tok/s, and *every* level from 2 upward settled at ~24-27 ms/token
    per request (~37-40 tok/s) regardless of batch size.  Total throughput went
    190 -> 72 at 2-way and only recovered to ~160 at 4-way, with zero errors,
    zero preemptions and zero waiting.  A batch-1 fast path is a real and
    common design (speculative decoding is often only worth it at batch 1), so
    the report has to name it rather than call it a measurement error.
    """
    # Only levels that could actually demonstrate their own concurrency, and
    # only when each carries enough samples to describe a curve.  Both gates
    # matter: a real run with a *pinned* 3 requests per level produced a
    # confident "this engine tops out at 2.9-way concurrency" and "concurrency
    # never pays off" - neither was a property of the engine, both were the
    # sample count.  Claiming a hardware ceiling from three samples is exactly
    # the kind of confident nonsense this whole report exists to avoid.
    ok = [l for l in levels
          if l.get("success_rate") and not l.get("underprovisioned")]
    if len(ok) < 2 or min((l.get("requests") or 0) for l in ok) < 8:
        return {}
    ordered = sorted(ok, key=lambda l: l["concurrency"])
    base = ordered[0]
    base_tpot = base.get("tpot_ms")
    batched = [l for l in ordered if l["concurrency"] > base["concurrency"]
               and l.get("tpot_ms")]
    if not base_tpot or not batched:
        return {}

    tpots = sorted(l["tpot_ms"] for l in batched)
    mid = tpots[len(tpots) // 2]
    tax = mid / base_tpot
    base_tp = base.get("token_throughput") or 0
    peak = max(ordered, key=lambda l: l.get("token_throughput") or 0)
    eff_peak = max((l.get("effective_concurrency") or 0) for l in ordered)
    eff_at = next((l["concurrency"] for l in ordered
                   if l.get("effective_concurrency") == eff_peak), None)
    out: dict[str, Any] = {
        "base_concurrency": base["concurrency"],
        "base_tpot_ms": round(base_tpot, 2),
        "base_decode_tps": round(1000 / base_tpot, 1) if base_tpot else None,
        "batched_tpot_ms": round(mid, 2),
        "batched_decode_tps": round(1000 / mid, 1) if mid else None,
        "batching_tax_x": round(tax, 2),
        "base_token_throughput": round(base_tp, 1),
        "peak_token_throughput": peak.get("token_throughput"),
        "peak_at_concurrency": peak.get("concurrency"),
        "throughput_gain_x": (round((peak.get("token_throughput") or 0) / base_tp, 2)
                              if base_tp else None),
        "effective_concurrency_ceiling": eff_peak,
        "ceiling_at": eff_at,
    }
    out["batch1_fast_path"] = bool(tax >= 2.0 and (out["throughput_gain_x"] or 0)
                                   <= 1.2)
    return out


def _percentiles(stat: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": stat.get("mean"),
        "p50": stat.get("p50"),
        "p95": stat.get("p95"),
        "p99": stat.get("p99"),
        "min": stat.get("min"),
        "max": stat.get("max"),
    }


#: summary keys that read through to a per-stat sub-dict of the level
_SUB_FIELDS = {
    "ttft_p50": ("ttft_ms", "p50"),
    "ttft_p95": ("ttft_ms", "p95"),
    "ttft_p99": ("ttft_ms", "p99"),
    "e2e_p50": ("e2e_ms", "p50"),
    "e2e_p95": ("e2e_ms", "p95"),
    "e2e_p99": ("e2e_ms", "p99"),
}


def _series(levels: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """One point per concurrency level."""
    points: list[dict[str, Any]] = []
    for l in levels:
        if key in _SUB_FIELDS:
            bucket, field = _SUB_FIELDS[key]
            value = (l.get(bucket) or {}).get(field)
        else:
            value = l.get(key)
        if value is not None:
            points.append({"x": l["concurrency"], "y": value})
    nums = [p["y"] for p in points]
    return {
        "count": len(points),
        "points": points,
        "min": round(min(nums), 4) if nums else None,
        "max": round(max(nums), 4) if nums else None,
        "mean": round(sum(nums) / len(nums), 4) if nums else None,
    }


def _scaling(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """How throughput and latency move as load rises."""
    if len(levels) < 2:
        return {}
    ordered = sorted(levels, key=lambda l: l["concurrency"])
    first, last = ordered[0], ordered[-1]
    out: dict[str, Any] = {
        "from_concurrency": first["concurrency"],
        "to_concurrency": last["concurrency"],
        "concurrency_ratio": round(
            last["concurrency"] / max(first["concurrency"], 1), 2),
        "throughput_first": first["request_throughput"],
        "throughput_last": last["request_throughput"],
    }
    if first["token_throughput"]:
        out["throughput_gain_x"] = round(
            last["request_throughput"] / first["request_throughput"], 2) \
            if first["request_throughput"] else None
    a = (first.get("ttft_ms") or {}).get("p95")
    b = (last.get("ttft_ms") or {}).get("p95")
    if a and b:
        out["ttft_p95_first"] = a
        out["ttft_p95_last"] = b
        out["ttft_p95_change_x"] = round(b / a, 3)
    # a rising token throughput that then flattens is the real ceiling
    if len(ordered) >= 3:
        gains = []
        for prev, cur in zip(ordered, ordered[1:]):
            if prev["token_throughput"]:
                gains.append({
                    "at": cur["concurrency"],
                    "gain_x": round(
                        cur["token_throughput"] / prev["token_throughput"], 3),
                })
        out["token_throughput_steps"] = gains
    return out


def _concurrency_warnings(levels: list[dict[str, Any]],
                          slots: int | None,
                          evidence: dict[str, Any] | None = None,
                          batching: dict[str, Any] | None = None) -> list[str]:
    out: list[str] = []
    for l in levels:
        if l["error_rate"] >= 0.2:
            out.append(
                f"并发 {l['concurrency']} 档有 {l['error_rate'] * 100:.0f}% 请求失败"
                f"（{l.get('failure_kind')}），该档吞吐不可比"
            )
        # A level that claims concurrency but ran requests end to end is not a
    # concurrency measurement; say so rather than publish a flat throughput.
        eff = l.get("effective_concurrency")
        if l["concurrency"] >= 4 and eff is not None and eff < 1.3:
            out.append(
                f"并发 {l['concurrency']} 档的实测并发度只有 {eff}"
                "（Σ 单请求耗时 ÷ 墙钟时间）—— 请求实际上接近串行，"
                "该档的吞吐数字不反映并发能力"
            )
    if slots:
        above = [l["concurrency"] for l in levels if l["concurrency"] > slots]
        if above:
            out.append(
                f"档位 {above} 超过了服务端槽位数 {slots}：超出的请求只能排队，"
                "这些档位测到的是「排队后的延迟」，不能读作并行能力"
            )

    # ---- a level can never exceed the number of requests it was given -----
    # This is the harness's own arithmetic, not the engine's behaviour: firing
    # 12 requests and calling the level "concurrency 128" measures 12.
    arithmetic = [l for l in levels if l["requests"] < l["concurrency"]]
    if arithmetic:
        detail = "、".join(
            f"{l['concurrency']} 档只发了 {l['requests']} 个请求"
            for l in arithmetic[:4])
        out.append(
            f"{detail}：一档的在飞请求数**不可能超过它发出的请求数**，"
            "所以这些档位测到的是同一个并发水平，"
            "把它们读成更高级别的并发能力会得到一条假的「吞吐见顶」曲线。"
            "要么提高「每档请求数」，要么把并发阶梯收到这个值以内"
        )

    # ---- enough requests, but the wave shape still diluted the level -----
    # Subtler than the arithmetic case and much easier to miss: the requests
    # are there, yet a single wave of N means the mean in-flight count is only
    # ~0.7*N (measured: 32 requests at level 32 read eff 22.69).  The
    # throughput is not wrong, it just is not a steady-state N-way reading.
    #
    # Engine-queued levels are excluded: their shortfall has a different cause
    # with its own message above, and telling a 64-way level with 128 requests
    # that it "只发了一波" would be simply false.
    diluted = [
        l for l in levels
        if l["requests"] >= l["concurrency"]
        and l.get("success_rate")
        and not l.get("engine_queued")
        and (l.get("effective_concurrency") or 0) < 0.8 * l["concurrency"]
    ]
    if diluted:
        detail = "、".join(
            f"{l['concurrency']} 档实测 {l.get('effective_concurrency')}"
            for l in diluted[:4])
        out.append(
            f"{detail} —— 不足声称并发数的 80%：请求数是够的，"
            "但这一档只发了一波，头尾阶段在飞请求不满把均值拉低了。"
            "把每档请求数提到 ≥ 2×并发档即可（两波能到 ~0.85–0.95N）："
            "这些吞吐数字本身没错，只是不能读作「N 路并发的稳定值」"
        )

    # ---- the engine's own parallelism, and the queueing above it ---------
    # The most important engine-side finding, and the one a client cannot make
    # alone: `effective_concurrency` counts a request as in flight for its whole
    # life, *including* time queued inside the engine.  Measured on a live node
    # at nominal 32: eff 28.58, engine running max 16, engine queue max 16 - so
    # over half of the "parallelism" was waiting, and every level above the
    # engine's limit was really a latency measurement wearing a throughput
    # label.  A boundary snapshot showed 0 queued (the engine was idle then).
    engine_cap = (max((l.get("engine_running_max") or 0) for l in levels)
                  if any(l.get("engine_running_max") is not None for l in levels)
                  else None)
    queued = [l for l in levels
              if l.get("engine_queued") and l.get("success_rate")]
    if queued and engine_cap:
        detail = "、".join(
            f"{l['concurrency']} 档（引擎同时在跑 {l.get('engine_running_max')}、"
            f"排队峰值 {l.get('engine_waiting_max')}）" for l in queued[:4])
        out.append(
            f"**引擎自己在排队**：{detail}。这些档位的「实测并发度」把**排队时间**"
            "也算进了在飞 —— 请求确实发出去了，但引擎并没那么多个同时在算。"
            f"这台引擎真正同时在跑的请求数上限约 **{engine_cap:.0f}**"
            "（vLLM 里就是 `--max-num-seqs`）。"
            "所以这些档位测到的是「排队后的延迟」，不是它们声称的并发能力：\n"
            f"  * 要测**并行度**：把阶梯收到 ≤ {engine_cap:.0f}，看「引擎在跑」列；\n"
            "  * 要测**容量**：就保持这种高压力负载，看「峰值 Token 吞吐」"
            "（它取所有档位的最高值，排队是达顶所必需的）。"
        )

    # ---- what the box delivers once the engine is saturated --------------
    # Above the engine's parallelism ceiling every extra request is queued, so
    # those levels all perform the same amount of real work.  If their aggregate
    # throughput barely moves, one number describes the machine - and it is also
    # the number the user should stop pushing load for.
    if engine_cap:
        sat = [(l["concurrency"], l["token_throughput"]) for l in levels
               if l["concurrency"] >= engine_cap and l.get("success_rate")
               and l.get("token_throughput")]
        if len(sat) >= 2:
            lo = min(sat, key=lambda p: p[1])
            hi = max(sat, key=lambda p: p[1])
            spread = (hi[1] - lo[1]) / hi[1] if hi[1] else 0.0
            if spread <= 0.15:
                pair = "、".join(f"并发 {c} 档 {v:.0f}"
                                for c, v in sorted(sat))
                out.append(
                    f"并发 ≥ {engine_cap:.0f} 的档位（引擎已饱和）聚合吞吐基本持平："
                    f"{pair} tok/s（相差 {spread * 100:.0f}%）—— 这台机器的"
                    "**容量就是这个水平**，再多加并发只会增加排队和延迟，"
                    "不会提高吞吐"
                )

    eff_ceiling = (batching or {}).get("effective_concurrency_ceiling")
    ceiling_at = (batching or {}).get("ceiling_at")
    engine_ceiling = engine_cap
    # Prefer the engine's own parallelism when we have it: `eff` is inflated by
    # queueing exactly in the regime where the ceiling matters.
    #
    # The `eff` quoted here is the maximum over *all* successful levels, not the
    # one inside the batching profile: that profile drops levels whose measured
    # concurrency fell just short of nominal (it is describing a decode curve),
    # and quoting its filtered value produced "highest eff 26.85" in a run whose
    # 64-way level read 50.64 - a plain factual error in a sentence whose whole
    # point is arithmetic.
    if engine_ceiling:
        at_level = next((l["concurrency"] for l in levels
                         if l.get("engine_running_max") == engine_ceiling), None)
        eff_points = [(l["concurrency"], l["effective_concurrency"])
                      for l in levels
                      if l.get("success_rate") and l.get("effective_concurrency")]
        if eff_points:
            # NB the tuple order: `eff_points` is (concurrency, eff), so the
            # max by value gives the *level* first and the value second.
            eff_at, eff_best = max(eff_points, key=lambda p: p[1])
            if eff_best > engine_ceiling * 1.15:
                out.append(
                    f"实测并发度最高 {eff_best}（并发 {eff_at} 档）**高于引擎"
                    f"自报的同时在跑上限 {engine_ceiling:.0f}**"
                    + (f"（引擎在并发 {at_level} 档就已到顶）" if at_level else "")
                    + "：差额就是引擎内部的排队 —— 所以这一列读的是「在飞」，"
                    "不是「在跑」。真正的并行度是引擎那个数"
                )
    if eff_ceiling and ceiling_at and not engine_ceiling:
        top = max(l["concurrency"] for l in levels)
        achievable = max(min(l["concurrency"], l["requests"]) for l in levels)
        top_level = next((l for l in levels if l["concurrency"] == top), None)
        eff_at_top = (top_level or {}).get("effective_concurrency") or 0
        if eff_ceiling < achievable * 0.7:
            out.append(
                f"整张表的实测并发度最高只有 {eff_ceiling}"
                f"（出现在并发 {ceiling_at} 档），而这些档位的理论上限是 "
                f"{achievable}：再往上加档位并没有真的增加并行度，"
                "多出来的请求在排队或被引擎内部串行化了"
            )
        elif top > eff_ceiling * 1.5 and eff_at_top <= eff_ceiling * 1.05:
            out.append(
                f"实测并发度在 {ceiling_at} 档就到达 {eff_ceiling} 并停止增长"
                f"（最高档 {top} 也只有 {eff_at_top}）—— "
                "这就是这台机器/这套配置的真实并发天花板，"
                f"并发 {ceiling_at} 以上的档位测的都是同一个水平"
            )

    # ---- throughput that does not rise with concurrency ------------------
    ordered = sorted(levels, key=lambda l: l["concurrency"])
    drops = [
        (ordered[i]["concurrency"], ordered[i + 1]["concurrency"])
        for i in range(len(ordered) - 1)
        if ordered[i]["request_throughput"] and ordered[i + 1]["request_throughput"]
        and ordered[i + 1]["request_throughput"]
        < ordered[i]["request_throughput"] * 0.7
    ]
    signs = sorted(set((evidence or {}).get("signs_of_overload") or []))
    if drops and signs:
        pairs = "; ".join(f"{a}→{b}" for a, b in drops[:3])
        out.append(
            f"吞吐在 {pairs} 处明显下降，且引擎自报有 {signs}："
            "说明服务端确实在拒请求、抢占或排队，"
            "建议核对引擎的并发上限（如 vLLM 的 --max-num-seqs）"
        )

    # ---- the batch-1 fast path: a curve that is FLAT, not falling --------
    # `ordered` is defined above (used for the drop pairs)
    b = batching or {}
    if b.get("batch1_fast_path"):
        out.append(
            f"**并发没有换来吞吐**：并发 {b['base_concurrency']} 时单请求解码 "
            f"{b.get('base_decode_tps')} tok/s，而 2 及以上的档位都降到约 "
            f"{b.get('batched_decode_tps')} tok/s"
            f"（单请求代价 {b.get('batching_tax_x')}×），"
            f"总吞吐最高只有 {b.get('peak_token_throughput')} tok/s"
            f"（并发 {b.get('peak_at_concurrency')} 档，"
            f"仅基线的 {b.get('throughput_gain_x')}×）。"
            "这不是测量误差 —— 引擎没有报错、没有排队、没有抢占，"
            "是它自己的批处理代价（常见于 batch-1 走快路径 / 投机解码只在 "
            "batch-1 生效）。要提吞吐得改引擎配置或换机器，加并发没用"
        )
    elif (b.get("batching_tax_x") or 0) >= 2.0:
        # Even when the ladder as a whole gains, the *first* step can be a
        # cliff: batch-1 is a fast path, so 1 -> 2 costs several times the
        # per-request decode and the aggregate actually falls before it
        # recovers.  That single step is what makes the curve look wrong.
        base_c = b.get("base_concurrency")
        low = next((l for l in ordered if l["concurrency"] == base_c), None)
        first_batched = next(
            (l for l in ordered if l["concurrency"] > base_c
             and l.get("token_throughput")), None)
        if (low and first_batched and low.get("token_throughput")
                and first_batched["token_throughput"]
                < low["token_throughput"]):
            out.append(
                f"并发 {base_c}→{first_batched['concurrency']} 时单请求解码从 "
                f"{b.get('base_decode_tps')} 掉到 "
                f"{first_batched.get('decode_tps')} tok/s"
                f"（单请求代价 {b.get('batching_tax_x')}×），"
                f"总吞吐先从 {b.get('base_token_throughput')} 降到 "
                f"{first_batched.get('token_throughput')} tok/s，"
                f"要到并发 {b.get('peak_at_concurrency')} 才回升到 "
                f"{b.get('peak_token_throughput')} tok/s。"
                "这不是测量误差（引擎无报错、无抢占、无排队），"
                "是它自己的批处理代价：batch-1 走的是快路径，"
                "一进入批处理单请求就慢下来，"
                "所以吞吐曲线前段先跌再涨 —— 看趋势要用后面的档位"
            )
    if drops and not signs and not any("批处理代价" in m for m in out):
        pairs = "; ".join(f"{a}→{b2}" for a, b2 in drops[:3])
        why = ("上方「实测并发度」和「批处理代价」两条说明了真实原因。"
               if (diluted or arithmetic
                   or b.get("effective_concurrency_ceiling"))
               else "更可能是批处理带来的单请求变慢，请对照单请求 TPOT 看。")
        # Be explicit about WHICH evidence backs "no queueing".  A gauge read
        # only at the level boundaries is read while the engine is idle, so it
        # can never rule queueing out - it just cannot see it.
        if (evidence or {}).get("measured_in_load"):
            quiet = ("引擎的仪表在**运行期间**采样过，没有出现抢占或排队"
                     "（计数器无异常）")
        else:
            quiet = ("引擎没有报抢占或排队（计数器无异常）—— 但仪表只在档位边界"
                     "（引擎空闲时）读过，不足以保证负载中没有排队")
        out.append(
            f"吞吐在 {pairs} 处下降，{quiet}："
            "所以不能归因于「服务端拒请求或排队超时」。"
            + why
        )

    # ---- every output token was thinking ---------------------------------
    # A thinking-by-default engine plus a small max_tokens means the output
    # budget is consumed by reasoning: the table then measures thinking speed
    # and calls it decode speed.  Measured on a real node: output_tokens ==
    # reasoning_tokens for every single sample of a full concurrency sweep.
    scored = [s for l in levels for s in (l.get("samples") or [])
              if s.get("ok")]
    n_out = sum(s.get("output_tokens") or 0 for s in scored)
    n_reason = sum(s.get("reasoning_tokens") or 0 for s in scored)
    if n_out and n_reason >= n_out * 0.99:
        out.append(
            "本次**全部输出 token 都是思考链**（正文为空）：表里的 decode / "
            "TPOT / token 吞吐衡量的是思考速度，不是回答速度。"
            "该引擎默认开启思考链，比并发前建议先把 `enable_thinking` 设为 false"
        )
    return out
