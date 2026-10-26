"""ContextTask - performance as a function of input length.

Two things make this honest:

1. **Exact input length.**  Inputs are built to a target *token* count using
   the engine's own ``/tokenize`` (and trimmed with ``/detokenize``).  Both
   the target and the measured count are recorded, so a report never has to
   pretend the two are equal.

2. **No prefix sharing between ladder steps.**  With repeated filler text a
   short input is a literal prefix of a longer one, so an engine with prompt
   caching would serve part of the longer prefill from cache and make long
   contexts look *faster* than they are.  Every step therefore carries a
   unique marker as its very first tokens, which structurally defeats any
   prefix cache regardless of whether the engine honours ``cache_prompt``.

The auto-detected ceiling is called **Maximum Verified Context**: it is the
largest input this run *observed to work*, not the model's theoretical limit.
"""

from __future__ import annotations

from typing import Any

from ..config import (
    CONTEXT_DECODE_TOKS_ASSUMED,
    CONTEXT_DETECT_MAX_STEPS,
    CONTEXT_DETECT_SAFE_CEILING,
    CONTEXT_PREFILL_TOKS_ASSUMED,
    CONTEXT_TIMEOUT_MARGIN_S,
    HARD_MAX_CONTEXT,
    HARD_MAX_TIMEOUT_S,
)
from ..errors import KIND_TOO_LONG, KIND_UNREACHABLE
from .base import BenchmarkTask, ProgressEvent
from .context_builder import Calibration, build_input, calibrate
from .probe import (
    MARKER_PLACEHOLDER,
    MARKER_WIDTH,
    cache_warning,
    decode_coalesced_warning,
    input_fingerprint,
    input_identity_warning,
    input_length_warning,
    input_marker,
    measure_once,
    prefill_contamination_warning,
    summarize_samples,
    worst_kind,
)


def _pad_to(chars: int) -> str:
    """A calibration probe body: filler text of a known character length."""
    from .context_builder import _FILLER

    if chars <= 0:
        return ""
    reps = chars // max(1, len(_FILLER)) + 1
    return (_FILLER * reps)[:chars]


class ContextTask(BenchmarkTask):
    task_type = "context"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._contexts: list[dict[str, Any]] = []
        self._max_context: dict[str, Any] | None = None
        # set the moment the engine stops answering; every later measurement
        # would be fiction, so the task stops instead of inventing numbers
        self._abort_reason: str | None = None
        # tightest known-good / known-bad pair seen so far
        self._bracket: dict[str, int | None] = {"lo": None, "hi": None}
        # why the ladder stopped climbing (the engine may still be alive)
        self._halt: str | None = None
        # measurement-quality notes collected while running (widened timeouts,
        # output that turned out to be pure reasoning, ...)
        self._notes: list[str] = []
        #: measured text->token ratio, only needed when the engine has no
        #: /tokenize endpoint
        self._calibration: Calibration | None = None

    # ------------------------------------------------------------------ #
    async def prepare(self) -> None:
        try:
            self.doc["metrics_snapshot"]["before"] = await self.provider.get_metrics()
        except Exception:  # noqa: BLE001
            pass
        # Ask the *engine* whether it can tokenize, with the model name, rather
        # than trusting the capability flag.  A real vLLM node advertised
        # ``/tokenize`` while its adapter needed an explicit model, so every
        # input was silently built from a hard-coded 4 chars/token and landed
        # 25% short - and because the flag said "supported", calibration was
        # skipped too, so nothing corrected it.
        if not await self._tokenizer_works():
            await self._calibrate()

    async def _tokenizer_works(self) -> bool:
        try:
            ids = await self.provider.tokenize("calibration probe",
                                               self.config.get("model"))
        except Exception:  # noqa: BLE001 - a probe must not break the run
            return False
        return bool(ids)

    async def _calibrate(self) -> None:
        """Measure the engine's own chars-per-token ratio.

        Without ``/tokenize`` there is no exact way to hit a target length, and
        a hard-coded "4 characters per token" was measured to be off by
        10-19% on a real engine - enough to make "8K" mean 6.6K.  Two probe
        requests of different length let us solve for the ratio *and* the
        chat-template overhead on this specific engine, which costs a few
        seconds once per run and removes most of the error.
        """
        observations: list[tuple[int, int]] = []
        for chars in (1200, 4800):
            text = input_marker(True) + _pad_to(chars)
            timeout_s, _w = scaled_timeout(
                max(64, int(chars / 2)), 8,
                float(self.config.get("request_timeout_s") or HARD_MAX_TIMEOUT_S))
            sample = await self._generate(text, timeout_s, max_tokens=8)
            got = sample.get("input_tokens")
            if sample.get("ok") and got:
                observations.append((len(text), int(got)))

        self._calibration = calibrate(observations)
        cal = self._calibration
        if cal.observations:
            note = (f"引擎未提供 /tokenize，已用 {len(cal.observations)} 次探针"
                    f"标定长度换算：{cal.chars_per_token:.2f} 字符/token，"
                    f"模板开销约 {cal.overhead_tokens:.0f} tokens")
            self._notes.append(note)
            await self.emit(ProgressEvent(
                status="warmup", phase="calibrate", message=note,
            ))
        else:
            self._notes.append(
                "引擎未提供 /tokenize，且标定探针未取到有效样本，"
                "输入长度只能按默认比例估算（约 4 字符/token）"
            )

    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        cfg = self.config
        ladder: list[int] = list(cfg.get("context_ladder") or [])
        ceiling = _resolve_ceiling(cfg)
        # the output budget competes with the input for the same window:
        # an input of exactly `n_ctx` leaves no room for a single token
        budget = max(0, int(cfg.get("max_tokens") or 0))
        usable = max(512, ceiling - budget)
        strategy = cfg.get("context_fill") or "repeat_padding"
        repeats = max(1, int(cfg.get("repeats", 1)))
        warmup = max(0, int(cfg.get("warmup", 0)))

        runnable = [t for t in ladder if t <= usable]
        skipped = [t for t in ladder if t > usable]

        # Failure bracket discovered by the ladder itself.  Initialised wide
        # and narrowed as steps fail; the binary search below then only has to
        # work inside [largest ok, smallest failed] instead of re-probing a
        # range it already knows the answer to.
        self._bracket = {"lo": None, "hi": None}

        self._contexts.append({
            "kind": "plan",
            "ceiling": usable,
            "raw_ceiling": ceiling,
            "output_budget": budget,
            "strategy": strategy,
            "skipped": skipped,
            "calibration": (self._calibration.to_dict()
                            if self._calibration else None),
        })

        total = len(runnable)
        for i, target in enumerate(runnable):
            if self._abort_reason or self._halt:
                break
            await self.emit(ProgressEvent(
                status="testing", phase="context", done=i, total=total,
                message=f"context {_human(target)}（{i + 1}/{total}）",
                extra={"target_tokens": target},
            ))
            entry = await self._probe_context(target, strategy, repeats, warmup)
            self._contexts.append(entry)
            self._track_bracket(target, entry)
            self.doc["_done"], self.doc["_total"] = i + 1, total
            await self.emit(ProgressEvent(
                status="testing", phase="context", done=i + 1, total=total,
                message=f"context {_human(target)} {'OK' if entry['ok'] else 'FAILED'}",
                extra={
                    "target_tokens": target,
                    "last": {
                        "ok": entry["ok"],
                        "ttft_ms": entry.get("ttft_ms"),
                        "prefill_tps": entry.get("prefill_tps"),
                        "e2e_ms": entry.get("e2e_ms"),
                        "failure_kind": entry.get("failure_kind"),
                    },
                },
            ))

        # Anything not attempted is reported, never silently dropped: a ladder
        # that stops at 16K must not look like a ladder that only had 8K steps.
        attempted = [c["target_tokens"] for c in self._contexts
                     if c.get("kind") == "context" and c.get("samples")]
        reason = ("engine_unavailable" if self._abort_reason
                  else "beyond_first_failure" if self._halt else None)
        if reason and attempted:
            last = max(attempted)
            for target in runnable:
                if target <= last:
                    continue
                self._contexts.append({
                    "kind": "context",
                    "target_tokens": target,
                    "built": None,
                    "samples": [],
                    "ok": False,
                    "skipped_reason": reason,
                    "failure_kind": (KIND_UNREACHABLE
                                     if self._abort_reason else None),
                    "error": self._abort_reason or self._halt,
                })

        if cfg.get("detect_max_context") and not self._abort_reason:
            self._max_context = await self._detect_max_context(
                runnable, usable, strategy, warmup)
        elif cfg.get("detect_max_context") and self._abort_reason:
            self._max_context = self._aborted_detection(
                "引擎在探测过程中失联，未能完成最大上下文探测",
                lo=self._bracket.get("lo"), hi=self._bracket.get("hi"))

    def _track_bracket(self, target: int, entry: dict[str, Any]) -> None:
        """Remember the tightest known-good / known-bad pair."""
        if entry.get("skipped_reason"):
            return
        if entry.get("ok"):
            lo = self._bracket.get("lo")
            self._bracket["lo"] = max(lo or 0, target)
        else:
            hi = self._bracket.get("hi")
            self._bracket["hi"] = min(hi, target) if hi else target

    # ------------------------------------------------------------------ #
    async def _probe_context(self, target: int, strategy: str,
                             repeats: int, warmup: int) -> dict[str, Any]:
        cfg = self.config
        # One build per step, measured with a placeholder marker so the marker
        # is part of the token budget; every sample then swaps in a fresh tag
        # of the same width.  Sharing one *exact* input across warmup+repeats
        # was a real bug: on an engine that cannot switch off its prefix cache
        # the repeats were served from cache, and prefill inside a single
        # ladder step was measured at both 252 and 2000 tok/s.
        prefix = input_marker(False)          # placeholder, same width
        built = await build_input(
            self.provider, target, strategy=strategy,
            calibration=self._calibration,
            model=cfg.get("model"),
            prefix=prefix,
        )
        # the built text starts with the marker; keep the body so each sample
        # can substitute a fresh tag while the token count stays the same
        body = built.text[MARKER_WIDTH:]
        # Honour the switch instead of always forcing uniqueness: the user may
        # deliberately want to measure the cached path.  Either way the input
        # fingerprint is recorded, so a run that reused one input is detectable
        # rather than something the reader has to guess at.
        unique = bool(cfg.get("unique_prompt", True))

        def fresh_text() -> str:
            return input_marker(unique) + body

        timeout_s, widened = scaled_timeout(
            built.actual_tokens or target,
            int(cfg.get("max_tokens") or 0),
            float(cfg.get("request_timeout_s") or HARD_MAX_TIMEOUT_S),
        )
        if widened:
            note = (f"{_human(target)} 档的 prefill 按保守估算需要更久，"
                    f"单请求超时已从 "
                    f"{cfg.get('request_timeout_s'):g}s 放宽到 "
                    f"{timeout_s:g}s（上限 {HARD_MAX_TIMEOUT_S:g}s）")
            self._notes.append(note)
            await self.emit(ProgressEvent(
                status="testing", phase="context", done=0, total=0,
                message=note,
            ))

        samples: list[dict[str, Any]] = []
        for i in range(warmup):
            s = await self._generate(fresh_text(), timeout_s)
            s["warmup"] = True
            s["index"] = -1 - i
            samples.append(s)
            if not s.get("ok"):
                await self._settle_after_failure(target)

        for i in range(repeats):
            s = await self._generate(fresh_text(), timeout_s)
            s["warmup"] = False
            s["index"] = i
            samples.append(s)
            if not s.get("ok"):
                await self._settle_after_failure(target)

        summary = summarize_samples(samples)
        stats = summary["stats"]
        ok = bool(summary["ok_samples"])
        errors = [s["error"] for s in samples if s.get("error")]
        kind = worst_kind(samples)

        # A failed step means one of two very different things.  Asking the
        # engine whether it is still there settles it, and a run that keeps
        # probing a dead server is worse than useless: it produces a number.
        if not ok:
            await self._handle_failure(kind, target, errors[0] if errors else None)

        built_info = built.to_dict()
        # the builder's notes were previously only visible inside the step
        # payload; a degraded input length is a result-level fact
        for note in built.notes:
            if note not in self._notes:
                self._notes.append(note)

        return {
            "kind": "context",
            "target_tokens": target,
            "built": built_info,
            "requested_samples": repeats + warmup,
            "scored_samples": summary["scored"],
            "warmup_excluded": summary["warmup_excluded"],
            "samples": samples,
            "ok": ok,
            "error": errors[0] if errors else None,
            "errors": errors,
            "failure_kind": kind,
            "ttft_ms": stats["ttft_ms"]["mean"],
            "ttft_ms_p95": stats["ttft_ms"]["p95"],
            "prefill_tps": stats["prefill_tps"]["mean"],
            "decode_tps": stats["decode_tps"]["mean"],
            "e2e_ms": stats["e2e_ms"]["mean"],
            "input_tokens": stats["input_tokens"]["mean"],
            "output_tokens": stats["output_tokens"]["mean"],
            # server-reported, for cross-checking our own client-side numbers
            "server_prompt_ms": stats["server_prompt_ms"]["mean"],
            "server_prefill_tps": stats["server_prefill_tps"]["mean"],
            "cache_hit_tokens": stats["cache_hit_tokens"]["mean"],
            # the slice of TTFT that was NOT prefill (queueing / retry /
            # restore).  Surfaced per step because it is the one figure that
            # says whether this point's `prefill_tps` may be read as a rate.
            "queue_wait_ms": stats["queue_wait_ms"]["max"],
            "prefill_contaminated": bool(
                prefill_contamination_warning(samples,
                                             f"{_human(target)} 档")),
            "stats": stats,
        }

    async def _handle_failure(self, kind: str | None, target: int,
                              detail: str | None) -> None:
        """Decide what a failed step means: capacity limit, or dead engine.

        A drop or a refusal both stop the ladder - a length that failed cannot
        become usable at a *longer* length - but only a dead engine aborts the
        run.  The one cheap liveness GET tells the two apart, and it is worth
        paying on every failure: an engine that just OOM-restarted under a long
        prefill must not be mistaken for one that politely said "too long".
        """
        if not await self._engine_alive():
            if await self._wait_for_engine():
                # it came back: the failed step keeps its failure kind, but the
                # run continues from a live server
                self._halt = self._halt or (
                    f"{_human(target)} 处引擎短暂中断后已恢复")
                return
            self._abort_reason = (
                f"引擎在 {_human(target)} 这一档之后失联"
                f"（{kind or 'unknown'}: {detail or 'no detail'}），"
                "后续测量已中止 —— 结果只覆盖失联之前的档位"
            )
            return
        self._halt = self._halt or (
            f"引擎在 {_human(target)} 处拒绝或中断了请求"
            f"（{kind or 'unknown'}），更长的输入不再尝试")

    async def _engine_alive(self) -> bool:
        return await self.engine_alive()

    async def _wait_for_engine(self) -> bool:
        return await self.wait_for_engine()

    async def _settle_after_failure(self, target: int) -> None:
        """Let the engine finish the work it did NOT abandon when we did.

        This is the difference between a ladder that measures prefill and one
        that measures a queue.  A client-side failure - a read error, a broken
        stream - ends *our* interest in the request, not the engine's: the
        prefill it started keeps running and holds a slot.  Sending the next
        sample straight away puts it behind that job, and the wait is charged
        to the next sample's TTFT.

        Measured on a real fork: the 128K warmup died on a read error at 50 s
        while its prefill still needed ~85 s, so the scored sample came back at
        121,144 ms of TTFT against the engine's own 85,410 ms of prefill - a
        prefill rate of 1083 tok/s where the truth is 1536.  That single point
        is what made the whole ladder read as "longer context is faster".
        """
        waited = await self.wait_for_idle()
        if waited < 1.0:
            return
        note = (f"{_human(target)} 档有一次请求失败，引擎仍在处理它 —— "
                f"已等待 {waited:.1f}s 让引擎排空后再继续，"
                "否则下一个样本会把这段等待算进 prefill")
        if note not in self._notes:
            self._notes.append(note)
        await self.emit(ProgressEvent(
            status="testing", phase="context", done=0, total=0, message=note,
        ))

    async def _generate(self, text: str, timeout_s: float | None = None,
                        max_tokens: int | None = None) -> dict[str, Any]:
        cfg = self.config
        use_chat = bool(cfg.get("use_chat", True))
        return await measure_once(
            self.provider,
            model=cfg["model"],
            messages=[{"role": "user", "content": text}] if use_chat else None,
            prompt=None if use_chat else text,
            max_tokens=max_tokens if max_tokens is not None
            else cfg.get("max_tokens", 16),
            temperature=cfg.get("temperature", 0.0),
            options=cfg.get("options"),
            timeout_s=timeout_s or cfg.get("request_timeout_s"),
        )

    # ------------------------------------------------------------------ #
    async def _detect_max_context(self, runnable: list[int], ceiling: int,
                                  strategy: str, warmup: int) -> dict[str, Any]:
        """Binary search for the largest input this engine accepts.

        Bounded by a safe ceiling, a per-probe timeout, a retry budget and a
        hard step limit.  The ladder that just ran usually already brackets
        the answer, so the search starts from ``[largest ok, smallest failed]``
        rather than from zero - which both saves probes and avoids hammering
        the engine with inputs we already know it cannot hold.
        """
        cfg = self.config
        retries = int(cfg.get("retries") or 0)
        known_good = min(runnable) if runnable else min(1024, ceiling)
        for entry in self._contexts:
            if entry.get("kind") == "context" and entry.get("ok"):
                known_good = max(known_good, int(entry["target_tokens"]))

        lo = max(known_good, self._bracket.get("lo") or 0)
        hi = self._bracket.get("hi") or ceiling
        bracket_from_ladder = bool(self._bracket.get("hi"))
        steps: list[dict[str, Any]] = []

        await self.emit(ProgressEvent(
            status="testing", phase="max-context", done=0, total=0,
            message=(f"二分搜索最大可验证上下文"
                     f"（下界 {_human(lo)}，上限 {_human(hi)}"
                     f"{'，来自阶梯夹逼' if bracket_from_ladder else ''}）"),
        ))

        aborted: str | None = None
        while (hi - lo > max(1024, int(lo * 0.1))
               and len(steps) < CONTEXT_DETECT_MAX_STEPS):
            mid = (lo + hi) // 2
            ok, detail, kind = await self._probe_once(mid, strategy, retries)
            if kind == KIND_UNREACHABLE:
                aborted = (f"二分搜索在 {_human(mid)} 处遇到引擎失联：{detail}")
                break
            steps.append({"target_tokens": mid, "ok": ok,
                          "detail": detail, "failure_kind": kind})
            if ok:
                lo = mid
            else:
                hi = mid
            await self.emit(ProgressEvent(
                status="testing", phase="max-context", done=len(steps),
                total=CONTEXT_DETECT_MAX_STEPS,
                message=f"试探 {_human(mid)} → {'成功' if ok else '失败'}",
                extra={"last": {"target_tokens": mid, "ok": ok}},
            ))

        if aborted:
            self._max_context = self._aborted_detection(
                aborted, lo=lo, hi=hi, steps=steps)
            self._max_context["bracket_from_ladder"] = bracket_from_ladder
            return self._max_context

        # With a ladder bracket, the tightest honest statement is the tested
        # bracket itself: `lo` verified working, `hi` observed failing.
        return {
            "label": "Maximum Verified Context",
            "value": lo,
            "lower_bound": known_good,
            "upper_bound": hi if hi < ceiling else None,
            "safe_ceiling": ceiling,
            "method": "binary_search",
            "steps": steps,
            "step_count": len(steps),
            "bracket_from_ladder": bracket_from_ladder,
            "verified": bool(steps and lo > known_good),
            "disclaimer": (
                "这是本次运行**实测通过**的最大输入长度，不是模型的理论上下文上限。"
                "理论值请以服务端上报的上下文窗口为准。"
            ),
        }

    @staticmethod
    def _aborted_detection(reason: str, lo: int | None = None,
                           hi: int | None = None,
                           steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """A detection that stopped early still knows *something*.

        ``value`` stays the largest length observed to work and
        ``upper_bound`` the smallest observed to fail, so the answer is an
        honest bracket instead of a bare "no result".
        """
        return {
            "label": "Maximum Verified Context",
            "value": lo,
            "upper_bound": hi,
            "method": "binary_search",
            "steps": steps or [],
            "step_count": len(steps or []),
            "aborted": True,
            "abort_reason": reason,
            "verified": False,
            "disclaimer": (
                "探测**未能完成**：`value` 只是中断之前实测通过的最大长度，"
                "`upper_bound` 是首次观察到的失败长度 —— 真正的上限夹在这两者之间。"
                "请确认引擎存活后重跑，以获得收敛后的结论。"
            ),
        }

    async def _probe_once(self, target: int, strategy: str,
                          retries: int) -> tuple[bool, str | None, str | None]:
        """One probe. Returns ``(ok, detail, failure_kind)``.

        Only transport hiccups are retried; an engine that answered "no" is
        not going to change its mind, and retrying it just burns minutes.
        """
        last_detail: str | None = None
        last_kind: str | None = None
        for _ in range(retries + 1):
            built = await build_input(
                self.provider, target, strategy=strategy,
                calibration=self._calibration,
                model=self.config.get("model"),
            )
            if built.tokenize_failed:
                # the tokenizer is unreachable, so sending the request would
                # only produce a misleading failure
                return False, built.notes[-1] if built.notes else "tokenize failed", \
                    KIND_UNREACHABLE
            text = input_marker(True) + built.text
            timeout_s, _widened = scaled_timeout(
                built.actual_tokens or target,
                int(self.config.get("max_tokens") or 0),
                float(self.config.get("request_timeout_s") or HARD_MAX_TIMEOUT_S),
            )
            sample = await self._generate(text, timeout_s)
            if sample.get("ok"):
                return True, None, None
            last_detail = sample.get("error") or "unknown error"
            last_kind = sample.get("error_kind")
            if last_kind == KIND_UNREACHABLE:
                return False, last_detail, last_kind
            if last_kind != "timeout":
                return False, last_detail, last_kind  # a rejection
        return False, f"重试 {retries} 次后仍失败：{last_detail}", last_kind

    # ------------------------------------------------------------------ #
    async def collect(self) -> None:
        try:
            self.doc["metrics_snapshot"]["after"] = await self.provider.get_metrics()
        except Exception:  # noqa: BLE001
            pass

    async def calculate(self) -> None:
        contexts = [c for c in self._contexts if c.get("kind") == "context"]
        tested = [c for c in contexts if c.get("samples")]
        good = [c for c in tested if c.get("ok")]
        failed = [c for c in tested if not c.get("ok")]
        plan = next((c for c in self._contexts if c.get("kind") == "plan"), {})

        self.doc["contexts"] = contexts
        self.doc["stats"] = {
            k: _series(good, k)
            for k in ("ttft_ms", "prefill_tps", "decode_tps", "e2e_ms",
                      "input_tokens", "output_tokens", "server_prompt_ms",
                      "server_prefill_tps")
        }
        all_samples = [s for c in tested for s in (c.get("samples") or [])]
        cached = cache_warning(all_samples)
        # the same comparability rules the performance task applies, applied
        # per ladder step: a step's mean must not mix cache hits with cold runs
        comparability: list[str] = []
        for c in tested:
            label = _human(c["target_tokens"])
            for msg in (input_identity_warning(c.get("samples") or [],
                                               f"{label} 档"),
                        input_length_warning(c.get("samples") or [],
                                             f"{label} 档")):
                if msg and msg not in comparability:
                    comparability.append(msg)
        coalesced: list[str] = []
        for c in tested:
            if not c.get("samples"):
                continue
            msg = decode_coalesced_warning(c["samples"],
                                           f"{_human(c['target_tokens'])} 档")
            if msg and msg not in coalesced:
                coalesced.append(msg)

        contaminated: list[str] = []
        for c in tested:
            if not c.get("samples"):
                continue
            msg = prefill_contamination_warning(
                c["samples"], f"{_human(c['target_tokens'])} 档")
            if msg and msg not in contaminated:
                contaminated.append(msg)

        self.doc["data_warnings"] = (
            comparability + coalesced + contaminated
            + _reasoning_only_warning(good) + self._notes
            + _context_warnings(tested, plan)
            + ([cached] if cached else [])
        )

        self.doc["result"] = {
            "ladder": [c["target_tokens"] for c in contexts],
            "tested": len(tested),
            "passed": len(good),
            "failed": len(failed),
            "skipped": plan.get("skipped", []),
            "safe_ceiling": plan.get("ceiling"),
            "output_budget": plan.get("output_budget"),
            "fill_strategy": plan.get("strategy"),
            "max_verified_context": self._max_context,
            "largest_passing": max((c["target_tokens"] for c in good), default=None),
            "smallest_failing": min((c["target_tokens"] for c in failed), default=None),
            "scaling": _scaling(good),
            "prefill_fit": _prefill_fit(good),
            "calibration": (self._calibration.to_dict()
                            if self._calibration else None),
            "aborted": bool(self._abort_reason),
            "abort_reason": self._abort_reason,
        }
        if self._abort_reason:
            # not a failure - the data we did collect is valid, it just stops
            # at a known point, and the report must say so
            self.doc["status"] = "partial"
            self._record_error(self._abort_reason)

    async def save(self) -> None:
        return None


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _resolve_ceiling(cfg: dict[str, Any]) -> int:
    """Never exceed the engine's own context, our hard cap, or the safe ceiling."""
    declared = cfg.get("server_context") or cfg.get("model_context")
    candidates = [HARD_MAX_CONTEXT, CONTEXT_DETECT_SAFE_CEILING]
    if declared:
        candidates.append(int(declared))
    if cfg.get("max_context"):
        candidates.append(int(cfg["max_context"]))
    return max(512, min(candidates))


def scaled_timeout(input_tokens: int, max_tokens: int,
                   base_timeout: float) -> tuple[float, bool]:
    """A request timeout that fits the work, not the config default.

    Prefill is the dominant cost of a long step and runs at a fairly
    predictable tok/s, so the needed time is computable *before* sending.
    Widening here is not the same as removing the limit: it is still bounded,
    it just has to be bounded by the work, not by a number chosen for short
    prompts.  Observed failure mode without this: a 64K step needs ~180s of
    prefill, the 120s default kills it mid-computation, and the engine goes
    down with it.

    Returns ``(timeout_s, widened)`` - always clamped by the hard cap.
    """
    assumed = (input_tokens / CONTEXT_PREFILL_TOKS_ASSUMED
               + max_tokens / CONTEXT_DECODE_TOKS_ASSUMED
               + CONTEXT_TIMEOUT_MARGIN_S)
    needed = max(base_timeout, min(assumed, HARD_MAX_TIMEOUT_S))
    return needed, needed > base_timeout + 0.5


def _human(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):g}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


def _series(entries: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Trimmed summary: one point per context, not a distribution."""
    values = [(e["target_tokens"], e.get(key)) for e in entries
              if e.get(key) is not None]
    if not values:
        return {"count": 0, "points": [], "min": None, "max": None, "mean": None}
    nums = [v for _t, v in values]
    return {
        "count": len(values),
        "points": [{"x": t, "y": v} for t, v in values],
        "min": round(min(nums), 4),
        "max": round(max(nums), 4),
        "mean": round(sum(nums) / len(nums), 4),
    }


def _prefill_fit(good: list[dict[str, Any]]) -> dict[str, Any]:
    """Split prefill cost into a fixed part and a marginal rate.

    This exists to answer a question the raw table cannot.  ``prefill_tps``
    is ``tokens / ttft`` - an *average* over the whole climb, so it necessarily
    rises while a fixed per-request cost is being amortised and then falls once
    the engine's real bandwidth limit dominates.  Read as a curve it looks like
    "longer context is faster", which is nonsense for a throughput.

    Fitting ``ttft = fixed + tokens / rate`` over the ladder separates the two
    effects, so the short-context rise has a number attached instead of being
    mistaken for an anomaly.  Measured on a real fork (1.3 M ctx): the naive
    rate climbs 1321 -> 1680 tok/s from 2K to 16K and then degrades to 1325 at
    512K - i.e. a fixed cost that matters only at the short end, plus a genuine
    decline at the long end.

    Only *clean* points take part: a step whose TTFT contained a queue wait
    would corrupt the slope, which is the whole reason contamination is
    detected above.
    """
    points = [
        (float(c["input_tokens"]), float(c["ttft_ms"]))
        for c in good
        if c.get("input_tokens") and c.get("ttft_ms")
        and not c.get("prefill_contaminated")
    ]
    if len(points) < 3:
        return {}
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if den == 0:
        return {}
    slope = (n * sxy - sx * sy) / den          # ms per token
    intercept = (sy - slope * sx) / n          # ms, the fixed cost
    if slope <= 0:
        return {}
    out: dict[str, Any] = {
        "fixed_ms": round(intercept, 1),
        "marginal_ms_per_token": round(slope, 4),
        "marginal_tps": round(1000 / slope, 1),
        "points": n,
    }
    # the honest reading of a *negative* intercept: it means the rate degrades
    # with length, so no single fixed+marginal pair describes the whole ladder
    if intercept < 0:
        out["note"] = (
            "固定开销拟合为负值，说明解码率本身随输入变长而下降；"
            "边际速率只代表长档位区间的平均，不是全阶梯的常数"
        )
    else:
        out["note"] = (
            f"每次请求约有 {intercept / 1000:.2f}s 的固定开销（不随输入变化），"
            f"在此之上按约 {1000 / slope:,.0f} tok/s 处理输入。"
            "短档位的 `prefill_tps` 偏低是因为固定开销被摊薄得少，"
            "不代表长档位真的更快。"
        )
    return out


def _scaling(good: list[dict[str, Any]]) -> dict[str, Any]:
    """How badly throughput degrades as the input grows."""
    if len(good) < 2:
        return {}
    ordered = sorted(good, key=lambda c: c["target_tokens"])
    first, last = ordered[0], ordered[-1]
    out: dict[str, Any] = {
        "from_tokens": first["target_tokens"],
        "to_tokens": last["target_tokens"],
        "ratio": round(last["target_tokens"] / max(first["target_tokens"], 1), 2),
    }
    for key in ("ttft_ms", "prefill_tps", "e2e_ms"):
        a, b = first.get(key), last.get(key)
        if a and b:
            out[f"{key}_first"] = a
            out[f"{key}_last"] = b
            out[f"{key}_change_x"] = round(b / a, 3)
    return out


def _reasoning_only_warning(good: list[dict[str, Any]]) -> list[str]:
    """The output budget was eaten by the thinking chain.

    Observed on a real run: every sample reported 64 output tokens and *zero*
    visible content - the whole budget went to reasoning.  That silently
    changes what "decode" measures and leaves `ttft_content` undefined, so the
    report has to say so instead of presenting the numbers unqualified.
    """
    samples = [s for c in good for s in (c.get("samples") or []) if s.get("ok")]
    if not samples:
        return []
    with_content = [s for s in samples if (s.get("content_chars") or 0) > 0]
    if with_content or not any((s.get("reasoning_chars") or 0) > 0 for s in samples):
        return []
    return [
        "输出 token 全部是思考链，没有正文（ttft_content 缺失）—— decode 指标"
        "衡量的是 reasoning 生成速度。上下文测试建议把 `enable_thinking` 设为 "
        "false，让输出预算花在正文上。"
    ]


def _context_warnings(contexts: list[dict[str, Any]],
                      plan: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for c in contexts:
        if c.get("ok"):
            continue
        kind = c.get("failure_kind")
        label = _human(c["target_tokens"])
        if kind == KIND_TOO_LONG:
            out.append(
                f"context {label} 被引擎拒绝（{c.get('error')}）—— "
                "这通常是真实的容量边界，不是测量错误"
            )
        elif kind == KIND_UNREACHABLE:
            out.append(
                f"context {label} 时引擎失联（{c.get('error')}）—— "
                "该档及之后的结果已作废，请确认服务存活后重跑"
            )
        else:
            out.append(f"context {label} 失败：{c.get('error')}")
    if plan.get("skipped"):
        skipped = ", ".join(_human(t) for t in plan["skipped"])
        budget = plan.get("output_budget") or 0
        out.append(
            f"以下长度超过可用上限 {_human(plan['ceiling'])}"
            f"（已为输出预留 {budget} tokens），已跳过：{skipped}"
        )
    # a non-monotone TTFT curve usually means contention, not physics
    good = sorted((c for c in contexts if c.get("ok")),
                  key=lambda c: c["target_tokens"])
    drops = [
        (good[i]["target_tokens"], good[i + 1]["target_tokens"])
        for i in range(len(good) - 1)
        if good[i].get("ttft_ms") and good[i + 1].get("ttft_ms")
        and good[i + 1]["ttft_ms"] < good[i]["ttft_ms"] * 0.7
    ]
    if drops:
        pairs = "; ".join(f"{_human(a)}→{_human(b)}" for a, b in drops[:3])
        out.append(
            f"TTFT 在 {pairs} 处不升反降，可能是有其它请求在抢资源，"
            "或缓存未完全绕过；建议重跑确认"
        )
    return out
