"""PerformanceTask - TTFT / Prefill / Decode / TPOT / ITL / E2E.

Measurement itself lives in :mod:`app.engine.probe` so every task measures
identically.  This module owns the sampling loop (warmup + repeats), the
unique-prompt guarantee against prompt caching, and the data-quality guards.
"""

from __future__ import annotations

from typing import Any

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
)

DEFAULT_PROMPT = (
    "Write a concise technical paragraph (about 120 words) explaining why "
    "the KV cache is the dominant memory consumer during long-context LLM "
    "inference. Be precise and avoid filler."
)


class PerformanceTask(BenchmarkTask):
    task_type = "performance"

    # ------------------------------------------------------------------ #
    async def prepare(self) -> None:
        try:
            self.doc["metrics_snapshot"]["before"] = await self.provider.get_metrics()
        except Exception:  # noqa: BLE001 - metrics are best-effort
            pass

    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        cfg = self.config
        warmup = int(cfg.get("warmup", 1))
        repeats = int(cfg.get("repeats", 3))
        total = warmup + repeats
        done = 0

        for i in range(warmup):
            await self.emit(ProgressEvent(
                status="warmup", phase="performance", done=done, total=total,
                message=f"warmup {i + 1}/{warmup}",
            ))
            sample = await self._one_sample(index=-1, warmup=True)
            if not sample.get("ok"):
                self._record_error(f"warmup: {sample.get('error')}")
            done += 1

        for i in range(repeats):
            sample = await self._one_sample(index=i)
            self.doc["samples"].append(sample)
            if not sample.get("ok"):
                self._record_error(sample.get("error") or "unknown error")
            done += 1
            self.doc["_done"], self.doc["_total"] = done, total
            await self.emit(ProgressEvent(
                status="testing", phase="performance", done=done, total=total,
                message=f"sample {i + 1}/{repeats}",
                extra={"last": _brief(sample)},
            ))

    # ------------------------------------------------------------------ #
    async def _one_sample(self, index: int, warmup: bool = False) -> dict[str, Any]:
        cfg = self.config
        use_chat = bool(cfg.get("use_chat", True))

        prompt_text = cfg.get("prompt") or DEFAULT_PROMPT
        # A content-level guarantee against prompt caching: even if the engine
        # ignores `cache_prompt`, no two inputs are ever equal.  The marker is
        # shared with the other tasks so all three produce the same kind of
        # input - see `probe.input_marker` for why it leads the prompt.
        unique = bool(cfg.get("unique_prompt", True))
        if unique:
            prompt_text = input_marker(True) + prompt_text

        if use_chat:
            messages = cfg.get("messages") or [
                {"role": "user", "content": prompt_text}
            ]
            prompt = None
        else:
            messages = None
            prompt = prompt_text

        sample = await measure_once(
            self.provider,
            model=cfg["model"],
            messages=messages,
            prompt=prompt,
            max_tokens=cfg.get("max_tokens", 256),
            temperature=cfg.get("temperature", 0.0),
            options=cfg.get("options"),
        )
        sample["index"] = index
        sample["warmup"] = warmup
        sample["unique_prompt"] = unique
        return sample

    # ------------------------------------------------------------------ #
    async def collect(self) -> None:
        try:
            self.doc["metrics_snapshot"]["after"] = await self.provider.get_metrics()
        except Exception:  # noqa: BLE001
            pass

    async def calculate(self) -> None:
        summary = summarize_samples(self.doc["samples"])
        stats = summary["stats"]
        ok_samples = summary["ok_samples"]
        n = len(self.doc["samples"])
        n_ok = len(ok_samples)

        self.doc["stats"] = stats
        self.doc["data_warnings"] = _data_warnings(self.doc["samples"], n, n_ok)

        self.doc["result"] = {
            "samples_total": n,
            "samples_ok": n_ok,
            "success_rate": round(n_ok / n, 4) if n else 0.0,
            "primary": {
                "ttft_ms": stats["ttft_ms"]["mean"],
                "decode_tps": stats["decode_tps"]["mean"],
                "prefill_tps": stats["prefill_tps"]["mean"],
                "tpot_ms": stats["tpot_ms"]["mean"],
                "e2e_ms": stats["e2e_ms"]["mean"],
                "output_tokens": stats["output_tokens"]["mean"],
            },
            "server": _server_crosscheck(ok_samples),
        }


def _server_crosscheck(ok: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare our client-side TTFT with the engine's own prefill time.

    When both are available they should agree; a large gap means one of the
    two is not measuring what it claims (a proxy in the path, a queueing
    delay, or a cached prefill the client cannot see).  Reporting both is what
    makes the number auditable rather than merely asserted.
    """
    with_server = [s for s in ok if s.get("server_prompt_ms") is not None]
    if not with_server:
        return {"available": False}
    client = [s["ttft_ms"] for s in with_server if s.get("ttft_ms")]
    server = [s["server_prompt_ms"] for s in with_server]
    cached = sum(1 for s in ok if (s.get("cache_hit_tokens") or 0) > 0)
    out: dict[str, Any] = {
        "available": True,
        "samples": len(with_server),
        "prompt_ms_mean": round(sum(server) / len(server), 2),
        "cache_hit_samples": cached,
    }
    if client:
        out["client_ttft_ms_mean"] = round(sum(client) / len(client), 2)
        ratio = out["prompt_ms_mean"] / out["client_ttft_ms_mean"]
        out["prompt_over_ttft"] = round(ratio, 3)
    drafts = [s["draft_tokens"] for s in with_server
              if s.get("draft_tokens")]
    if drafts and sum(drafts) > 0:
        acc = [s.get("draft_accepted") or 0 for s in with_server]
        # Deliberately NOT called an "acceptance ratio": on the real fork the
        # two fields tracked each other exactly (51/51, 52/52) but were also
        # seen at 54/49, i.e. accepted > proposed, which is impossible if both
        # are token counts.  The semantics of the pair are not documented, so
        # the field is named after the arithmetic, not after a claim.
        out["draft_accepted_over_draft_n"] = round(sum(acc) / sum(drafts), 4)
    return out


def _data_warnings(samples: list[dict[str, Any]], n: int, n_ok: int) -> list[str]:
    """A number is only meaningful if the samples behind it are comparable.

    The two comparability checks are shared helpers (see ``probe``) so this
    task cannot drift from the context and concurrency tasks: all three must
    judge their own samples by the same rules.
    """
    out: list[str] = []
    ok = [s for s in samples if s.get("ok")]
    identity = input_identity_warning(samples)
    if identity:
        out.append(identity)
    length = input_length_warning(samples)
    if length:
        out.append(length)
    if n and (n - n_ok) / n >= 0.2:
        out.append(f"{n - n_ok}/{n} 次采样失败，均值可能不代表稳定状态")
    cached = cache_warning(samples)
    if cached:
        out.append(cached)
    coalesced = decode_coalesced_warning(samples)
    if coalesced:
        out.append(coalesced)
    contaminated = prefill_contamination_warning(samples)
    if contaminated:
        out.append(contaminated)
    if _reasoning_only(ok):
        out.append(
            "输出 token 全部是思考链，没有正文（ttft_content 缺失）—— decode "
            "指标衡量的是 reasoning 生成速度。本机默认思考强度较高，"
            "建议把 `enable_thinking` 设为 false。"
        )
    return out


def _reasoning_only(ok: list[dict[str, Any]]) -> bool:
    """Every successful sample spent its whole output budget on thinking."""
    usable = [s for s in ok if s.get("output_tokens")]
    if not usable:
        return False
    return (not any((s.get("content_chars") or 0) > 0 for s in usable)
            and any((s.get("reasoning_chars") or 0) > 0 for s in usable))


def _brief(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": sample.get("ok"),
        "ttft_ms": sample.get("ttft_ms"),
        "decode_tps": sample.get("decode_tps"),
        "e2e_ms": sample.get("e2e_ms"),
    }
