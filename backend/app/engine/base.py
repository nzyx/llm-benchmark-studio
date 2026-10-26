"""BenchmarkTask base class + the unified result JSON schema."""

from __future__ import annotations

import abc
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------- #
# progress
# ---------------------------------------------------------------------- #
@dataclass
class ProgressEvent:
    """A single progress tick streamed to the frontend."""

    status: str            # running | warmup | testing | completed | failed | timeout
    phase: str = ""        # human readable phase, e.g. "performance"
    done: int = 0
    total: int = 0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressEmitter = Callable[[ProgressEvent], Awaitable[None]]


async def _noop_emitter(_: ProgressEvent) -> None:
    return None


# ---------------------------------------------------------------------- #
# result document
# ---------------------------------------------------------------------- #
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def build_result_document(
    *,
    task_type: str,
    provider_info: dict[str, Any],
    model_info: dict[str, Any],
    config: dict[str, Any],
    preset: str | None = None,
    label: str | None = None,
    host_agent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an empty result document following the unified schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": new_run_id(),
        "created_at": utcnow_iso(),
        "task_type": task_type,
        "preset": preset,
        "label": label,
        "status": "pending",
        "provider": provider_info,
        "model": model_info,
        "config": config,
        "started_at": None,
        "finished_at": None,
        "duration_ms": None,
        "samples": [],
        "stats": {},
        "result": {},
        "errors": {"count": 0, "rate": 0.0, "items": []},
        "metrics_snapshot": {"before": None, "after": None},
        "host_agent": host_agent,
    }


# ---------------------------------------------------------------------- #
# task base
# ---------------------------------------------------------------------- #
class BenchmarkTask(abc.ABC):
    """Unified lifecycle: prepare() -> run() -> collect() -> calculate() -> save().

    Subclasses implement the hooks; :meth:`execute` drives the whole thing and
    keeps the result document consistent.
    """

    task_type: str = "task"

    def __init__(
        self,
        provider: Any,
        config: dict[str, Any],
        doc: dict[str, Any],
        emit: ProgressEmitter | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.doc = doc
        self.emit = emit or _noop_emitter
        self._t_start = 0.0

    # -- hooks ------------------------------------------------------------
    async def prepare(self) -> None:
        """Validate config, resolve model, capture pre-run metrics."""

    @abc.abstractmethod
    async def run(self) -> None:
        """Execute the benchmark, filling ``self.doc['samples']``."""

    async def collect(self) -> None:
        """Post-run collection (metrics snapshot, cleanup)."""

    @abc.abstractmethod
    async def calculate(self) -> None:
        """Turn raw samples into aggregate stats in ``self.doc['stats']``."""

    async def save(self) -> None:
        """Persistence hook - performed by the caller after ``execute``."""
        return None

    # -- driver -----------------------------------------------------------
    async def execute(self) -> dict[str, Any]:
        import time
        self._t_start = time.perf_counter()
        self.doc["started_at"] = utcnow_iso()
        try:
            await self.emit(ProgressEvent(status="running", phase=self.task_type,
                                          message="preparing"))
            await self.prepare()
            await self.run()
            await self.collect()
            await self.calculate()
            # a task may have already decided its own outcome (e.g. the
            # context task stops early when the engine dies and reports
            # ``partial``); don't overwrite a more specific verdict
            if self.doc.get("status") not in ("partial", "failed", "timeout"):
                self.doc["status"] = "completed"
        except TimeoutError as exc:
            self.doc["status"] = "timeout"
            self._record_error(f"timeout: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.doc["status"] = "failed"
            self._record_error(f"{type(exc).__name__}: {exc}")
        finally:
            self.doc["finished_at"] = utcnow_iso()
            self.doc["duration_ms"] = round(
                (time.perf_counter() - self._t_start) * 1000.0, 3)
            self._finalize_errors()
            await self.emit(ProgressEvent(
                status=self.doc["status"], phase=self.task_type,
                done=self.doc.get("_done", 0), total=self.doc.get("_total", 0),
                message="finished",
            ))
            self.doc.pop("_done", None)
            self.doc.pop("_total", None)
        return self.doc

    # -- helpers ----------------------------------------------------------
    async def engine_alive(self) -> bool:
        """Cheap liveness probe: no generation, tiny timeout, never raises.

        Lives on the base class because every long-running task needs the same
        answer to the same question: is the engine still there?  A task that
        keeps issuing requests to a dead server produces failures, not
        measurements.
        """
        from ..config import CONTEXT_ENGINE_PING_TIMEOUT_S

        try:
            return await self.provider.ping(timeout_s=CONTEXT_ENGINE_PING_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - liveness must never raise
            return False

    async def wait_for_engine(self) -> bool:
        """Give an engine that just died a short chance to come back.

        Observed behaviour under a long prefill: the process exits and is
        serving again seconds later (an OOM restart).  Waiting is cheap
        compared with throwing away the rest of a run.
        """
        import asyncio

        from ..config import (
            CONTEXT_ENGINE_RECOVERY_TRIES,
            CONTEXT_ENGINE_RECOVERY_WAIT_S,
        )

        for _ in range(CONTEXT_ENGINE_RECOVERY_TRIES):
            await asyncio.sleep(CONTEXT_ENGINE_RECOVERY_WAIT_S)
            if await self.engine_alive():
                return True
        return False

    async def busy_slots(self) -> int | None:
        """How many requests the engine says it is currently processing.

        ``None`` means "the engine does not tell us" - a different statement
        from ``0``, and the difference decides whether a caller may trust a
        quiet reading.  llama.cpp exposes ``llamacpp:requests_processing``,
        vLLM ``vllm:num_requests_running``.
        """
        try:
            metrics = await self.provider.get_metrics()
        except Exception:  # noqa: BLE001 - best effort, never fatal
            return None
        normalized = (metrics or {}).get("normalized") or {}
        value = normalized.get("requests_running")
        return int(value) if isinstance(value, (int, float)) else None

    async def wait_for_idle(self, max_wait_s: float = 60.0,
                            poll_s: float = 1.0,
                            fallback_s: float = 3.0) -> float:
        """Let the engine finish whatever it is still doing.  Returns seconds.

        A sample that fails mid-stream leaves the engine computing work the
        client has already given up on - the engine cannot un-start a prefill.
        Firing the next request immediately puts it in a queue, and the queue
        wait lands inside the *next* request's TTFT, where it silently becomes
        part of a "prefill rate".  Measured on a real fork: a 128K warmup died
        at 50 s while its prefill still needed ~85 s, and the following sample
        reported 1083 tok/s instead of the engine's own 1536.

        Polls the engine's own busy gauge when it has one.  When it does not,
        waits a short bounded grace period instead of pretending nothing
        happened - a fixed sleep is honest about being a guess.
        """
        import asyncio
        import time

        started = time.perf_counter()

        seen = await self.busy_slots()
        if seen is None:
            await asyncio.sleep(min(fallback_s, max_wait_s))
            return time.perf_counter() - started

        while seen > 0 and (time.perf_counter() - started) < max_wait_s:
            await asyncio.sleep(poll_s)
            seen = await self.busy_slots()
            if seen is None:
                break
        return time.perf_counter() - started

    # -- helpers ----------------------------------------------------------
    def _record_error(self, message: str) -> None:
        self.doc["errors"]["items"].append({
            "at": utcnow_iso(),
            "message": message,
        })

    def _finalize_errors(self) -> None:
        errs = self.doc["errors"]["items"]
        total = len(self.doc["samples"]) + len(errs)
        self.doc["errors"]["count"] = len(errs)
        self.doc["errors"]["rate"] = round(len(errs) / total, 4) if total else 0.0
