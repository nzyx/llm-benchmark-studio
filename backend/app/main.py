"""FastAPI application: probe, parameters, run, progress and history."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import db, params
from .config import PRESETS, resolve_retries, resolve_timeout
from .engine.base import ProgressEvent, build_result_document, utcnow_iso
from .engine.concurrency import ConcurrencyTask
from .engine.context import ContextTask
from .engine.performance import PerformanceTask
from .providers import (
    ProviderError,
    build_provider,
    detect,
    select_provider,
)
from .providers.llamacpp import LlamaCppProvider
from .schemas import DetectRequest, ProbeRequest, RunRequest

TERMINAL = ("completed", "partial", "failed", "timeout", "cancelled")
ACTIVE = ("queued", "running", "warmup", "testing", "connect", "pending")

# Task types that are actually implemented.  Presets list the ones they want;
# anything else is reported as pending rather than silently skipped.
TASK_REGISTRY: dict[str, type] = {
    "performance": PerformanceTask,
    "context": ContextTask,
    "concurrency": ConcurrencyTask,
}


def _resolve_tasks(preset_name: str | None, override: str | None) -> tuple[list[str], list[str]]:
    """Return ``(runnable, unimplemented)`` task lists for a preset."""
    if override:
        if override not in TASK_REGISTRY:
            return [], [override]
        return [override], []
    wanted = list((PRESETS.get(preset_name or "") or {}).get("tasks") or ["performance"])
    runnable = [t for t in wanted if t in TASK_REGISTRY]
    unimplemented = [t for t in wanted if t not in TASK_REGISTRY]
    return runnable or ["performance"], unimplemented


# ---------------------------------------------------------------------- #
# in-memory run registry (progress + live subscribers)
# ---------------------------------------------------------------------- #
RUNS: dict[str, "RunState"] = {}


class RunState:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status = "queued"
        self.events: list[dict[str, Any]] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.doc: dict[str, Any] | None = None
        self.task: asyncio.Task | None = None
        self.settings_warnings: list[str] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if len(self.events) > 3000:
            self.events = self.events[-1500:]
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield
    for state in list(RUNS.values()):
        if state.task and not state.task.done():
            state.task.cancel()


app = FastAPI(title="Local LLM Benchmark Studio", version="0.2.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------- #
# basics
# ---------------------------------------------------------------------- #
@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "llm-benchmark-studio",
        "version": "0.2.0",
        "phase": "Phase 6",
        "tasks": sorted(TASK_REGISTRY),
        "runs_total": db.count_runs(),
        "active_runs": sum(1 for s in RUNS.values() if s.status in ACTIVE),
    }


@app.get("/api/presets")
async def api_presets() -> dict[str, Any]:
    return {"presets": PRESETS}


@app.get("/api/params")
async def api_params() -> dict[str, Any]:
    """Parameter registry: the single source of truth the UI renders from."""
    return params.registry_payload()


# ---------------------------------------------------------------------- #
# probe
# ---------------------------------------------------------------------- #
def _model_param_values(model: Any, health: dict[str, Any]) -> dict[str, Any]:
    """Read-only values for the registry's ``model`` category."""
    values = model.to_param_values()
    values["server_slots"] = health.get("total_slots") or health.get("slots")
    values["server_kv_pool"] = (health.get("kv_pool_positions")
                                or health.get("slot_ctx"))
    values["server_kv_concurrency"] = health.get("kv_cache_max_concurrency")
    return values


@app.post("/api/detect")
async def api_detect(req: DetectRequest) -> dict[str, Any]:
    """Fingerprint the engine without issuing any generation request."""
    t0 = time.perf_counter()
    detection = await detect(req.base_url, req.api_key,
                             timeout_s=resolve_timeout(req.timeout_s),
                             retries=1)
    payload = detection.to_dict()
    payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return payload


@app.post("/api/probe")
async def api_probe(req: ProbeRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    timeout = resolve_timeout(req.timeout_s)

    detection = await detect(req.base_url, req.api_key,
                             timeout_s=timeout, retries=1)
    provider = build_provider(req.base_url, req.api_key,
                              family=detection.family, timeout_s=timeout,
                              retries=1, detection=detection)
    try:
        health = await provider.health_check()
        models = await provider.get_models()
        if isinstance(provider, LlamaCppProvider):
            models = provider.enrich_models(models, health)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        await provider.aclose()

    # Measured capabilities supersede what static inspection could infer: a
    # vLLM node declares nothing about its chat template, so the only way to
    # know whether thinking can be steered is to try it - and once tried, that
    # answer must reach the UI instead of the earlier "unsupported" guess.
    measured = dict(provider.capabilities or {})
    for key, value in measured.items():
        if getattr(detection.capabilities, key, None) is not None:
            setattr(detection.capabilities, key, bool(value))

    warnings: list[str] = list(detection.notes)
    warnings.extend(health.get("_warnings") or [])
    if not models:
        warnings.append("服务未返回任何模型，请确认 /v1/models 是否可用。")

    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        "provider": provider.provider_info(),
        "adapter": provider.kind,
        "detection": detection.to_dict(),
        "capabilities": measured,
        "thinking_probe": getattr(provider, "thinking_probe", None),
        "thinks_by_default": getattr(provider, "thinks_by_default", None),
        "models": [m.to_dict() for m in models],
        # read-only values for the `model` parameter category, per model id
        "model_param_values": {
            m.id: _model_param_values(m, health) for m in models
        },
        "health": health,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------- #
# runs
# ---------------------------------------------------------------------- #
@app.post("/api/runs")
async def api_create_run(req: RunRequest) -> dict[str, Any]:
    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset: {req.preset}")

    runnable, unimplemented = _resolve_tasks(req.preset, req.task_type)
    if not runnable:
        raise HTTPException(
            status_code=400,
            detail=f"task_type {req.task_type!r} 尚未实现（可用："
                   + ", ".join(TASK_REGISTRY) + "）")

    doc = build_result_document(
        task_type=runnable[0],
        provider_info={"kind": "openai_compatible", "base_url": req.base_url,
                       "engine": None, "api_version": None},
        model_info={"id": req.model or "auto"},
        config={},
        preset=req.preset,
        label=req.label,
    )
    run_id = doc["run_id"]
    state = RunState(run_id)
    RUNS[run_id] = state

    payload = {
        "base_url": req.base_url,
        "api_key": req.api_key,
        "model": req.model,
        "preset": req.preset,
        "label": req.label,
        "timeout_s": req.timeout_s,
        "retries": req.retries,
        "family": req.family,
        "task_types": runnable,
        "performance": req.performance.model_dump(exclude_none=True)
        if req.performance else {},
        "context": req.context.model_dump(exclude_none=True)
        if req.context else {},
        "concurrency": req.concurrency.model_dump(exclude_none=True)
        if req.concurrency else {},
    }
    state.task = asyncio.create_task(_run_worker(run_id, payload, doc))

    notes: list[str] = []
    if unimplemented:
        notes.append("以下任务尚未实现，将在后续阶段接入："
                     + ", ".join(unimplemented))

    return {
        "run_id": run_id,
        "status": "queued",
        "preset": req.preset,
        "task_types": runnable,
        "settings_warnings": [],
        "note": " / ".join(notes) if notes else None,
    }


@app.get("/api/runs")
async def api_list_runs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    return {"runs": db.list_runs(limit=limit, offset=offset),
            "total": db.count_runs()}


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: str) -> dict[str, Any]:
    state = RUNS.get(run_id)
    if state is not None and state.doc is not None:
        return {"status": state.status, "live": True, "result": state.doc,
                "settings_warnings": state.settings_warnings}
    stored = db.get_run(run_id)
    if stored is not None:
        return {"status": stored.get("status"), "live": False, "result": stored}
    if state is not None:
        return {"status": state.status, "live": True, "result": None,
                "progress": state.events[-1] if state.events else None,
                "settings_warnings": state.settings_warnings}
    raise HTTPException(status_code=404, detail="run not found")


@app.delete("/api/runs/{run_id}")
async def api_delete_run(run_id: str) -> dict[str, Any]:
    state = RUNS.get(run_id)
    if state and state.task and not state.task.done():
        state.task.cancel()
    removed = db.delete_run(run_id)
    RUNS.pop(run_id, None)
    if not removed and not state:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": True, "run_id": run_id}


@app.get("/api/runs/{run_id}/events")
async def api_run_events(run_id: str) -> StreamingResponse:
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")

    queue = state.subscribe()
    backlog = list(state.events)

    async def gen():
        try:
            for ev in backlog:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if state.status in TERMINAL:
                yield ("data: "
                       + json.dumps({"status": state.status, "final": True},
                                    ensure_ascii=False)
                       + "\n\n")
                return
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    if state.status in TERMINAL:
                        yield ("data: "
                               + json.dumps({"status": state.status, "final": True},
                                            ensure_ascii=False)
                               + "\n\n")
                        return
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev.get("final"):
                    return
        finally:
            state.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------- #
# worker
# ---------------------------------------------------------------------- #
def _capability_gates(provider: Any, cfg: dict[str, Any],
                      task_type: str = "performance") -> list[str]:
    """Drop settings the engine cannot honour, and say so.

    Silently ignoring a knob is worse than refusing it: the run would look
    configured while measuring something else entirely.
    """
    warnings: list[str] = []

    options = cfg.get("options") or {}
    cache_on = provider.supports("prompt_cache_enabled")
    if options.get("cache_prompt") is False and not provider.supports(
            "cache_prompt_control"):
        options.pop("cache_prompt", None)
        cfg["options"] = options or None
        warnings.append(
            "该引擎未暴露 prompt cache 开关，`cache_prompt=false` 已忽略"
        )

    # If the cache cannot be switched off but is known to be on, a repeated
    # input is *guaranteed* to be served partly from cache - which makes TTFT
    # and prefill describe the cache rather than the engine.  Leaving it to
    # the user to notice is not good enough; take the one action that is
    # always correct here and explain it.
    if cache_on and not provider.supports("cache_prompt_control") \
            and "unique_prompt" in cfg and not cfg.get("unique_prompt"):
        cfg["unique_prompt"] = True
        warnings.append(
            "该引擎 prompt cache 默认开启且无法关闭，已自动改用**不重复输入**"
            "（`unique_prompt`）—— 否则每次采样都会命中缓存，"
            "TTFT / Prefill 测到的是缓存而不是算力"
        )

    if (cfg.get("enable_thinking") is not None
            and not provider.supports("thinking_control")):
        cfg["enable_thinking"] = None
        nested = (cfg.get("options") or {}).get("chat_template_kwargs")
        if isinstance(nested, dict):
            nested.pop("enable_thinking", None)
            if not nested:
                cfg["options"].pop("chat_template_kwargs", None)
        warnings.append(
            "该引擎既未声明、实测也未生效思考链控制，`enable_thinking` 已忽略"
        )

    # A thinking-by-default engine spends the output budget on reasoning
    # before it writes anything.  At small max_tokens that can consume the
    # whole budget, so `ttft_content` never arrives and "decode" silently
    # becomes "reasoning speed".  Warn *before* the run, not after.
    if (cfg.get("enable_thinking") is None and provider.supports(
            "thinking_control")):
        effort = getattr(provider, "reasoning_effort_default", None)
        measured = getattr(provider, "thinks_by_default", None) is True
        budget = int(cfg.get("max_tokens") or 0)
        if (effort in ("high", "xhigh") or measured) and 0 < budget <= 256:
            how = (f"reasoning_effort={effort}" if effort in ("high", "xhigh")
                   else "实测默认会先输出思考链")
            warnings.append(
                f"该引擎默认开启思考链（{how}），而本次 max_tokens={budget} "
                "—— 输出预算很可能被思考链吃光，正文为空。"
                "建议把 `enable_thinking` 设为 false"
            )

    if cfg.get("detect_max_context") and not provider.supports("tokenize"):
        warnings.append(
            "该引擎未提供 /tokenize，上下文长度只能按字符数估算，"
            "最大上下文探测的结果会偏保守"
        )

    # Decode is measured from *when the tokens arrive*.  A very short output
    # can be flushed in one burst, and then the first-to-last token span
    # collapses: measured on a live vLLM node, the same engine reported a
    # steady 135-152 tok/s at max_tokens>=32 and an apparent 196,078 tok/s at
    # max_tokens=8, where the whole reply arrived inside 0.036 ms.  The
    # harness now withholds such numbers, but the user should hear it before
    # the run rather than wonder why a column says "—".
    if task_type_gate := (cfg.get("_task_type") or ""):  # noqa: F841 - see below
        # `max_tokens` only gates a decode measurement; the concurrency task
        # reports throughput per level and never publishes a decode rate, so
        # the warning would be noise there.
        pass
    budget = int(cfg.get("max_tokens") or 0)
    spec_on = provider.supports("speculative_decoding")
    if (cfg.get("_task_type") or "") != "concurrency" and 0 < budget < 32:
        if spec_on:
            warnings.append(
                f"max_tokens={budget} 偏小，且该引擎启用了投机解码"
                "（一个调度步产出多个 token）—— 这么短的输出很可能被合并成"
                "一次 flush，`decode_tps` 会被置空。测解码速度建议 "
                "`max_tokens` ≥ 64"
            )
        else:
            warnings.append(
                f"max_tokens={budget} 偏小：很短的输出可能被引擎作为一次 flush "
                "送达，首末 token 的时间差会塌缩，`decode_tps` 将无法测量并被"
                "置空。建议把 `max_tokens` 提到 64 以上再比较解码速度"
            )
    return warnings


async def _run_worker(run_id: str, payload: dict[str, Any],
                      doc: dict[str, Any]) -> None:
    state = RUNS[run_id]
    state.status = "running"
    timeout = resolve_timeout(payload.get("timeout_s"))
    retries = resolve_retries(payload.get("retries"))
    provider = None
    started = time.perf_counter()

    async def emit(event: ProgressEvent) -> None:
        state.publish(event.to_dict())

    try:
        state.publish({"status": "running", "phase": "detect",
                       "message": "识别引擎…"})
        provider, detection = await select_provider(
            payload["base_url"], payload.get("api_key"),
            family=payload.get("family"), timeout_s=timeout, retries=retries)
        doc["detection"] = detection.to_dict() if detection else None
        if detection:
            state.publish({
                "status": "running", "phase": "detect",
                "message": (f"引擎识别：{detection.family} → "
                            f"{detection.adapter}（置信度 {detection.confidence}）"),
            })

        state.publish({"status": "running", "phase": "connect",
                       "message": "探测模型与能力…"})
        try:
            models = await provider.get_models()
            health = await provider.health_check()
            if isinstance(provider, LlamaCppProvider):
                models = provider.enrich_models(models, health)
        except ProviderError as exc:
            raise RuntimeError(f"无法连接服务: {exc}") from exc

        if not models:
            raise RuntimeError("服务未返回任何模型。")
        chosen = payload.get("model")
        model = next((m for m in models if m.id == chosen), models[0]) \
            if chosen else models[0]

        # Capabilities are only fully known *after* health_check: the thinking
        # probe runs there.  Snapshotting before it would record "cannot steer
        # thinking" for an engine that just proved it can.
        if detection is not None:
            for key, value in (provider.capabilities or {}).items():
                if getattr(detection.capabilities, key, None) is not None:
                    setattr(detection.capabilities, key, bool(value))
            doc["detection"] = detection.to_dict()
        doc["capabilities"] = dict(provider.capabilities)

        doc["provider"] = provider.provider_info()
        doc["model"] = model.to_dict()
        doc["model_param_values"] = _model_param_values(model, health)
        doc["status"] = "running"
        state.doc = doc

        # ---- run each task into its own document section ----------------
        # One run can carry several tasks (the "full" preset).  Each task gets
        # a fresh sub-document so tasks cannot clobber each other's samples;
        # the first one is mirrored to the top level for backwards
        # compatibility with the performance-only UI.
        preset_cfg = PRESETS.get(payload.get("preset") or "") or {}
        doc["task_results"] = {}
        mirrored = False
        model_values = doc.get("model_param_values") or {}
        server_context = model_values.get("context_window")
        server_slots = model_values.get("server_slots")

        for task_type in payload.get("task_types") or ["performance"]:
            cls = TASK_REGISTRY[task_type]
            merged = dict(preset_cfg.get(task_type) or {})
            merged.update(payload.get(task_type) or {})

            if task_type == "context":
                cfg, warnings = params.build_context_config(
                    model.id, merged, server_context=server_context)
            elif task_type == "concurrency":
                cfg, warnings = params.build_concurrency_config(
                    model.id, merged, server_slots=server_slots,
                    server_context=server_context)
            else:
                cfg, warnings = params.build_performance_config(model.id, merged)
            cfg["request_timeout_s"] = timeout
            # findings from the capability probe belong in the run's warnings:
            # they were measured, and some of them change what the user should do
            warnings = list(warnings) + list(health.get("_warnings") or [])
            warnings.extend(_capability_gates(provider, cfg, task_type))

            sub = build_result_document(
                task_type=task_type,
                provider_info=provider.provider_info(),
                model_info=model.to_dict(),
                config=cfg,
                preset=payload.get("preset"),
                label=payload.get("label"),
            )
            sub["run_id"] = doc["run_id"]
            sub["detection"] = doc.get("detection")
            sub["capabilities"] = doc.get("capabilities")
            sub["model_param_values"] = doc.get("model_param_values")
            sub["settings_warnings"] = warnings
            sub["status"] = "running"

            state.settings_warnings = warnings
            if warnings:
                state.publish({
                    "status": "running", "phase": "config",
                    "message": f"[{task_type}] 参数已校正：" + "；".join(warnings),
                })

            await cls(provider, cfg, sub, emit).execute()
            doc["task_results"][task_type] = sub

            if not mirrored:
                for key in ("samples", "stats", "result", "contexts", "levels",
                            "data_warnings", "config", "settings_warnings"):
                    if key in sub:
                        doc[key] = sub[key]
                mirrored = True

            if sub.get("status") != "completed":
                state.publish({
                    "status": sub.get("status") or "failed",
                    "message": f"[{task_type}] 未正常完成，后续任务中止",
                })
                break

        statuses = [s.get("status") for s in doc["task_results"].values()]
        if statuses and all(s == "completed" for s in statuses):
            doc["status"] = "completed"
        elif any(s in ("completed", "partial") for s in statuses):
            # something usable was measured, but not everything that was asked
            # for - calling that "completed" would be a lie
            doc["status"] = "partial"
        else:
            doc["status"] = "failed"
        state.status = doc["status"]
    except asyncio.CancelledError:
        state.status = "cancelled"
        doc["status"] = "cancelled"
        state.publish({"status": "cancelled", "message": "运行已取消"})
        raise
    except Exception as exc:  # noqa: BLE001
        state.status = "failed"
        doc["status"] = "failed"
        errors = doc.setdefault("errors", {"count": 0, "rate": 0.0, "items": []})
        errors["items"].append({"message": f"{type(exc).__name__}: {exc}"})
        errors["count"] = len(errors["items"])
        state.publish({"status": "failed", "message": str(exc)})
    finally:
        # the top-level document must carry its own timing: each task records
        # its own, but the report's "total duration" is the whole run.
        # Saving happens here (and only here) so that whatever the outcome -
        # success, failure, cancellation - exactly one write lands, and it
        # includes the timing.
        doc["finished_at"] = utcnow_iso()
        doc["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        try:
            db.save_run(doc)
        except Exception:  # noqa: BLE001 - persistence must not mask the run
            pass
        if provider is not None:
            await provider.aclose()
        state.publish({"status": state.status, "message": "run finished",
                       "run_id": run_id, "final": True})


# ---------------------------------------------------------------------- #
# optional: serve the built frontend (used by Docker / preview)
# ---------------------------------------------------------------------- #
def _mount_static() -> None:
    from pathlib import Path

    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if not dist.exists():
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(dist), html=True), name="static")


_mount_static()
