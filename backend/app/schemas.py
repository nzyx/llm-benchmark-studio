"""Pydantic request models for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProbeRequest(BaseModel):
    base_url: str = Field(..., description="OpenAI-compatible base URL")
    api_key: str | None = None
    timeout_s: float | None = None


class DetectRequest(BaseModel):
    base_url: str
    api_key: str | None = None
    timeout_s: float | None = None


class PerformanceOverrides(BaseModel):
    """Mirrors ``app.params.PERFORMANCE_KEYS`` plus the passthrough keys.

    Anything outside this set is rejected by the registry with a warning
    rather than being silently applied.
    """

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    warmup: int | None = None
    repeats: int | None = None
    use_chat: bool | None = None
    stream: bool | None = None
    cache_prompt: bool | None = None
    unique_prompt: bool | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    options: dict[str, Any] | None = None
    # accepted so the registry can explain that they land in a later phase
    concurrency: int | None = None


class ContextOverrides(BaseModel):
    """Mirrors ``app.params.CONTEXT_KEYS`` plus the passthrough keys."""

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    warmup: int | None = None
    repeats: int | None = None
    use_chat: bool | None = None
    cache_prompt: bool | None = None
    unique_prompt: bool | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    context_ladder: str | None = None
    context_fill: str | None = None
    detect_max_context: bool | None = None
    options: dict[str, Any] | None = None


class ConcurrencyOverrides(BaseModel):
    """Mirrors ``app.params.CONCURRENCY_KEYS`` plus the passthrough keys."""

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    warmup: int | None = None
    use_chat: bool | None = None
    cache_prompt: bool | None = None
    unique_prompt: bool | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    concurrency_ladder: str | None = None
    requests_per_level: int | None = None
    options: dict[str, Any] | None = None


class RunRequest(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str | None = None
    preset: str = "quick"
    label: str | None = None
    timeout_s: float | None = None
    retries: int | None = None
    # pin the adapter instead of auto-detecting (llamacpp | vllm | openai_compatible)
    family: str | None = None
    # force a single task instead of the preset's list
    # (performance | context | concurrency)
    task_type: str | None = None
    performance: PerformanceOverrides | None = None
    context: ContextOverrides | None = None
    concurrency: ConcurrencyOverrides | None = None
