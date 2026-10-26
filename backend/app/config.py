"""Global configuration and HARD safety ceilings.

Every benchmark request is clamped against these values on the server side,
so a malformed or malicious client can never make the engine loop forever.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------
# By default the benchmark client connects DIRECTLY to the inference server
# and ignores HTTP_PROXY / HTTPS_PROXY from the environment.  A local model
# server lives on the LAN, and routing it through a proxy routinely breaks
# plain-HTTP requests (observed: a local sandbox proxy turned /v1/models and
# /metrics into 404s).  Set LLMB_TRUST_ENV_PROXY=1 if you really are behind a
# corporate proxy and need the environment honoured.
TRUST_ENV_PROXY = os.environ.get("LLMB_TRUST_ENV_PROXY", "").strip() in (
    "1", "true", "True", "yes",
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "bench.sqlite3"

# --------------------------------------------------------------------------
# HARD ceilings - these are absolute and cannot be raised by a request.
# --------------------------------------------------------------------------
HARD_MAX_CONTEXT = 1_048_576      # 1M tokens
HARD_MAX_CONCURRENCY = 256
HARD_MAX_OUTPUT_TOKENS = 65_536
HARD_MAX_REQUESTS = 500           # total samples per single run
HARD_MAX_TIMEOUT_S = 1800.0       # 30 min per HTTP request
HARD_MAX_PROMPT_CHARS = 8_000_000

# --------------------------------------------------------------------------
# Transport defaults.
#
# NOTE: per-parameter defaults and bounds live in ``app/params.py`` - that is
# the single source of truth.  Only transport-level knobs belong here.
# --------------------------------------------------------------------------
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RETRIES = 2

# Default context ladder (tokens).  Values above the model's advertised
# context are skipped automatically.
DEFAULT_CONTEXT_LADDER = [
    512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144,
]

DEFAULT_CONCURRENCY_LADDER = [1, 2, 4, 8, 16, 32]

# Context auto-detect (binary search) guards.
CONTEXT_DETECT_SAFE_CEILING = 1_048_576
CONTEXT_DETECT_MAX_STEPS = 12

# A long-context prefill can take an engine down (observed: a 128K prefill
# killed a llama.cpp server, which was serving again seconds later - an OOM
# restart).  Before declaring a run dead, give the engine a short grace
# window: a recovered server is worth waiting for, a permanently dead one is
# not worth measuring at all.
CONTEXT_ENGINE_RECOVERY_TRIES = 3
CONTEXT_ENGINE_RECOVERY_WAIT_S = 5.0
#: how long the liveness check waits before calling the engine dead
CONTEXT_ENGINE_PING_TIMEOUT_S = 5.0

# Long-step timeout scaling.
#
# Observed on the real engine: prefill runs at ~365-450 tok/s on long inputs,
# so a 64K step needs ~180s of *pure prefill* - while the default request
# timeout is 120s.  A fixed timeout therefore kills the request while the
# engine is still computing, which both wastes the work and (observed) can
# take the engine down with it.  The context task widens the timeout from a
# deliberately conservative tok/s floor instead.
CONTEXT_PREFILL_TOKS_ASSUMED = 150
CONTEXT_DECODE_TOKS_ASSUMED = 10
CONTEXT_TIMEOUT_MARGIN_S = 30.0

# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "quick": {
        "label": "Quick Test",
        "description": "短时间快速评测：1 次预热 + 5 次正式请求。",
        "tasks": ["performance"],
        "performance": {"warmup": 1, "repeats": 5, "max_tokens": 256},
    },
    "performance": {
        "label": "Performance Test",
        "description": "重点测试 Prefill / Decode / TTFT。",
        "tasks": ["performance"],
        "performance": {"warmup": 1, "repeats": 10, "max_tokens": 512},
    },
    "context": {
        "label": "Long Context Test",
        "description": "重点测试不同上下文长度下的性能（Context Scaling）。"
                       "阶梯越长越慢：真机实测 8K 输入单档 TTFT 已 ~20s，"
                       "128K 以上有把服务打挂的风险。",
        "tasks": ["context"],
        # each ladder step is expensive, so 1 sample and no warmup by default;
        # raise `repeats` when you need percentiles per point
        "context": {"warmup": 0, "repeats": 1, "max_tokens": 16},
    },
    "concurrency": {
        "label": "Concurrency Test",
        "description": "重点测试多用户并发下的吞吐与延迟劣化。"
                       "注意：超过服务端槽位数的档位测到的是排队，不是并行能力。",
        "tasks": ["concurrency"],
        # 12 requests per level gives a usable P95; raise it for P99
        "concurrency": {"warmup": 1, "max_tokens": 128,
                        "concurrency_ladder": "1,2,4,8,16,32",
                        "requests_per_level": 12},
    },
    "full": {
        "label": "Full Benchmark",
        "description": "完整运行性能 / 上下文 / 并发测试。",
        "tasks": ["performance", "context", "concurrency"],
        "performance": {"warmup": 1, "repeats": 5, "max_tokens": 256},
        "context": {"warmup": 0, "repeats": 1, "max_tokens": 16},
        "concurrency": {"warmup": 1, "max_tokens": 128,
                        "requests_per_level": 12},
    },
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def resolve_timeout(value: float | None) -> float:
    if value is None:
        return DEFAULT_TIMEOUT_S
    return clamp(float(value), 1.0, HARD_MAX_TIMEOUT_S)


def resolve_retries(value: int | None) -> int:
    if value is None:
        return DEFAULT_RETRIES
    return int(clamp(float(value), 0, 5))
