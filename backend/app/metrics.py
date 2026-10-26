"""Numeric helpers: percentile / distribution summaries."""

from __future__ import annotations

import math
from typing import Any, Iterable


def percentile(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolated percentile. ``q`` in [0, 1]."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def summarize(values: Iterable[float | int | None]) -> dict[str, Any]:
    """Return mean/median/p50/p95/p99/min/max/std/count for a series."""
    vals = [float(v) for v in values if v is not None and not _isnan(v)]
    if not vals:
        return {
            "count": 0, "mean": None, "median": None, "p50": None,
            "p95": None, "p99": None, "min": None, "max": None, "std": None,
        }
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(percentile(vals, 0.5), 4),
        "p50": round(percentile(vals, 0.5), 4),
        "p95": round(percentile(vals, 0.95), 4),
        "p99": round(percentile(vals, 0.99), 4),
        "min": round(vals[0], 4),
        "max": round(vals[-1], 4),
        "std": round(std, 4),
    }


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True
