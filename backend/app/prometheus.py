"""Prometheus text exposition parser.

The Phase 1 parser was a naive ``line.rsplit(" ", 1)``.  That is wrong for
real engines: vLLM emits nearly every series with labels
(``vllm:num_requests_running{model_name="..."} 3.0``), so the metric name
came out glued to its labels and lookups always missed.

This module implements the subset of the exposition format that inference
engines actually use:

    # HELP / # TYPE comments
    name value
    name{label="v",label2="v2"} value
    name{...} value timestamp
    histogram series:  name_bucket{le="..."} / name_sum / name_count
    summary series:    name{quantile="0.5"} / name_sum / name_count

Label values are unescaped per spec (``\\\\`` ``\\"`` ``\\n``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Sample", "parse_exposition", "collapse", "histogram_mean",
           "info_metric", "pick", "as_ratio"]

# name{labels} value [timestamp]
_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'      # metric name
    r'(?:\{(.*)\})?'                     # optional label block
    r'[ \t]+([^\s]+)'                    # value
    r'(?:[ \t]+\S+)?$'                   # optional timestamp
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    @property
    def key(self) -> str:
        """Identity including labels, useful for de-duplication."""
        if not self.labels:
            return self.name
        inner = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        return f"{self.name}{{{inner}}}"


def _unescape(value: str) -> str:
    return (value
            .replace("\\\\", "\x00")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\x00", "\\"))


def _parse_labels(block: str) -> dict[str, str]:
    return {m.group(1): _unescape(m.group(2)) for m in _LABEL_RE.finditer(block)}


def parse_exposition(text: str) -> tuple[list[Sample], dict[str, str]]:
    """Parse exposition text into ``(samples, metric_types)``.

    ``metric_types`` maps a base metric name to ``counter`` / ``gauge`` /
    ``histogram`` / ``summary`` as declared by ``# TYPE``.
    """
    samples: list[Sample] = []
    types: dict[str, str] = {}

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                types[parts[2]] = parts[3].strip()
            continue

        m = _LINE_RE.match(line)
        if not m:
            continue
        name, label_block, raw_value = m.group(1), m.group(2), m.group(3)
        try:
            value = float(raw_value)
        except ValueError:
            # NaN / +Inf / -Inf are legal in exposition format
            lowered = raw_value.lower()
            if lowered in ("nan",):
                value = float("nan")
            elif lowered in ("+inf", "inf"):
                value = float("inf")
            elif lowered == "-inf":
                value = float("-inf")
            else:
                continue

        samples.append(Sample(name, _parse_labels(label_block or ""), value))

    return samples, types


def collapse(samples: list[Sample]) -> dict[str, float]:
    """Fold labels away by **summing** per metric name.

    Summing is the right default here: engine metrics are either counters
    (``*_total`` - summing across label sets is exactly the total) or
    per-model gauges where the aggregate is what an operator wants to see.
    """
    out: dict[str, float] = {}
    for s in samples:
        out[s.name] = out.get(s.name, 0.0) + s.value
    return out


def histogram_mean(samples: dict[str, float], base: str) -> float | None:
    """Mean of a histogram from its ``_sum`` / ``_count`` pair."""
    total = samples.get(f"{base}_sum")
    count = samples.get(f"{base}_count")
    if not count:
        return None
    return total / count if total is not None else None


def pick(samples: dict[str, float], *names: str) -> float | None:
    """First present value among ``names``."""
    for n in names:
        if n in samples:
            return samples[n]
    return None


def info_metric(parsed: list[Sample], name: str) -> dict[str, str]:
    """Labels of an ``*_info`` metric, merged across label sets.

    ``*_info`` metrics carry their payload in the *labels* and a constant
    ``1.0`` as the value, so they are invisible to :func:`collapse`.  vLLM
    reports its whole cache configuration this way
    (``vllm:cache_config_info{enable_prefix_caching="True",...}``), which is
    the engine stating facts about itself - worth reading verbatim.
    """
    out: dict[str, str] = {}
    for s in parsed:
        if s.name == name:
            out.update(s.labels)
    return out


def as_ratio(value: float | None, *, percent_style: bool = False) -> float | None:
    """Normalise a usage metric to 0..1.

    Some engines report percentages as 0..100, others as 0..1, and the same
    metric has changed convention between releases (vLLM's
    ``gpu_cache_usage_perc`` is the classic offender).  Values above 1.0 are
    therefore treated as percentages.
    """
    if value is None:
        return None
    if percent_style and value > 1.0:
        return value / 100.0
    return value
