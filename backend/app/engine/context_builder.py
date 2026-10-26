"""Build an input of a *precise* token length.

The Context benchmark is only meaningful if "32K" really means 32K tokens.
Estimating from character counts is off by a wide margin across languages and
tokenizers, so whenever the engine exposes ``/tokenize`` we measure, and we
use ``/detokenize`` to trim to an exact count.

Three levels of fidelity, reported honestly in ``source``:

  ``tokenize``     measured with ``/tokenize``; ``actual_tokens`` is exact
  ``calibrated``   no tokenizer, but the ratio was **measured on this engine**
                   from two probe requests, so the error is small
  ``estimated``    no tokenizer and no calibration; a crude default ratio

The caller always records both the *target* and the *actual* token count, so a
report never has to pretend the two are equal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Rough characters-per-token used only when the engine cannot tokenize and
# calibration was not possible either.  English technical prose lands near 4;
# this is deliberately crude and is always flagged as an estimate.
_CHARS_PER_TOKEN_FALLBACK = 4.0

_FILLER = (
    "Retrieval-augmented generation combines a parametric language model with "
    "a non-parametric memory, so the model can ground its answers in documents "
    "it was never trained on. The retriever encodes the query and the corpus "
    "into a shared vector space and returns the nearest passages; the generator "
    "then conditions on both the query and those passages. Because the evidence "
    "is supplied at inference time, updating the corpus does not require any "
    "change to the weights, which makes the approach attractive whenever facts "
    "change faster than a training cycle. The dominant costs are embedding the "
    "corpus, storing and querying the index, and the extra prompt tokens that "
    "the retrieved passages add to every single request.\n\n"
)

# A small varied corpus, so long inputs are not one paragraph repeated.
_CORPUS: tuple[str, ...] = (
    _FILLER,
    (
        "Attention is a weighted average over value vectors, where the weights "
        "come from a compatibility score between a query and a set of keys. "
        "Because the computation is permutation invariant, positional "
        "information has to be injected explicitly, either by adding a "
        "sinusoidal signal, by learning relative offsets, or by rotating query "
        "and key vectors before the dot product. Long-context models usually "
        "combine several of these tricks and then rescale the attention logits "
        "so that the softmax does not collapse as the sequence grows.\n\n"
    ),
    (
        "A key-value cache trades memory for latency. During autoregressive "
        "decoding each new token attends to every previous one, so without a "
        "cache the cost of every step would grow with the sequence length. "
        "Storing the keys and values once turns that into a constant amount of "
        "work per step, at the price of memory that scales linearly with the "
        "context. In practice the cache, not the weights, is what limits how "
        "many concurrent long requests a server can hold.\n\n"
    ),
    (
        "Speculative decoding separates proposal from verification. A small "
        "draft model guesses several tokens ahead; the target model then scores "
        "those guesses in a single forward pass and keeps the longest prefix "
        "that matches its own distribution. The output is identical to what the "
        "target model would have produced alone, because every accepted token "
        "is verified under the target distribution. The speed-up depends "
        "entirely on the acceptance rate, which is why the draft model has to "
        "be well aligned with the target.\n\n"
    ),
    (
        "Batch scheduling decides how much throughput a serving engine can "
        "extract. Continuous batching admits new requests between decode steps "
        "instead of waiting for a whole batch to finish, which keeps the device "
        "busy but makes the memory footprint dynamic. Paged attention addresses "
        "the resulting fragmentation by allocating the cache in fixed-size "
        "blocks, much like virtual memory does for an operating system.\n\n"
    ),
    (
        "Quantisation shrinks weights to fewer bits so a model fits in a "
        "smaller memory budget. Post-training schemes pick a scale per group of "
        "weights and round to the nearest level, optionally keeping a small set "
        "of outliers at higher precision. The accuracy cost is usually small at "
        "four bits and grows quickly below that, and the throughput benefit is "
        "mostly about memory bandwidth rather than arithmetic.\n\n"
    ),
)


@dataclass
class BuiltInput:
    """A prompt built to hit ``target_tokens`` as closely as possible."""

    text: str
    target_tokens: int
    actual_tokens: int | None
    source: str                      # tokenize | calibrated | estimated
    units_used: int = 0
    notes: list[str] = field(default_factory=list)
    # True when the engine *claims* to support /tokenize but the call failed.
    # That is a transport symptom, not a missing feature, and the caller uses
    # it to decide whether the engine just went away.
    tokenize_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "actual_tokens": self.actual_tokens,
            "source": self.source,
            "units_used": self.units_used,
            "deviation_pct": (
                round((self.actual_tokens - self.target_tokens)
                      / max(self.target_tokens, 1) * 100, 3)
                if self.actual_tokens is not None else None
            ),
            "notes": self.notes,
            "tokenize_failed": self.tokenize_failed,
        }


def _tolerance(target: int) -> int:
    """Allowed absolute error. Tighter for small targets."""
    return max(2, int(target * 0.01))


# ---------------------------------------------------------------------- #
# calibration: measuring the tokenizer we cannot call
# ---------------------------------------------------------------------- #
@dataclass
class Calibration:
    """The engine's text->token ratio, measured rather than assumed."""

    chars_per_token: float
    overhead_tokens: float
    observations: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chars_per_token": round(self.chars_per_token, 4),
            "overhead_tokens": round(self.overhead_tokens, 2),
            "observations": [{"chars": c, "prompt_tokens": t}
                             for c, t in self.observations],
        }


def calibrate(observations: list[tuple[int, int]],
              default_ratio: float = _CHARS_PER_TOKEN_FALLBACK) -> Calibration:
    """Turn ``(chars, prompt_tokens)`` pairs into a usable ratio.

    A prompt is not just ``chars / ratio``: the chat template adds a roughly
    fixed overhead, which matters a lot at 512 tokens and hardly at all at
    64K.  Two observations of different length are enough to solve both
    unknowns of ``P = overhead + chars / ratio``, so that is what we ask for.
    With fewer (or degenerate) observations we fall back to the default ratio
    and zero overhead - the old, cruder behaviour - rather than pretend.
    """
    pts = sorted({c: t for c, t in observations if c > 0 and t > 0}.items())
    if len(pts) < 2:
        return Calibration(default_ratio, 0.0, list(pts))
    (c1, t1), (c2, t2) = pts[0], pts[-1]
    if c2 == c1 or t2 == t1:
        return Calibration(default_ratio, 0.0, list(pts))
    tokens_per_char = (t2 - t1) / (c2 - c1)
    if tokens_per_char <= 0:
        return Calibration(default_ratio, 0.0, list(pts))
    ratio = 1.0 / tokens_per_char
    overhead = max(0.0, t1 - c1 * tokens_per_char)
    # A ratio outside this band means the observations do not describe a
    # tokenizer at all (an engine echoing a fixed count, say) - refuse to
    # claim we learned something.
    if not (1.0 <= ratio <= 12.0):
        return Calibration(default_ratio, 0.0, list(pts))
    return Calibration(ratio, overhead, list(pts))


def _build_by_chars(target_tokens: int, units: list[str],
                    calibration: Calibration | None,
                    prefix: str = "", prefix_tokens: int = 0) -> BuiltInput:
    """Pad to a character count derived from the measured ratio.

    Used when the engine has no tokenizer.  The achieved token count is
    unknown until the request comes back, so ``actual_tokens`` stays ``None``
    here and the report reads the server's ``usage`` instead.
    """
    calibrated = calibration is not None
    ratio = calibration.chars_per_token if calibration else _CHARS_PER_TOKEN_FALLBACK
    overhead = calibration.overhead_tokens if calibration else 0.0

    body_tokens = max(1.0, target_tokens - overhead - prefix_tokens)
    need = int(body_tokens * ratio)
    parts: list[str] = []
    total = 0
    i = 0
    while total < need:
        part = units[i % len(units)]
        parts.append(part)
        total += len(part)
        i += 1
    text = prefix + "".join(parts)[:need]

    if calibrated:
        notes = [
            f"引擎未提供 /tokenize；长度按**实测标定**换算"
            f"（{ratio:.2f} 字符/token，模板开销约 {overhead:.0f} tokens），"
            "实际 token 数以服务端 usage 为准"
        ]
    else:
        notes = [
            "引擎未提供 /tokenize，输入长度按字符数估算"
            f"（约 {ratio:.1f} 字符/token），实际 token 数以服务端 usage 为准",
        ]
    return BuiltInput(text=text, target_tokens=target_tokens,
                      actual_tokens=None,
                      source="calibrated" if calibrated else "estimated",
                      units_used=i, notes=notes)


async def build_input(
    provider: Any,
    target_tokens: int,
    *,
    strategy: str = "repeat_padding",
    max_iterations: int = 6,
    calibration: Calibration | None = None,
    model: str | None = None,
    prefix: str = "",
) -> BuiltInput:
    """Build a prompt of approximately ``target_tokens`` tokens.

    ``repeat_padding`` repeats one paragraph; ``corpus`` cycles a small set of
    paragraphs so the text stays varied at long lengths.

    ``model`` is passed through to the tokenizer because some engines require
    it (vLLM's ``/tokenize`` answers only when told which model to use).
    Omitting it silently degraded this whole module to character estimation on
    a real vLLM node - the capability probe said "tokenize available" while
    every input was built from a hard-coded 4.0 chars/token, landing 25% short
    of the requested length.

    ``prefix`` counts towards ``target_tokens``: callers put their uniqueness
    marker in it, and a target that ignores the marker would be systematically
    short by the marker's length.
    """
    units = list(_CORPUS) if strategy == "corpus" else [_FILLER]

    prefix_ids = await provider.tokenize(prefix, model) if prefix else None
    prefix_tokens = len(prefix_ids or [])

    probe_ids = await provider.tokenize(units[0], model)
    if not probe_ids:
        built = _build_by_chars(target_tokens, units, calibration,
                                prefix=prefix, prefix_tokens=prefix_tokens)
        supports = getattr(provider, "supports", None)
        if callable(supports) and supports("tokenize"):
            # The engine advertised /tokenize, so an empty result is a
            # transport symptom - say so instead of quietly degrading.
            built.tokenize_failed = True
            built.notes.append(
                "引擎声明支持 /tokenize，但本次调用未返回结果（服务可能中断），"
                "该档输入长度只能按字符估算"
            )
        return built

    body_target = max(1, target_tokens - prefix_tokens)
    per_unit = len(probe_ids)
    reps = max(1, math.ceil(body_target / per_unit))

    def assemble(r: int) -> str:
        return prefix + "".join(units[i % len(units)] for i in range(r))

    text = assemble(reps)
    notes: list[str] = []
    #: (abs_error, text, tokens, reps) of the closest attempt so far
    best: tuple[int, str, int, int] | None = None
    seen: set[int] = set()
    granularity_note = False

    for _ in range(max_iterations):
        ids = await provider.tokenize(text, model)
        if not ids:
            break
        n = len(ids)
        if abs(n - target_tokens) <= _tolerance(target_tokens):
            return BuiltInput(text=text, target_tokens=target_tokens,
                              actual_tokens=n, source="tokenize",
                              units_used=reps, notes=notes)
        if best is None or abs(n - target_tokens) < best[0]:
            best = (abs(n - target_tokens), text, n, reps)

        # Without /detokenize a trim can only add or drop a whole unit, and one
        # unit is usually far longer than the 1% tolerance.  Adding then
        # removing then adding again would loop until the iteration budget ran
        # out and then return whichever attempt came *last* - which can be the
        # worse of the two.  Stop as soon as a length repeats and keep the
        # closest attempt instead.
        if n in seen:
            notes.append(
                "引擎未提供 /detokenize，长度只能按整段增减；"
                "最小单元比目标精度粗，已在可达范围内取最接近的值"
            )
            break
        seen.add(n)

        if n > target_tokens:
            # trim exactly, using the inverse of the tokenizer when available.
            # The prefix is part of `ids`, so the trimmed text keeps it.
            trimmed = await provider.detokenize(ids[:target_tokens], model)
            if trimmed:
                text = trimmed
                reps = max(1, reps)
                continue
            if not granularity_note:
                notes.append("引擎未提供 /detokenize，只能按整段增减，精度受段长限制")
                granularity_note = True
            if reps <= 1:
                notes.append("最小单元已超过目标长度，实际输入偏大")
                break
            reps -= 1
            text = assemble(reps)
        else:
            reps += 1
            text = assemble(reps)

    if best is not None:
        return BuiltInput(text=best[1], target_tokens=target_tokens,
                          actual_tokens=best[2], source="tokenize",
                          units_used=best[3], notes=notes)

    ids = await provider.tokenize(text, model)
    actual = len(ids) if ids else None
    return BuiltInput(text=text, target_tokens=target_tokens,
                      actual_tokens=actual, source="tokenize",
                      units_used=reps, notes=notes)
