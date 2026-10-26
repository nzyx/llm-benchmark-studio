"""Shared error types.

Kept in a leaf module so both the HTTP client layer and the provider
adapters can raise the same type without circular imports.
"""

from __future__ import annotations

import re

# The HTTP status often survives only inside the message text: the streaming
# path turns an ApiError into a string before it reaches the measuring code.
_HTTP_RE = re.compile(r"^\s*(?:[A-Za-z_]*Error:\s*)*HTTP\s+(\d{3})\b", re.I)


class ApiError(RuntimeError):
    """Any failure while talking to an upstream HTTP service.

    ``retryable`` tells the client layer whether another attempt makes sense.
    ``url`` is the *path only* - never include a query string that could
    carry a credential.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.path = path


# ---------------------------------------------------------------------- #
# failure classification
# ---------------------------------------------------------------------- #
# A benchmark needs to tell three very different situations apart, because
# they call for opposite responses:
#
#   ``too_long``      the engine *answered* that it cannot do this input
#                     (4xx/5xx, or it closed an oversized stream).  Retrying
#                     is pointless; this is real capacity information.
#   ``unreachable``   the engine is gone (connection refused / no route).
#                     Every later measurement would be fiction, so the run
#                     must stop and say so.
#   ``timeout``       no answer in time.  Inconclusive: could be a very long
#                     prefill or a stalled server.  Retrying is legitimate.
#
# Before this existed, an engine that *crashed* under a 128K prefill was
# indistinguishable from an engine that *rejected* 128K, and the binary search
# happily reported a "Maximum Verified Context" that was really just the
# largest step measured before the crash.
KIND_TOO_LONG = "too_long"
KIND_UNREACHABLE = "unreachable"
KIND_TIMEOUT = "timeout"
KIND_AUTH = "auth"
KIND_SERVER = "server_error"
KIND_UNKNOWN = "unknown"

# Substrings that betray a dead socket rather than an oversized request.
_UNREACHABLE_HINTS = (
    "connecterror", "connecttimeout", "all connection attempts failed",
    "connection refused", "actively refused", "10061", "10060",
    "name or service not known", "nodename nor servname",
    "no route to host", "network is unreachable", "temporary failure in name",
)
_TIMEOUT_HINTS = ("readtimeout", "pooltimeout", "timed out", "timeout")
# The engine accepted the connection and then died mid-answer.  For a *long
# input* that is capacity information (it could not hold the work), but it
# also means the process may have restarted - callers should re-check
# liveness before trusting anything measured afterwards.
_DROPPED_HINTS = (
    "remoteprotocolerror", "peer closed connection", "incomplete chunked read",
    "server disconnected", "connection reset", "broken pipe",
)


def classify_error(exc_or_message: object, status: int | None = None) -> str:
    """Map any exception (or an error string) onto a stable failure kind."""
    if status is None:
        status = getattr(exc_or_message, "status", None)
    if isinstance(exc_or_message, BaseException):
        text = f"{type(exc_or_message).__name__}: {exc_or_message}"
    else:
        text = str(exc_or_message)
    if status is None:
        m = _HTTP_RE.match(text)
        if m:
            status = int(m.group(1))
    low = text.lower()

    # a real HTTP status is the strongest signal we can get
    if isinstance(status, int):
        if status in (401, 403):
            return KIND_AUTH
        if status in (408, 504):
            return KIND_TIMEOUT
        if status == 413:
            return KIND_TOO_LONG
        if 400 <= status < 500:
            return KIND_TOO_LONG
        if status >= 500:
            # deliberately *not* reported as capacity: a 500 during a long
            # prefill usually means the engine ran out of room, but it can
            # equally mean the engine is broken.  The caller's liveness check
            # settles which, so the kind stays neutral.
            return KIND_SERVER

    if any(h in low for h in _UNREACHABLE_HINTS):
        return KIND_UNREACHABLE
    if any(h in low for h in _DROPPED_HINTS):
        return KIND_TOO_LONG
    if any(h in low for h in _TIMEOUT_HINTS):
        return KIND_TIMEOUT
    if "statuscodeerror" in low or "status error" in low:
        return KIND_TOO_LONG
    return KIND_UNKNOWN
