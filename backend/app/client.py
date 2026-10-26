"""Unified async HTTP client used by every provider adapter.

Phase 2 extraction: all transport concerns live here so adapters only deal
with protocol shape.

Responsibilities
  * one pooled ``httpx.AsyncClient`` per provider instance
  * explicit timeout budget (connect vs total)
  * a single retry policy with exponential backoff
  * streaming line iteration that retries *only* before the first byte
  * redaction so an API key can never reach a log or an error message

Deliberate default: ``trust_env=False``.

Environment proxies break plain-HTTP requests to LAN inference servers
(measured on this machine: ``/v1/models`` and ``/metrics`` return 404 through
the proxy while ``/health`` intermittently returns 200).  A benchmark client
must measure the server, not a proxy.  Set ``LLMB_TRUST_ENV_PROXY=1`` to
opt back in when a real corporate proxy is required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .errors import ApiError

# Statuses worth retrying: transient upstream / rate limiting, never 4xx client bugs.
RETRY_STATUSES: tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)
# Transport failures worth retrying.
RETRY_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)


def redact(secret: str | None) -> str:
    """Render a credential safely for logs and error messages."""
    if not secret:
        return "(none)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


@dataclass
class RetryPolicy:
    """Retry budget for a single logical request."""

    attempts: int = 3                 # total attempts; 1 disables retrying
    base_delay_s: float = 0.5
    max_delay_s: float = 4.0
    jitter: bool = False
    statuses: tuple[int, ...] = RETRY_STATUSES

    def delay_for(self, attempt: int) -> float:
        """``attempt`` is 0-based."""
        import random

        raw = min(self.base_delay_s * (2 ** attempt), self.max_delay_s)
        return raw * (0.5 + random.random() * 0.5) if self.jitter else raw

    @classmethod
    def from_retries(cls, retries: int) -> "RetryPolicy":
        """``retries`` = extra attempts after the first one."""
        return cls(attempts=max(1, int(retries) + 1))


class ApiClient:
    """Thin, pooled, retrying JSON/stream client."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 15.0,
        retry: RetryPolicy | None = None,
        trust_env: bool = False,
        max_connections: int = 512,
        max_keepalive: int = 128,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.retry = retry or RetryPolicy()
        self.trust_env = trust_env
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s,
                                      connect=self.connect_timeout_s),
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive,
                ),
                trust_env=self.trust_env,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> "ApiClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # headers
    # ------------------------------------------------------------------ #
    def headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        out = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            out["Authorization"] = f"Bearer {self._api_key}"
        if extra:
            out.update(extra)
        return out

    def _timeout(self, override: float | None) -> httpx.Timeout:
        if override is None:
            return httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s)
        return httpx.Timeout(float(override), connect=self.connect_timeout_s)

    @staticmethod
    def _classify(status: int, path: str, body: str) -> ApiError:
        retryable = status in RETRY_STATUSES
        preview = body.replace("\n", " ")[:300]
        return ApiError(f"HTTP {status}: {preview}", status=status,
                        retryable=retryable, path=path)

    # ------------------------------------------------------------------ #
    # capability probing
    # ------------------------------------------------------------------ #
    async def probe(self, path: str, *, method: str = "GET",
                    body: Any = None,
                    timeout_s: float | None = None) -> tuple[int, str]:
        """Never raises. Returns ``(status, body)``; status ``-1`` = transport failure.

        Used by engine detection and capability probing, where a 404 or a 405
        is *information* rather than an error.

        Note: OPTIONS is deliberately not used for route discovery - llama.cpp
        answers ``200`` to ``OPTIONS`` on **every** path, including ones that
        do not exist, so it cannot discriminate.
        """
        try:
            resp = await self.client.request(
                method, path, json=body, timeout=self._timeout(timeout_s))
            return resp.status_code, resp.text
        except Exception as exc:  # noqa: BLE001 - probing must not raise
            return -1, f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ #
    # JSON
    # ------------------------------------------------------------------ #
    async def json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        last: ApiError | None = None
        for attempt in range(self.retry.attempts):
            try:
                resp = await self.client.request(
                    method, path, params=params, json=body,
                    headers=self.headers(headers), timeout=self._timeout(timeout_s),
                )
                if resp.status_code >= 400:
                    raise self._classify(resp.status_code, path, resp.text)
                try:
                    return resp.json()
                except ValueError as exc:
                    raise ApiError(f"invalid JSON from {path}: {exc}",
                                   status=resp.status_code, path=path) from exc
            except ApiError as exc:
                last = exc
                if not exc.retryable:
                    raise
            except RETRY_EXCEPTIONS as exc:
                last = ApiError(f"{type(exc).__name__}: {exc}", retryable=True,
                                path=path)
            except httpx.HTTPError as exc:
                raise ApiError(f"{type(exc).__name__}: {exc}", path=path) from exc
            if attempt < self.retry.attempts - 1:
                await asyncio.sleep(self.retry.delay_for(attempt))
        assert last is not None
        raise last

    async def text(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Plain-text GET (used for Prometheus ``/metrics``)."""
        last: ApiError | None = None
        for attempt in range(self.retry.attempts):
            try:
                extra = dict(headers or {})
                extra["Accept"] = "text/plain"
                resp = await self.client.request(
                    method, path, headers=self.headers(extra),
                    timeout=self._timeout(timeout_s),
                )
                if resp.status_code >= 400:
                    raise self._classify(resp.status_code, path, resp.text)
                return resp.text
            except ApiError as exc:
                last = exc
                if not exc.retryable:
                    raise
            except RETRY_EXCEPTIONS as exc:
                last = ApiError(f"{type(exc).__name__}: {exc}", retryable=True,
                                path=path)
            if attempt < self.retry.attempts - 1:
                await asyncio.sleep(self.retry.delay_for(attempt))
        assert last is not None
        raise last

    # ------------------------------------------------------------------ #
    # streaming
    # ------------------------------------------------------------------ #
    async def stream_lines(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield decoded lines from a streaming response.

        Retries are only safe **before** the first line is handed to the
        caller; once bytes have flowed, a retry would double-count tokens, so
        the error is surfaced instead.
        """
        last: ApiError | None = None
        for attempt in range(self.retry.attempts):
            emitted = False
            try:
                async with self.client.stream(
                    method, path, json=body, headers=self.headers(headers),
                    timeout=self._timeout(timeout_s),
                ) as resp:
                    if resp.status_code >= 400:
                        raw = (await resp.aread()).decode("utf-8", "replace")
                        raise self._classify(resp.status_code, path, raw)
                    async for line in resp.aiter_lines():
                        emitted = True
                        yield line
                return
            except ApiError as exc:
                last = exc
                if emitted or not exc.retryable:
                    raise
            except RETRY_EXCEPTIONS as exc:
                last = ApiError(f"{type(exc).__name__}: {exc}",
                                retryable=not emitted, path=path)
                if emitted:
                    raise last from exc
            except httpx.HTTPError as exc:
                raise ApiError(f"{type(exc).__name__}: {exc}", path=path) from exc
            if attempt < self.retry.attempts - 1:
                await asyncio.sleep(self.retry.delay_for(attempt))
        assert last is not None
        raise last
