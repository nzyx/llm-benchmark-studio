"""SQLite persistence layer for benchmark runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from .config import DB_PATH

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    task_type     TEXT NOT NULL,
    preset        TEXT,
    label         TEXT,
    engine        TEXT,
    model_id      TEXT,
    base_url      TEXT,
    status        TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    result_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_model   ON runs(model_id);
"""

_LIST_COLUMNS = (
    "run_id, created_at, task_type, preset, label, engine, "
    "model_id, status, schema_version, base_url"
)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with _LOCK, _connect() as conn:
        conn.executescript(_SCHEMA)


def save_run(result: dict[str, Any]) -> None:
    """Upsert a full result document. ``result`` must follow the run schema."""
    run_id = result["run_id"]
    provider = result.get("provider") or {}
    model = result.get("model") or {}
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, created_at, task_type, preset, label,
                              engine, model_id, base_url, status,
                              schema_version, result_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                result_json=excluded.result_json,
                model_id=excluded.model_id,
                engine=excluded.engine
            """,
            (
                run_id,
                result.get("created_at", ""),
                result.get("task_type", "unknown"),
                result.get("preset"),
                result.get("label"),
                provider.get("kind"),
                model.get("id"),
                provider.get("base_url"),
                result.get("status", "unknown"),
                result.get("schema_version", "1.0"),
                json.dumps(result, ensure_ascii=False),
            ),
        )
        conn.commit()


def list_runs(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {_LIST_COLUMNS} FROM runs "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (int(limit), int(offset)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if not row:
        return None
    return json.loads(row["result_json"])


def delete_run(run_id: str) -> bool:
    with _LOCK, _connect() as conn:
        cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()
    return cur.rowcount > 0


def count_runs() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
    return int(row["n"])
