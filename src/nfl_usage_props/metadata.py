"""SQLite run metadata: ingest runs, credit balances, API errors.

Deliberately small. Parquet holds the data; this holds the *story* of how the
data got there, which is what you need when a Sunday-morning pull silently
half-fails.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name    TEXT    NOT NULL,
    season        INTEGER NOT NULL,
    status        TEXT    NOT NULL,  -- fetched | skipped_complete | error
    rows          INTEGER,
    bytes_written INTEGER,
    duration_s    REAL,
    message       TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ingest_runs_table_season
    ON ingest_runs (table_name, season);

CREATE TABLE IF NOT EXISTS season_state (
    table_name  TEXT    NOT NULL,
    season      INTEGER NOT NULL,
    complete    INTEGER NOT NULL,  -- 1 once the season is immutable
    rows        INTEGER,
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (table_name, season)
);

CREATE TABLE IF NOT EXISTS odds_calls (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint           TEXT    NOT NULL,
    params             TEXT,
    status_code        INTEGER,
    requests_used      INTEGER,   -- x-requests-used header
    requests_remaining INTEGER,   -- x-requests-remaining header
    cost               INTEGER,   -- x-requests-last header
    duration_s         REAL,
    error              TEXT,
    created_at         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_odds_calls_created ON odds_calls (created_at);
"""


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class MetadataStore:
    """Thin wrapper over a SQLite file. Safe to construct repeatedly."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ ingest

    def record_ingest(
        self,
        *,
        table_name: str,
        season: int,
        status: str,
        rows: int | None = None,
        bytes_written: int | None = None,
        duration_s: float | None = None,
        message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ingest_runs "
                "(table_name, season, status, rows, bytes_written, duration_s, message, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    table_name,
                    season,
                    status,
                    rows,
                    bytes_written,
                    duration_s,
                    message,
                    utcnow_iso(),
                ),
            )

    def set_season_state(
        self, *, table_name: str, season: int, complete: bool, rows: int | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO season_state (table_name, season, complete, rows, updated_at) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(table_name, season) DO UPDATE SET "
                "complete=excluded.complete, rows=excluded.rows, updated_at=excluded.updated_at",
                (table_name, season, int(complete), rows, utcnow_iso()),
            )

    def get_season_state(self, table_name: str, season: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM season_state WHERE table_name=? AND season=?",
                (table_name, season),
            ).fetchone()
        return dict(row) if row else None

    def is_season_complete(self, table_name: str, season: int) -> bool:
        state = self.get_season_state(table_name, season)
        return bool(state and state["complete"])

    # -------------------------------------------------------------------- odds

    def record_odds_call(
        self,
        *,
        endpoint: str,
        params: str | None = None,
        status_code: int | None = None,
        requests_used: int | None = None,
        requests_remaining: int | None = None,
        cost: int | None = None,
        duration_s: float | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO odds_calls (endpoint, params, status_code, requests_used, "
                "requests_remaining, cost, duration_s, error, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    endpoint,
                    params,
                    status_code,
                    requests_used,
                    requests_remaining,
                    cost,
                    duration_s,
                    error,
                    utcnow_iso(),
                ),
            )

    def latest_credit_balance(self) -> int | None:
        """Most recent `x-requests-remaining` we saw, if any."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT requests_remaining FROM odds_calls "
                "WHERE requests_remaining IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return int(row["requests_remaining"]) if row else None

    def credits_spent_since(self, since_iso: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost), 0) AS spent FROM odds_calls WHERE created_at >= ?",
                (since_iso,),
            ).fetchone()
        return int(row["spent"])

    def recent_odds_calls(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM odds_calls ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def ingest_summary(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT table_name, season, complete, rows, updated_at FROM season_state "
                "ORDER BY table_name, season"
            ).fetchall()
        return [dict(r) for r in rows]
