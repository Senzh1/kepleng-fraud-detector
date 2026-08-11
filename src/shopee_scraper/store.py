"""SQLite persistence. All SQL in the project lives here.

Raw API responses are stored alongside everything else so feature definitions
can change without re-scraping. Re-deriving from stored raw takes seconds;
re-scraping takes days and risks the collector's access.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .models import FetchStatus, ItemRecord, ShopRecord, utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS shops (
    shop_id      INTEGER PRIMARY KEY,
    username     TEXT,
    raw_base     TEXT,
    raw_detail   TEXT,
    fetched_at   TEXT NOT NULL,
    label        INTEGER,
    label_source TEXT
);

CREATE TABLE IF NOT EXISTS shop_items (
    shop_id    INTEGER NOT NULL,
    item_id    INTEGER NOT NULL,
    raw        TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (shop_id, item_id)
);

-- Keyed by seed rather than shop_id: a username-only seed has no numeric id
-- until its first successful fetch, and resume has to work for those too.
CREATE TABLE IF NOT EXISTS fetch_log (
    seed_key   TEXT PRIMARY KEY,
    shop_id    INTEGER,
    status     TEXT NOT NULL,
    detail     TEXT,
    attempts   INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_shop ON shop_items (shop_id);
CREATE INDEX IF NOT EXISTS idx_log_status ON fetch_log (status);
"""


def seed_key(shop_id: int | None, username: str | None) -> str:
    """Stable identity for a seed, usable before the shop id is known.

    A username seed keeps its `u:` key for the whole run so that resuming does
    not re-fetch it, while `shop_id` is backfilled once resolved.
    """
    if shop_id is not None:
        return str(shop_id)
    if username:
        return f"u:{username.lower()}"
    raise ValueError("seed has neither shop_id nor username")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, creating it and its schema when absent."""
    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_shop(conn: sqlite3.Connection, record: ShopRecord) -> None:
    """Insert or refresh a shop, preserving any label already attached."""
    conn.execute(
        """
        INSERT INTO shops (shop_id, username, raw_base, raw_detail, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(shop_id) DO UPDATE SET
            username   = COALESCE(excluded.username, shops.username),
            raw_base   = COALESCE(excluded.raw_base, shops.raw_base),
            raw_detail = COALESCE(excluded.raw_detail, shops.raw_detail),
            fetched_at = excluded.fetched_at
        """,
        (
            record.shop_id,
            record.username,
            json.dumps(record.raw_base) if record.raw_base is not None else None,
            json.dumps(record.raw_detail) if record.raw_detail is not None else None,
            record.fetched_at,
        ),
    )
    conn.commit()


def upsert_items(conn: sqlite3.Connection, items: list[ItemRecord]) -> int:
    """Insert or refresh listings. Returns the number written."""
    if not items:
        return 0
    conn.executemany(
        """
        INSERT INTO shop_items (shop_id, item_id, raw, fetched_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(shop_id, item_id) DO UPDATE SET
            raw        = excluded.raw,
            fetched_at = excluded.fetched_at
        """,
        [
            (item.shop_id, item.item_id, json.dumps(item.raw), item.fetched_at)
            for item in items
        ],
    )
    conn.commit()
    return len(items)


def record_status(
    conn: sqlite3.Connection,
    key: str,
    status: FetchStatus,
    detail: str | None = None,
    shop_id: int | None = None,
) -> None:
    """Record the outcome of one seed, incrementing its attempt counter."""
    conn.execute(
        """
        INSERT INTO fetch_log (seed_key, shop_id, status, detail, attempts, updated_at)
        VALUES (?, ?, ?, ?, 1, ?)
        ON CONFLICT(seed_key) DO UPDATE SET
            shop_id    = COALESCE(excluded.shop_id, fetch_log.shop_id),
            status     = excluded.status,
            detail     = excluded.detail,
            attempts   = fetch_log.attempts + 1,
            updated_at = excluded.updated_at
        """,
        (key, shop_id, status.value, detail, utc_now_iso()),
    )
    conn.commit()


def completed_keys(conn: sqlite3.Connection) -> set[str]:
    """Seed keys already collected successfully. These are skipped on re-run."""
    rows = conn.execute(
        "SELECT seed_key FROM fetch_log WHERE status = ?", (FetchStatus.OK.value,)
    ).fetchall()
    return {row["seed_key"] for row in rows}


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Outcome breakdown across every seed attempted so far."""
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM fetch_log GROUP BY status"
    ).fetchall()
    return {row["status"]: row["n"] for row in rows}


def failures(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Seeds that did not succeed, for the operator's retry list."""
    rows = conn.execute(
        """
        SELECT seed_key, shop_id, status, detail, attempts
        FROM fetch_log WHERE status != ? ORDER BY updated_at DESC LIMIT ?
        """,
        (FetchStatus.OK.value, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def import_labels(
    conn: sqlite3.Connection, rows: list[dict[str, str]]
) -> tuple[int, int]:
    """Attach labels to already-collected shops.

    Returns `(applied, unknown)`. A label for a shop that has not been scraped
    is reported rather than silently inserted, so a typo in the label file
    surfaces instead of creating a ghost row.
    """
    applied = 0
    unknown = 0
    for row in rows:
        shop_id = int(row["shop_id"])
        exists = conn.execute(
            "SELECT 1 FROM shops WHERE shop_id = ?", (shop_id,)
        ).fetchone()
        if not exists:
            unknown += 1
            continue
        conn.execute(
            "UPDATE shops SET label = ?, label_source = ? WHERE shop_id = ?",
            (int(row["label"]), row.get("source"), shop_id),
        )
        applied += 1
    conn.commit()
    return applied, unknown


def iter_shops(
    conn: sqlite3.Connection,
) -> Iterator[tuple[ShopRecord, list[ItemRecord]]]:
    """Stream every collected shop with its listings, for feature derivation."""
    shop_rows = conn.execute("SELECT * FROM shops ORDER BY shop_id").fetchall()
    for row in shop_rows:
        record = ShopRecord(
            shop_id=row["shop_id"],
            username=row["username"],
            raw_base=json.loads(row["raw_base"]) if row["raw_base"] else None,
            raw_detail=json.loads(row["raw_detail"]) if row["raw_detail"] else None,
            fetched_at=row["fetched_at"],
        )
        item_rows = conn.execute(
            "SELECT * FROM shop_items WHERE shop_id = ?", (row["shop_id"],)
        ).fetchall()
        items = [
            ItemRecord(
                shop_id=item["shop_id"],
                item_id=item["item_id"],
                raw=json.loads(item["raw"]),
                fetched_at=item["fetched_at"],
            )
            for item in item_rows
        ]
        yield record, items


def shop_labels(conn: sqlite3.Connection) -> dict[int, tuple[int | None, str | None]]:
    """Every shop's label and label source, keyed by shop id."""
    rows = conn.execute("SELECT shop_id, label, label_source FROM shops").fetchall()
    return {row["shop_id"]: (row["label"], row["label_source"]) for row in rows}
