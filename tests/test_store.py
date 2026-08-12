"""Persistence: idempotent upserts, resume state, and label attachment."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from shopee_scraper import store
from shopee_scraper.models import FetchStatus, ItemRecord, SampleOutcome, ShopRecord


@pytest.fixture
def conn(tmp_path: Path):
    connection = store.connect(tmp_path / "nested" / "shops.db")
    yield connection
    connection.close()


def _record(shop_id: int = 123456789, **overrides) -> ShopRecord:
    defaults = {
        "shop_id": shop_id,
        "username": "contoh_toko",
        "raw_base": {"data": {"shopid": shop_id}},
        "raw_detail": None,
        "fetched_at": "2026-08-11T04:22:31Z",
    }
    return ShopRecord(**{**defaults, **overrides})


def test_connect_creates_the_database_and_its_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "shops.db"
    connection = store.connect(path)
    assert path.exists()
    connection.close()


def test_seed_key_prefers_shop_id_and_namespaces_usernames() -> None:
    assert store.seed_key(123456789, None) == "123456789"
    assert store.seed_key(None, "Contoh_Toko") == "u:contoh_toko"
    assert store.seed_key(123456789, "contoh_toko") == "123456789"

    with pytest.raises(ValueError):
        store.seed_key(None, None)


def test_upserting_the_same_shop_twice_keeps_one_row(conn: sqlite3.Connection) -> None:
    store.upsert_shop(conn, _record())
    store.upsert_shop(conn, _record(fetched_at="2026-08-12T00:00:00Z"))

    rows = conn.execute("SELECT * FROM shops").fetchall()
    assert len(rows) == 1
    assert rows[0]["fetched_at"] == "2026-08-12T00:00:00Z"


def test_re_scraping_does_not_wipe_an_existing_label(conn: sqlite3.Connection) -> None:
    store.upsert_shop(conn, _record())
    store.import_labels(
        conn, [{"shop_id": "123456789", "label": "1", "source": "manual"}]
    )

    store.upsert_shop(conn, _record(fetched_at="2026-08-12T00:00:00Z"))

    row = conn.execute("SELECT label, label_source FROM shops").fetchone()
    assert row["label"] == 1
    assert row["label_source"] == "manual"


def test_re_scraping_does_not_null_out_a_previously_seen_payload(
    conn: sqlite3.Connection,
) -> None:
    store.upsert_shop(conn, _record(raw_detail={"data": {"place": "Bandung"}}))
    store.upsert_shop(conn, _record(raw_detail=None))

    row = conn.execute("SELECT raw_detail FROM shops").fetchone()
    assert "Bandung" in row["raw_detail"]


def test_items_upsert_is_idempotent(conn: sqlite3.Connection) -> None:
    items = [
        ItemRecord(
            shop_id=123456789,
            item_id=item_id,
            raw={"itemid": item_id},
            fetched_at="2026-08-11T04:22:31Z",
        )
        for item_id in (111, 222)
    ]
    assert store.upsert_items(conn, items) == 2
    store.upsert_items(conn, items)

    assert conn.execute("SELECT COUNT(*) AS n FROM shop_items").fetchone()["n"] == 2
    assert store.upsert_items(conn, []) == 0


def test_status_records_attempts_and_drives_resume(conn: sqlite3.Connection) -> None:
    store.record_status(conn, "u:contoh_toko", FetchStatus.BLOCKED, "HTTP 429")
    store.record_status(conn, "u:contoh_toko", FetchStatus.BLOCKED, "HTTP 429")

    row = conn.execute("SELECT * FROM fetch_log").fetchone()
    assert row["attempts"] == 2
    assert row["status"] == "blocked"
    assert store.completed_keys(conn) == set()

    store.record_status(conn, "u:contoh_toko", FetchStatus.OK, None, shop_id=123456789)
    assert store.completed_keys(conn) == {"u:contoh_toko"}

    stored = conn.execute("SELECT shop_id FROM fetch_log").fetchone()
    assert stored["shop_id"] == 123456789


def test_status_counts_and_failure_list(conn: sqlite3.Connection) -> None:
    store.record_status(conn, "1", FetchStatus.OK)
    store.record_status(conn, "2", FetchStatus.BLOCKED, "HTTP 429")
    store.record_status(conn, "3", FetchStatus.NOT_FOUND, "HTTP 404")

    assert store.status_counts(conn) == {"ok": 1, "blocked": 1, "not_found": 1}

    problems = store.failures(conn)
    assert {row["seed_key"] for row in problems} == {"2", "3"}


def test_labels_for_unscraped_shops_are_reported_not_inserted(
    conn: sqlite3.Connection,
) -> None:
    store.upsert_shop(conn, _record())

    applied, unknown = store.import_labels(
        conn,
        [
            {"shop_id": "123456789", "label": "1", "source": "manual"},
            {"shop_id": "999999999", "label": "0"},
        ],
    )

    assert (applied, unknown) == (1, 1)
    assert conn.execute("SELECT COUNT(*) AS n FROM shops").fetchone()["n"] == 1


def test_iter_shops_round_trips_records_and_items(conn: sqlite3.Connection) -> None:
    store.upsert_shop(conn, _record(raw_detail={"data": {"name": "Contoh Toko"}}))
    store.upsert_items(
        conn,
        [
            ItemRecord(
                shop_id=123456789,
                item_id=111,
                raw={"itemid": 111, "price": 1000},
                fetched_at="2026-08-11T04:22:31Z",
            )
        ],
    )

    collected = list(store.iter_shops(conn))
    assert len(collected) == 1

    record, items = collected[0]
    assert record.shop_id == 123456789
    assert record.raw_base == {"data": {"shopid": 123456789}}
    assert record.raw_detail == {"data": {"name": "Contoh Toko"}}
    assert [item.raw["price"] for item in items] == [1000]


def test_shop_labels_reports_unlabeled_shops_as_none(conn: sqlite3.Connection) -> None:
    store.upsert_shop(conn, _record())
    assert store.shop_labels(conn) == {123456789: (None, None)}


# --- rejected sample ids --------------------------------------------------


def test_a_rejected_id_is_remembered_for_later_runs(conn: sqlite3.Connection) -> None:
    store.record_sampled(conn, 11, SampleOutcome.NOT_FOUND)
    assert store.skippable_ids(conn, min_items=1) == {11}


def test_rejected_ids_stay_out_of_the_retry_list(conn: sqlite3.Connection) -> None:
    """The whole point of a separate table: dead ids must not look retryable."""
    store.record_sampled(conn, 11, SampleOutcome.NOT_FOUND)
    store.record_sampled(conn, 12, SampleOutcome.INACTIVE, 0)

    assert store.failures(conn) == []
    assert store.status_counts(conn) == {}


def test_recording_an_id_twice_keeps_one_row(conn: sqlite3.Connection) -> None:
    store.record_sampled(conn, 11, SampleOutcome.INACTIVE, 0)
    store.record_sampled(conn, 11, SampleOutcome.INACTIVE, 3)

    assert store.sampled_counts(conn) == {"inactive": 1}
    row = conn.execute("SELECT item_count FROM sampled_ids").fetchone()
    assert row["item_count"] == 3


def test_a_dormant_shop_returns_once_the_bar_is_lowered(
    conn: sqlite3.Connection,
) -> None:
    """Caching a verdict must not outlive the threshold that produced it."""
    store.record_sampled(conn, 11, SampleOutcome.INACTIVE, 4)

    assert store.skippable_ids(conn, min_items=10) == {11}
    assert store.skippable_ids(conn, min_items=4) == set()


def test_a_shop_with_no_reported_count_is_always_skipped(
    conn: sqlite3.Connection,
) -> None:
    """"Did not say" never satisfies a minimum, at any threshold."""
    store.record_sampled(conn, 11, SampleOutcome.INACTIVE, None)
    assert store.skippable_ids(conn, min_items=1) == {11}


def test_a_dead_id_is_dead_at_every_threshold(conn: sqlite3.Connection) -> None:
    store.record_sampled(conn, 11, SampleOutcome.NOT_FOUND)
    assert store.skippable_ids(conn, min_items=0) == {11}


def test_rejected_ids_are_counted_by_reason(conn: sqlite3.Connection) -> None:
    store.record_sampled(conn, 11, SampleOutcome.NOT_FOUND)
    store.record_sampled(conn, 12, SampleOutcome.NOT_FOUND)
    store.record_sampled(conn, 13, SampleOutcome.INACTIVE, 0)

    assert store.sampled_counts(conn) == {"not_found": 2, "inactive": 1}


# --- discovery runs -------------------------------------------------------


def test_a_run_is_recorded_from_the_moment_it_starts(conn: sqlite3.Connection) -> None:
    run_id = store.start_run(conn, target=5, min_items=1)

    run = store.latest_run(conn)
    assert run is not None
    assert run["run_id"] == run_id
    assert run["target"] == 5
    assert run["finished_at"] is None


def test_finishing_a_run_records_what_it_cost(conn: sqlite3.Connection) -> None:
    run_id = store.start_run(conn, target=5, min_items=1)
    store.finish_run(
        conn, run_id, candidates=100, resolved=10, stored=3, inactive=7, errors=1
    )

    run = store.latest_run(conn)
    assert run is not None
    assert (run["candidates"], run["stored"], run["errors"]) == (100, 3, 1)
    assert run["finished_at"] is not None


def test_no_runs_yet_is_none_not_an_error(conn: sqlite3.Connection) -> None:
    assert store.latest_run(conn) is None
    assert store.run_totals(conn) == {
        "runs": 0,
        "candidates": 0,
        "resolved": 0,
        "stored": 0,
        "inactive": 0,
        "errors": 0,
    }


def test_totals_sum_the_effort_across_runs(conn: sqlite3.Connection) -> None:
    """The measured cost per shop has to survive the operator restarting."""
    for stored in (2, 3):
        run_id = store.start_run(conn, target=5, min_items=1)
        store.finish_run(
            conn, run_id, candidates=50, resolved=5, stored=stored, inactive=2, errors=0
        )

    totals = store.run_totals(conn)
    assert totals["runs"] == 2
    assert totals["candidates"] == 100
    assert totals["stored"] == 5
