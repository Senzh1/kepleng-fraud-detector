"""Random id discovery.

The filtering behaviour carries the weight here. Sampling that happily stored
dormant accounts would fill the training set with empty shops, which is the
same failure as a seed list full of megabrands: a dataset that describes
something other than selling behaviour.
"""

from __future__ import annotations

import random
import re

import pytest

from shopee_scraper import sample, store
from shopee_scraper.models import FetchFailure, FetchStatus

SHOP_ID_RE = re.compile(r"shopid=(\d+)")


class FakeClient:
    """Answers shop-detail requests from a dict, recording what was asked."""

    def __init__(self, shops: dict[int, dict | Exception]) -> None:
        self.shops = shops
        self.asked: list[int] = []

    def get_json(self, url: str) -> dict:
        match = SHOP_ID_RE.search(url)
        assert match, f"expected a shopid in {url!r}"
        shop_id = int(match.group(1))
        self.asked.append(shop_id)

        payload = self.shops.get(shop_id)
        if payload is None:
            raise FetchFailure(FetchStatus.NOT_FOUND, "shopee error 4")
        if isinstance(payload, Exception):
            raise payload
        return payload


def _shop(shop_id: int, name: str, item_count: int | None) -> dict:
    data: dict = {"shopid": shop_id, "name": name, "username": name}
    if item_count is not None:
        data["item_count"] = item_count
    return {"data": data}


@pytest.fixture
def conn(tmp_path):
    connection = store.connect(tmp_path / "shops.db")
    yield connection
    connection.close()


def _single_band(ids: list[int]) -> tuple[tuple[int, int], ...]:
    """A band tight enough that sampling must draw from `ids`."""
    return ((min(ids), max(ids)),)


# --- candidate generation -------------------------------------------------


def test_candidates_stay_inside_their_bands() -> None:
    bands = ((100, 199), (5000, 5099))
    stream = sample.iter_candidate_ids(random.Random(0), bands)
    drawn = [next(stream) for _ in range(40)]

    assert all(100 <= n <= 199 or 5000 <= n <= 5099 for n in drawn)


def test_bands_are_cycled_so_a_short_run_spans_the_id_space() -> None:
    bands = ((100, 199), (5000, 5099), (90_000, 90_099))
    stream = sample.iter_candidate_ids(random.Random(1), bands)
    first_three = [next(stream) for _ in range(3)]

    assert first_three[0] < 200
    assert 5000 <= first_three[1] <= 5099
    assert first_three[2] >= 90_000


def test_a_candidate_is_never_offered_twice() -> None:
    # A band of four ids forces collisions almost immediately.
    stream = sample.iter_candidate_ids(random.Random(7), ((10, 13),))
    drawn = [next(stream) for _ in range(4)]

    assert sorted(drawn) == [10, 11, 12, 13]


def test_already_known_shops_are_not_resampled() -> None:
    stream = sample.iter_candidate_ids(random.Random(3), ((10, 13),), seen={10, 11})
    drawn = [next(stream) for _ in range(2)]

    assert sorted(drawn) == [12, 13]


def test_sampling_without_bands_is_rejected() -> None:
    with pytest.raises(ValueError):
        next(sample.iter_candidate_ids(random.Random(0), ()))


# --- activity filter ------------------------------------------------------


def test_activity_reads_the_listing_count() -> None:
    assert sample.shop_activity(_shop(1, "toko", 12)) == 12


def test_an_unreported_count_is_none_not_zero() -> None:
    """"Shopee did not say" is not evidence the shop is empty."""
    assert sample.shop_activity(_shop(1, "toko", None)) is None


def test_a_boolean_is_not_a_listing_count() -> None:
    assert sample.shop_activity({"data": {"item_count": True}}) is None


# --- discovery ------------------------------------------------------------


def test_active_shops_are_stored_and_dormant_ones_skipped(conn) -> None:
    ids = [10, 11, 12, 13]
    client = FakeClient(
        {
            10: _shop(10, "aktif_satu", 12),
            11: _shop(11, "kosong", 0),
            12: _shop(12, "aktif_dua", 3),
            # 13 is absent: a dead id.
        }
    )

    # Target deliberately unreachable so every id in the band is drawn,
    # regardless of the order the rng happens to pick.
    stats = sample.discover(
        client,
        conn,
        target=3,
        rng=random.Random(0),
        bands=_single_band(ids),
        max_candidates=len(ids),
    )

    assert stats.stored == 2
    assert stats.inactive == 1
    assert stats.errors == 0

    stored = {
        row["shop_id"]: row["username"]
        for row in conn.execute("SELECT shop_id, username FROM shops")
    }
    assert stored == {10: "aktif_satu", 12: "aktif_dua"}


def test_a_stored_shop_is_logged_as_collected(conn) -> None:
    client = FakeClient({10: _shop(10, "aktif", 5)})
    sample.discover(client, conn, target=1, rng=random.Random(0), bands=((10, 10),))

    row = conn.execute("SELECT * FROM fetch_log").fetchone()
    assert row["seed_key"] == "10"
    assert row["status"] == FetchStatus.OK.value


def test_dead_ids_do_not_pollute_the_retry_list(conn) -> None:
    """Thousands of random misses would swamp the operator's failure list."""
    client = FakeClient({12: _shop(12, "aktif", 5)})

    stats = sample.discover(
        client, conn, target=1, rng=random.Random(0), bands=((10, 13),)
    )

    assert stats.stored == 1
    assert store.failures(conn) == []


def test_a_real_error_is_counted_but_does_not_stop_the_run(conn) -> None:
    client = FakeClient(
        {
            10: FetchFailure(FetchStatus.ERROR, "unexpected JSON shape"),
            11: _shop(11, "aktif", 5),
        }
    )

    stats = sample.discover(
        client,
        conn,
        target=2,
        rng=random.Random(0),
        bands=((10, 11),),
        max_candidates=2,
    )

    assert stats.errors == 1
    assert stats.stored == 1


def test_shops_already_in_the_database_are_never_requested(conn) -> None:
    client = FakeClient({10: _shop(10, "aktif", 5), 11: _shop(11, "baru", 5)})
    sample.discover(client, conn, target=1, rng=random.Random(0), bands=((10, 10),))

    second = FakeClient({10: _shop(10, "aktif", 5), 11: _shop(11, "baru", 5)})
    sample.discover(second, conn, target=1, rng=random.Random(0), bands=((10, 11),))

    assert 10 not in second.asked


def test_the_candidate_budget_bounds_a_fruitless_run(conn) -> None:
    """Without a cap, an unreachable target would sample forever."""
    client = FakeClient({})

    stats = sample.discover(
        client,
        conn,
        target=5,
        rng=random.Random(0),
        bands=((1, 10_000_000),),
        max_candidates=25,
    )

    assert stats.candidates == 25
    assert stats.stored == 0
    assert stats.resolve_rate == 0.0


def test_min_items_raises_the_bar(conn) -> None:
    client = FakeClient({10: _shop(10, "kecil", 2), 11: _shop(11, "besar", 40)})

    stats = sample.discover(
        client,
        conn,
        target=2,
        rng=random.Random(0),
        bands=((10, 11),),
        min_items=10,
        max_candidates=10,
    )

    assert stats.stored == 1
    assert conn.execute("SELECT shop_id FROM shops").fetchone()["shop_id"] == 11


def test_the_same_seed_draws_the_same_shops(conn, tmp_path) -> None:
    """Reproducibility: a judge re-running the collection gets our sample."""
    shops = {n: _shop(n, f"toko{n}", 5) for n in range(100, 200)}

    first = FakeClient(dict(shops))
    sample.discover(first, conn, target=5, rng=random.Random(42), bands=((100, 199),))

    other = store.connect(tmp_path / "second.db")
    second = FakeClient(dict(shops))
    sample.discover(second, other, target=5, rng=random.Random(42), bands=((100, 199),))
    other.close()

    assert first.asked == second.asked


def test_a_nonpositive_target_is_rejected(conn) -> None:
    with pytest.raises(ValueError):
        sample.discover(FakeClient({}), conn, target=0)


# --- not paying twice for the same verdict --------------------------------


class InterruptingClient:
    """Stops mid-run the way an operator pressing Ctrl-C does."""

    def __init__(self, shops: dict[int, dict], stop_after: int) -> None:
        self.inner = FakeClient(shops)
        self.stop_after = stop_after

    @property
    def asked(self) -> list[int]:
        return self.inner.asked

    def get_json(self, url: str) -> dict:
        if len(self.inner.asked) >= self.stop_after:
            raise KeyboardInterrupt
        return self.inner.get_json(url)


def test_a_dead_id_is_remembered_rather_than_re_requested(conn) -> None:
    """The restart cost the operator noticed: dead ids re-tested every run."""
    client = FakeClient({12: _shop(12, "aktif", 5)})
    sample.discover(
        client, conn, target=99, rng=random.Random(0), bands=((10, 13),),
        max_candidates=4,
    )
    assert sorted(client.asked) == [10, 11, 12, 13]

    second = FakeClient({12: _shop(12, "aktif", 5)})
    sample.discover(
        second, conn, target=99, rng=random.Random(0), bands=((10, 13),),
        max_candidates=4,
    )

    assert second.asked == []


def test_a_dormant_shop_is_remembered_with_the_count_that_rejected_it(conn) -> None:
    client = FakeClient({10: _shop(10, "kosong", 0), 11: _shop(11, "aktif", 5)})
    sample.discover(
        client, conn, target=99, rng=random.Random(0), bands=((10, 11),),
        max_candidates=2,
    )

    row = conn.execute(
        "SELECT outcome, item_count FROM sampled_ids WHERE shop_id = 10"
    ).fetchone()
    assert row["outcome"] == "inactive"
    assert row["item_count"] == 0


def test_an_errored_id_stays_eligible_for_a_retry(conn) -> None:
    """An id that failed transiently was never judged, so it is not cached."""
    client = FakeClient({10: FetchFailure(FetchStatus.ERROR, "unexpected JSON")})
    sample.discover(
        client, conn, target=99, rng=random.Random(0), bands=((10, 10),),
        max_candidates=1,
    )

    assert store.skippable_ids(conn, min_items=1) == set()


def test_remembering_rejects_does_not_pollute_the_retry_list(conn) -> None:
    client = FakeClient({12: _shop(12, "aktif", 5)})
    sample.discover(
        client, conn, target=99, rng=random.Random(0), bands=((10, 13),),
        max_candidates=4,
    )

    assert store.failures(conn) == []
    assert store.status_counts(conn) == {"ok": 1}


# --- runs are measurable --------------------------------------------------


def test_a_run_records_what_the_sampling_cost(conn) -> None:
    client = FakeClient({10: _shop(10, "aktif", 5), 11: _shop(11, "kosong", 0)})
    sample.discover(
        client, conn, target=99, rng=random.Random(0), bands=((10, 13),),
        max_candidates=4,
    )

    run = store.latest_run(conn)
    assert run is not None
    assert (run["candidates"], run["resolved"], run["stored"], run["inactive"]) == (
        4,
        2,
        1,
        1,
    )


def test_an_interrupted_run_still_records_what_it_cost(conn) -> None:
    """Most runs end by hand; losing their numbers loses the measured rate."""
    shops = {n: _shop(n, f"toko{n}", 5) for n in range(100, 200)}
    client = InterruptingClient(shops, stop_after=3)

    with pytest.raises(KeyboardInterrupt):
        sample.discover(
            client, conn, target=50, rng=random.Random(0), bands=((100, 199),)
        )

    run = store.latest_run(conn)
    assert run is not None
    assert run["stored"] == 3
    assert run["finished_at"] is not None
