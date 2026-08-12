"""End-to-end CLI behaviour with the network faked out.

These cover the paths an operator actually exercises: a first run, a resumed
run, failures that do not abort the batch, labelling, and export.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import load_fixture
from typer.testing import CliRunner

from shopee_scraper import cli, store
from shopee_scraper.client import RunAborted, ShopFetch
from shopee_scraper.models import (
    FetchFailure,
    FetchStatus,
    ItemRecord,
    SampleOutcome,
    ShopRecord,
)

runner = CliRunner()


class FakeClient:
    """Stands in for ShopeeClient. Records calls, returns fixture payloads."""

    instances: list["FakeClient"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.fetched: list[object] = []
        FakeClient.instances.append(self)

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def fetch_shop(self, ref):
        self.fetched.append(ref)
        shop_id = ref.shop_id or 123456789
        items = [
            ItemRecord(
                shop_id=shop_id,
                item_id=item_id,
                raw={"itemid": item_id, "price": 1000000000, "stock": 1},
                fetched_at="2026-08-11T04:22:31Z",
            )
            for item_id in (111, 222)
        ]
        record = ShopRecord(
            shop_id=shop_id,
            username=ref.username,
            raw_base=load_fixture("shop_base.json"),
            raw_detail=load_fixture("shop_detail.json"),
            fetched_at="2026-08-11T04:22:31Z",
        )
        return ShopFetch(record, items, self.item_note)

    item_note: str | None = None


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run in a clean directory so a developer's real .env never leaks in."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHOPEE_RATE", "0")
    monkeypatch.delenv("SHOPEE_COOKIE", raising=False)
    FakeClient.instances = []
    return tmp_path


def _seeds(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "seeds.csv"
    path.write_text("url_or_id\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def _scrape(tmp_path: Path, seeds: Path, *extra: str):
    return runner.invoke(
        cli.app,
        ["scrape", "--seeds", str(seeds), "--db", str(tmp_path / "shops.db"), *extra],
    )


def test_scrape_collects_shops_and_listings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    seeds = _seeds(tmp_path, "https://shopee.co.id/shop/123456789")

    result = _scrape(tmp_path, seeds)

    assert result.exit_code == 0, result.output
    assert "ok, 2 listings" in result.output

    conn = store.connect(tmp_path / "shops.db")
    assert conn.execute("SELECT COUNT(*) AS n FROM shops").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM shop_items").fetchone()["n"] == 2
    assert store.completed_keys(conn) == {"123456789"}
    conn.close()


def test_a_second_run_skips_what_was_already_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    seeds = _seeds(tmp_path, "123456789")

    _scrape(tmp_path, seeds)
    result = _scrape(tmp_path, seeds)

    assert "1 seeds, 1 skipped, 0 to fetch" in result.output
    assert FakeClient.instances[-1].fetched == []


def test_one_failure_does_not_abort_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PartlyFailingClient(FakeClient):
        def fetch_shop(self, ref):
            if ref.shop_id == 999999999:
                raise FetchFailure(FetchStatus.BLOCKED, "HTTP 429")
            return super().fetch_shop(ref)

    monkeypatch.setattr(cli, "ShopeeClient", PartlyFailingClient)
    seeds = _seeds(tmp_path, "123456789", "999999999", "111111111")

    result = _scrape(tmp_path, seeds)

    assert "HTTP 429" in result.output
    assert "ok=2" in result.output and "blocked=1" in result.output

    conn = store.connect(tmp_path / "shops.db")
    assert store.completed_keys(conn) == {"123456789", "111111111"}
    assert [row["seed_key"] for row in store.failures(conn)] == ["999999999"]
    conn.close()


def test_unparseable_seed_lines_are_reported_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    seeds = _seeds(tmp_path, "https://tokopedia.com/shop/1", "123456789")

    result = _scrape(tmp_path, seeds)

    assert "not a Shopee host" in result.output
    assert "ok=1" in result.output


def test_a_seeds_file_with_nothing_usable_exits_nonzero(tmp_path: Path) -> None:
    seeds = _seeds(tmp_path, "https://tokopedia.com/shop/1")
    result = _scrape(tmp_path, seeds)

    assert result.exit_code == 1
    assert "No usable seeds" in result.output


def test_missing_seeds_file_is_a_usage_error(tmp_path: Path) -> None:
    result = _scrape(tmp_path, tmp_path / "absent.csv")
    assert result.exit_code != 0
    assert "seeds file not found" in result.output


def test_seeds_file_without_a_header_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("123456789\n", encoding="utf-8")

    result = _scrape(tmp_path, seeds)
    assert "ok=1" in result.output


def test_status_reports_progress_and_the_retry_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    conn = store.connect(tmp_path / "shops.db")
    store.record_status(conn, "999", FetchStatus.BLOCKED, "HTTP 429")
    conn.close()

    result = runner.invoke(cli.app, ["status", "--db", str(tmp_path / "shops.db")])

    assert result.exit_code == 0
    assert "Shops stored    : 1" in result.output
    assert "Listings stored : 2" in result.output
    assert "Needs retry" in result.output
    assert "999" in result.output


def test_status_on_a_missing_database_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["status", "--db", str(tmp_path / "none.db")])
    assert result.exit_code != 0
    assert "database not found" in result.output


def test_label_import_applies_and_reports_unknown_shops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    labels = tmp_path / "labels.csv"
    labels.write_text(
        "shop_id,label,source\n123456789,1,manual\n999999999,0,manual\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["label-import", "--db", str(tmp_path / "shops.db"), "--labels", str(labels)],
    )

    assert "Applied 1 labels." in result.output
    assert "1 labels referenced shops not in the database" in result.output


def test_label_import_rejects_a_file_missing_required_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    labels = tmp_path / "labels.csv"
    labels.write_text("shop,fraud\n1,1\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["label-import", "--db", str(tmp_path / "shops.db"), "--labels", str(labels)],
    )

    assert result.exit_code != 0
    assert "missing columns" in result.output


def test_export_writes_derived_features_as_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    out = tmp_path / "dataset.jsonl"
    result = runner.invoke(
        cli.app, ["export", "--db", str(tmp_path / "shops.db"), "--out", str(out)]
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["shop_id"] == 123456789
    assert rows[0]["rating_count_total"] == 400
    assert rows[0]["items_observed"] == 2
    assert rows[0]["label"] is None


def test_export_to_csv_includes_a_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    out = tmp_path / "dataset.csv"
    runner.invoke(
        cli.app,
        [
            "export",
            "--db",
            str(tmp_path / "shops.db"),
            "--out",
            str(out),
            "--format",
            "csv",
        ],
    )

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("shop_id,username,name")
    assert len(lines) == 2


def test_export_can_restrict_itself_to_labeled_shops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    out = tmp_path / "labeled.jsonl"
    result = runner.invoke(
        cli.app,
        [
            "export",
            "--db",
            str(tmp_path / "shops.db"),
            "--out",
            str(out),
            "--labeled-only",
        ],
    )

    assert result.exit_code == 1
    assert "Nothing to export" in result.output


def test_export_rejects_an_unknown_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    result = runner.invoke(
        cli.app,
        [
            "export",
            "--db",
            str(tmp_path / "shops.db"),
            "--out",
            str(tmp_path / "x.txt"),
            "--format",
            "parquet",
        ],
    )

    assert result.exit_code != 0
    assert "must be jsonl or csv" in result.output


def test_rate_and_cookie_settings_reach_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    monkeypatch.setenv("SHOPEE_COOKIE", "SPC_F=abc")

    _scrape(tmp_path, _seeds(tmp_path, "123456789"), "--rate", "2.5")

    assert FakeClient.instances[-1].kwargs["rate"] == 2.5
    assert FakeClient.instances[-1].kwargs["cookie"] == "SPC_F=abc"


def test_env_file_supplies_settings_when_the_environment_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    monkeypatch.delenv("SHOPEE_RATE", raising=False)
    (tmp_path / ".env").write_text(
        "# comment\nSHOPEE_RATE=0.5\nSHOPEE_COOKIE=SPC_F=xyz\n", encoding="utf-8"
    )

    result = _scrape(tmp_path, _seeds(tmp_path, "123456789"))

    assert FakeClient.instances[-1].kwargs["rate"] == 0.5
    assert FakeClient.instances[-1].kwargs["cookie"] == "SPC_F=xyz"
    assert "with session cookie" in result.output


def test_a_persistent_block_stops_the_run_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AbortingClient(FakeClient):
        def fetch_shop(self, ref):
            raise RunAborted("blocked 5 times in a row")

    monkeypatch.setattr(cli, "ShopeeClient", AbortingClient)

    result = _scrape(tmp_path, _seeds(tmp_path, "123456789", "987654321"))

    assert result.exit_code == 0
    assert "Run stopped" in result.output


def test_status_reports_the_measured_sampling_cost(tmp_path: Path) -> None:
    """The estimate that replaced the docstring guess has to actually show up."""
    conn = store.connect(tmp_path / "shops.db")
    run_id = store.start_run(conn, target=5, min_items=1)
    store.finish_run(
        conn, run_id, candidates=100, resolved=10, stored=4, inactive=6, errors=1
    )
    store.record_sampled(conn, 11, SampleOutcome.NOT_FOUND)
    store.record_sampled(conn, 12, SampleOutcome.INACTIVE, 0)
    conn.close()

    result = runner.invoke(cli.app, ["status", "--db", str(tmp_path / "shops.db")])

    assert result.exit_code == 0
    assert "ids sampled   100" in result.output
    assert "resolved      10 (10.0%)" in result.output
    assert "requests/shop 25.0" in result.output
    assert "dead 1, dormant 1" in result.output


def test_status_survives_a_run_that_kept_nothing(tmp_path: Path) -> None:
    """Cost per shop divides by shops kept, which can legitimately be zero."""
    conn = store.connect(tmp_path / "shops.db")
    run_id = store.start_run(conn, target=5, min_items=1)
    store.finish_run(
        conn, run_id, candidates=40, resolved=0, stored=0, inactive=0, errors=0
    )
    conn.close()

    result = runner.invoke(cli.app, ["status", "--db", str(tmp_path / "shops.db")])

    assert result.exit_code == 0
    assert "requests/shop n/a" in result.output


class DiscoveryClient:
    """Stands in for ShopeeClient on the discover path, which uses get_json."""

    stop_after: int | None = None

    def __init__(self, **kwargs) -> None:
        self.asked: list[str] = []
        FakeClient.instances.append(self)

    def __enter__(self) -> "DiscoveryClient":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_json(self, url: str) -> dict:
        if self.stop_after is not None and len(self.asked) >= self.stop_after:
            raise KeyboardInterrupt
        self.asked.append(url)
        shop_id = int(re.search(r"shopid=(\d+)", url).group(1))
        return {"data": {"shopid": shop_id, "username": f"toko{shop_id}",
                         "item_count": 7}}


def _discover(tmp_path: Path, *extra: str):
    return runner.invoke(
        cli.app,
        ["discover", "--db", str(tmp_path / "shops.db"), *extra],
    )


def test_discover_reports_the_run_it_just_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", DiscoveryClient)

    result = _discover(tmp_path, "--target", "2")

    assert result.exit_code == 0
    assert "2 stored from 2 ids sampled" in result.output
    assert "100.0% resolve rate" in result.output


def test_an_interrupted_discover_reports_real_numbers_not_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting "0 stored" for a run that stored two is worse than silence."""
    monkeypatch.setattr(cli, "ShopeeClient", DiscoveryClient)
    monkeypatch.setattr(DiscoveryClient, "stop_after", 2)

    result = _discover(tmp_path, "--target", "50")

    assert result.exit_code == 0
    assert "Interrupted" in result.output
    assert "2 stored" in result.output
    assert "0 stored" not in result.output


def _score(tmp_path: Path, *extra: str):
    return runner.invoke(cli.app, ["score", "--db", str(tmp_path / "shops.db"), *extra])


def _review(tmp_path: Path, *extra: str):
    return runner.invoke(
        cli.app, ["review-queue", "--db", str(tmp_path / "shops.db"), *extra]
    )


def _scrape_several(tmp_path: Path, count: int = 6) -> None:
    _scrape(tmp_path, _seeds(tmp_path, *[str(100 + n) for n in range(count)]))


def test_score_writes_a_ranked_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path)

    out = tmp_path / "queue.csv"
    result = _score(tmp_path, "--out", str(out))

    assert result.exit_code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("rank,shop_id,username")
    assert len(lines) == 7  # header plus one row per shop


def test_score_reports_rule_fire_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule at 0% or 100% carries no information and should be visible."""
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path)

    result = _score(tmp_path)

    assert "Rule fire rates:" in result.output
    assert "negative_reputation" in result.output


def test_score_says_precision_is_unmeasurable_without_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path)

    result = _score(tmp_path)

    assert "no labels yet" in result.output


def test_score_measures_precision_once_labels_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("shop_id,label\n100,1\n101,0\n", encoding="utf-8")
    runner.invoke(
        cli.app,
        ["label-import", "--db", str(tmp_path / "shops.db"), "--labels", str(labels)],
    )

    result = _score(tmp_path, "--at-k", "6")

    assert "precision@6: 50.0%" in result.output


def test_score_rejects_an_unknown_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path, 2)

    result = _score(tmp_path, "--format", "parquet")

    assert result.exit_code != 0


def test_score_rejects_a_weight_outside_the_unit_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path, 2)

    result = _score(tmp_path, "--weight", "1.5")

    assert result.exit_code != 0


def test_score_refuses_an_empty_database(tmp_path: Path) -> None:
    store.connect(tmp_path / "shops.db").close()

    result = _score(tmp_path)

    assert result.exit_code == 1
    assert "Nothing to score" in result.output


def test_score_requires_the_database_to_exist(tmp_path: Path) -> None:
    result = _score(tmp_path)

    assert result.exit_code != 0
    assert "database not found" in result.output


def test_review_queue_writes_a_blank_label_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path)

    out = tmp_path / "review.csv"
    result = _review(tmp_path, "--out", str(out), "--size", "3")

    assert result.exit_code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "shop_id,label,source,shop_url,rank,risk_score,rules_fired"
    assert lines[1].split(",")[1] == ""  # label left for the reviewer
    assert len(lines) == 4


def test_review_queue_output_imports_straight_back_as_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worksheet is only useful if label-import accepts it unedited."""
    monkeypatch.setattr(cli, "ShopeeClient", FakeClient)
    _scrape_several(tmp_path)
    out = tmp_path / "review.csv"
    _review(tmp_path, "--out", str(out))

    # Simulate a reviewer filling in the label column.
    filled = out.read_text(encoding="utf-8").replace(
        ",,manual_review", ",1,manual_review"
    )
    out.write_text(filled, encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["label-import", "--db", str(tmp_path / "shops.db"), "--labels", str(out)],
    )

    assert result.exit_code == 0
    assert "Applied 6 labels" in result.output


def test_review_queue_refuses_an_empty_database(tmp_path: Path) -> None:
    store.connect(tmp_path / "shops.db").close()

    result = _review(tmp_path)

    assert result.exit_code == 1
    assert "Nothing to review" in result.output


def test_review_queue_requires_the_database_to_exist(tmp_path: Path) -> None:
    result = _review(tmp_path)

    assert result.exit_code != 0
    assert "database not found" in result.output
