"""Feature derivation, including how it behaves on incomplete payloads.

The missing-data cases matter as much as the happy path: a feature that
silently becomes 0.0 when Shopee omits a field would poison training.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from shopee_scraper import endpoints
from shopee_scraper.features import derive_features
from shopee_scraper.models import ItemRecord, ShopRecord

# Pinned so age-derived features are deterministic. The fixture shop was
# created at unix 1600000000 = 2020-09-13T12:26:40Z, which is 2157.48 days
# before this instant.
NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)
EXPECTED_AGE_DAYS = 2157.4815


def _record(base: dict | None = None, detail: dict | None = None) -> ShopRecord:
    return ShopRecord(
        shop_id=123456789,
        username="contoh_toko",
        raw_base=base,
        raw_detail=detail,
        fetched_at="2026-08-11T04:22:31Z",
    )


def _items(payload: dict) -> list[ItemRecord]:
    return [
        ItemRecord(
            shop_id=123456789,
            item_id=item["itemid"],
            raw=item,
            fetched_at="2026-08-11T04:22:31Z",
        )
        for item in endpoints.extract_items(payload)
    ]


def test_account_maturity_is_derived_from_ctime(shop_base: dict) -> None:
    features = derive_features(_record(base=shop_base), now=NOW)
    assert features.shop_age_days == pytest.approx(EXPECTED_AGE_DAYS, abs=0.01)


def test_rating_velocity_is_ratings_per_day(shop_base: dict) -> None:
    features = derive_features(_record(base=shop_base), now=NOW)
    # 400 total ratings over 2157.48 days
    assert features.rating_velocity == pytest.approx(400 / EXPECTED_AGE_DAYS, rel=1e-4)


def test_bucketed_ratings_produce_total_and_bad_ratio(shop_base: dict) -> None:
    features = derive_features(_record(base=shop_base), now=NOW)

    assert features.rating_count_total == 400  # 12 bad + 8 normal + 380 good
    assert features.bad_rating_ratio == pytest.approx(0.03)
    assert features.rating_star == 4.7


def test_per_star_breakdown_wins_when_shopee_provides_one() -> None:
    payload = {
        "data": {
            "rating_star_1": 10,
            "rating_star_2": 5,
            "rating_star_3": 5,
            "rating_star_4": 30,
            "rating_star_5": 50,
            "rating_bad": 999,
        }
    }
    features = derive_features(_record(base=payload), now=NOW)

    assert features.rating_count_total == 100
    assert features.bad_rating_ratio == pytest.approx(0.15)


def test_standing_and_scale_fields_are_typed(shop_base: dict) -> None:
    features = derive_features(_record(base=shop_base), now=NOW)

    assert features.follower_count == 1500
    assert features.item_count == 42
    assert features.is_official_shop is False
    assert features.is_preferred_seller is True
    assert features.location == "Jakarta Barat"
    assert features.response_rate == 92
    assert features.response_time_seconds == 3600


def test_fulfilment_and_activity_signals_are_derived(shop_detail: dict) -> None:
    features = derive_features(_record(detail=shop_detail), now=NOW)

    assert features.cancellation_rate == pytest.approx(0.02)
    assert features.is_shopee_verified is True
    # last_active_time is exactly one day before the pinned NOW.
    assert features.days_since_active == pytest.approx(1.0)


def test_rating_buckets_fall_back_to_the_nested_shop_rating_object() -> None:
    """get_shop_base nests the buckets; get_shop_detail puts them top level."""
    payload = {
        "data": {
            "shop_rating": {
                "rating_bad": 5146,
                "rating_normal": 12804,
                "rating_good": 1180534,
            }
        }
    }
    features = derive_features(_record(base=payload), now=NOW)

    assert features.rating_count_total == 1198484
    assert features.bad_rating_ratio == pytest.approx(5146 / 1198484)


def test_a_shop_that_never_reported_activity_has_no_activity_feature() -> None:
    features = derive_features(_record(base={"data": {}}), now=NOW)
    assert features.days_since_active is None
    assert features.cancellation_rate is None


def test_catalog_signals_aggregate_across_listings(
    shop_base: dict, shop_items: dict
) -> None:
    features = derive_features(_record(base=shop_base), _items(shop_items), now=NOW)

    assert features.items_observed == 3
    # Prices 15000 / 25000 / 5000 IDR after dividing out Shopee's scaling.
    assert features.price_median == pytest.approx(15000)
    assert features.price_dispersion == pytest.approx(0.544331, rel=1e-4)
    assert features.zero_stock_ratio == pytest.approx(1 / 3)
    # 75 reviews across 150 units sold.
    assert features.review_to_sold_ratio == pytest.approx(0.5)


def test_detail_overrides_base_where_both_carry_a_field(
    shop_base: dict, shop_detail: dict
) -> None:
    detail = {"data": dict(shop_detail["data"], follower_count=9999)}
    features = derive_features(_record(base=shop_base, detail=detail), now=NOW)
    assert features.follower_count == 9999


def test_empty_payload_yields_nulls_not_zeros() -> None:
    features = derive_features(_record(base={}, detail={}), now=NOW)

    assert features.shop_id == 123456789
    assert features.shop_age_days is None
    assert features.rating_count_total is None
    assert features.bad_rating_ratio is None
    assert features.follower_count is None
    assert features.price_median is None
    assert features.items_observed == 0


def test_zero_ratings_give_an_undefined_ratio_not_zero() -> None:
    payload = {"data": {"rating_bad": 0, "rating_normal": 0, "rating_good": 0}}
    features = derive_features(_record(base=payload), now=NOW)

    assert features.rating_count_total == 0
    assert features.bad_rating_ratio is None
    assert features.rating_velocity is None


def test_future_ctime_is_rejected_rather_than_reported_negative() -> None:
    payload = {"data": {"ctime": int(NOW.timestamp()) + 86_400}}
    assert derive_features(_record(base=payload), now=NOW).shop_age_days is None


def test_single_listing_has_a_median_but_no_dispersion() -> None:
    item = ItemRecord(
        shop_id=123456789,
        item_id=1,
        raw={"itemid": 1, "price": 1000000000, "stock": 3},
        fetched_at="2026-08-11T04:22:31Z",
    )
    features = derive_features(_record(base={}), [item], now=NOW)

    assert features.price_median == pytest.approx(10000)
    assert features.price_dispersion is None
    assert features.zero_stock_ratio == 0.0


def test_listings_missing_sold_counts_skip_the_ratio() -> None:
    item = ItemRecord(
        shop_id=123456789,
        item_id=1,
        raw={"itemid": 1, "price": 1000000000, "item_rating": {"rating_count": [4]}},
        fetched_at="2026-08-11T04:22:31Z",
    )
    features = derive_features(_record(base={}), [item], now=NOW)

    assert features.review_to_sold_ratio is None
    assert features.zero_stock_ratio is None
