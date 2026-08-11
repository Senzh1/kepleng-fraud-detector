"""Contract tests.

These pin the field maps against recorded Shopee responses. When Shopee moves a
field, these fail loudly instead of letting the dataset fill up with NULLs.
"""

from __future__ import annotations

from shopee_scraper import endpoints


def test_urls_carry_the_identifiers_they_are_given() -> None:
    assert "username=contoh_toko" in endpoints.shop_base_url("contoh_toko")
    assert "shopid=123456789" in endpoints.shop_detail_url(123456789)

    items_url = endpoints.shop_items_url(123456789, offset=30, limit=30)
    assert "shopid=123456789" in items_url
    assert "newest=30" in items_url
    assert "limit=30" in items_url


def test_get_path_walks_nested_objects_and_lists() -> None:
    payload = {"data": {"account": {"username": "toko"}, "counts": [7, 8]}}
    assert endpoints.get_path(payload, "data.account.username") == "toko"
    assert endpoints.get_path(payload, "data.counts.1") == 8


def test_get_path_returns_none_instead_of_raising_on_missing_segments() -> None:
    payload = {"data": {"account": {}}}
    assert endpoints.get_path(payload, "data.account.username") is None
    assert endpoints.get_path(payload, "data.missing.deeper") is None
    assert endpoints.get_path(payload, "data.account.username.0") is None
    assert endpoints.get_path({"data": {"counts": [1]}}, "data.counts.9") is None


def test_first_present_prefers_earlier_candidates() -> None:
    payload = {"a": None, "b": 2, "c": 3}
    assert endpoints.first_present(payload, ("a", "b", "c")) == 2
    assert endpoints.first_present(payload, ("missing",)) is None


def test_shop_base_fixture_yields_every_mapped_field(shop_base: dict) -> None:
    fields = endpoints.extract(shop_base, endpoints.SHOP_FIELD_PATHS)

    assert fields["shop_id"] == 123456789
    assert fields["username"] == "contoh_toko"
    assert fields["name"] == "Contoh Toko"
    assert fields["ctime"] == 1600000000
    assert fields["item_count"] == 42
    assert fields["follower_count"] == 1500
    assert fields["rating_star"] == 4.7
    assert fields["rating_bad"] == 12
    assert fields["rating_normal"] == 8
    assert fields["rating_good"] == 380
    assert fields["response_rate"] == 92
    assert fields["response_time"] == 3600
    assert fields["is_preferred_seller"] is True
    assert fields["location"] == "Jakarta Barat"


def test_shop_detail_location_falls_back_to_place(shop_detail: dict) -> None:
    fields = endpoints.extract(shop_detail, endpoints.SHOP_FIELD_PATHS)
    assert fields["location"] == "Jakarta Barat"


def test_per_star_fields_are_absent_upstream_today(shop_base: dict) -> None:
    """Shopee buckets shop ratings; per-star paths must degrade to None."""
    fields = endpoints.extract(shop_base, endpoints.SHOP_FIELD_PATHS)
    assert all(fields[f"rating_star_{n}"] is None for n in range(1, 6))


def test_extract_items_unwraps_the_item_basic_container(shop_items: dict) -> None:
    items = endpoints.extract_items(shop_items)

    assert len(items) == 3
    assert [item["itemid"] for item in items] == [111, 222, 333]

    fields = endpoints.extract(items[0], endpoints.ITEM_FIELD_PATHS)
    assert fields["price"] == 1500000000
    assert fields["stock"] == 10
    assert fields["sold"] == 100
    assert fields["rating_count_total"] == 50


def test_extract_items_tolerates_a_response_with_no_items() -> None:
    assert endpoints.extract_items({"error": None}) == []
    assert endpoints.extract_items({"items": None}) == []
    assert endpoints.extract_items({"items": ["not-a-dict"]}) == []


def test_response_error_ignores_null_and_zero(shop_base: dict) -> None:
    assert endpoints.response_error(shop_base) is None
    assert endpoints.response_error({"error": 0}) is None
    assert endpoints.response_error({"error": False}) is None
    assert endpoints.response_error({"error": 4}) == 4
