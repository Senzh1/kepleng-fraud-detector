"""Seed parsing must accept every URL form a human might paste."""

from __future__ import annotations

import pytest

from shopee_scraper.urls import InvalidSeedError, load_seeds, resolve_seed


@pytest.mark.parametrize(
    "raw",
    [
        "https://shopee.co.id/shop/123456789",
        "https://www.shopee.co.id/shop/123456789",
        "http://shopee.co.id/shop/123456789/search",
        "shopee.co.id/shop/123456789",
        "123456789",
        "  123456789  ",
    ],
)
def test_resolves_shop_id_from_every_shop_form(raw: str) -> None:
    assert resolve_seed(raw).shop_id == 123456789


@pytest.mark.parametrize(
    "raw",
    [
        "https://shopee.co.id/Sepatu-Pria-Keren-i.123456789.987654321",
        "https://shopee.co.id/product/123456789/987654321",
        "https://shopee.co.id/Tas-Wanita-i.123456789.987654321/",
    ],
)
def test_resolves_shop_and_item_from_product_urls(raw: str) -> None:
    ref = resolve_seed(raw)
    assert ref.shop_id == 123456789
    assert ref.item_id == 987654321


def test_product_url_with_query_string_is_parsed() -> None:
    ref = resolve_seed(
        "https://shopee.co.id/Sepatu-i.123456789.987654321?sp_atk=abc&xptdk=def"
    )
    assert (ref.shop_id, ref.item_id) == (123456789, 987654321)


@pytest.mark.parametrize(
    "raw",
    [
        "https://shopee.co.id/contoh_toko",
        "shopee.co.id/contoh_toko",
        "contoh_toko",
    ],
)
def test_resolves_username_when_no_numeric_id_present(raw: str) -> None:
    ref = resolve_seed(raw)
    assert ref.username == "contoh_toko"
    assert ref.shop_id is None
    assert ref.is_resolved is False


def test_shop_id_seed_is_already_resolved() -> None:
    assert resolve_seed("123456789").is_resolved is True


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "https://tokopedia.com/shop/123",
        "https://shopee.co.id/search?keyword=hp",
        "https://shopee.co.id/cart",
        "https://shopee.co.id/user/purchase",
    ],
)
def test_rejects_unusable_seeds(raw: str) -> None:
    with pytest.raises(InvalidSeedError):
        resolve_seed(raw)


def test_reserved_segment_is_not_mistaken_for_a_username() -> None:
    with pytest.raises(InvalidSeedError):
        resolve_seed("https://shopee.co.id/mall")


def test_load_seeds_partitions_good_from_bad_without_aborting() -> None:
    refs, failures = load_seeds(
        [
            "123456789",
            "",
            "https://tokopedia.com/shop/1",
            "https://shopee.co.id/contoh_toko",
        ]
    )
    assert [r.shop_id for r in refs] == [123456789, None]
    assert len(failures) == 1
    assert failures[0][0] == "https://tokopedia.com/shop/1"
