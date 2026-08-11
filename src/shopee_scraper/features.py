"""Derive training-ready store features from raw Shopee payloads.

Pure functions, no I/O. Because raw responses are kept in SQLite, this module
can be rewritten and re-run over an existing database without re-scraping.

Every derivation is defensive: a missing or unusable upstream value yields None,
never a crash and never an imputed zero. None means "not observed" and must stay
distinguishable from a real zero during training.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from . import endpoints
from .models import ItemRecord, ShopFeatures, ShopRecord

SECONDS_PER_DAY = 86_400


def _as_number(value: Any) -> float | None:
    """Coerce to float, rejecting bools and anything non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    return None if number is None else int(number)


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    number = _as_number(value)
    return None if number is None else bool(number)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Safe division. Returns None when either side is missing or the
    denominator is zero — an undefined ratio is not the same as zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _sum_present(values: list[int | None]) -> int | None:
    """Sum the values that exist, or None when none of them do."""
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _days_since(timestamp: Any, now: datetime) -> float | None:
    """Days between a unix timestamp and `now`.

    Returns None for missing, non-positive, or future timestamps — a negative
    age is upstream nonsense, not a feature worth training on.
    """
    moment = _as_number(timestamp)
    if moment is None or moment <= 0:
        return None
    elapsed = now.timestamp() - moment
    return None if elapsed < 0 else elapsed / SECONDS_PER_DAY


def _shop_age_days(ctime: Any, now: datetime) -> float | None:
    """Days since the shop was created, from Shopee's unix `ctime`."""
    return _days_since(ctime, now)


def _rating_totals(shop: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return `(total_ratings, negative_ratings)`.

    Prefers a genuine per-star breakdown when Shopee exposes one. Falls back to
    the bucketed bad/normal/good counts that the shop endpoints actually return
    today, where "bad" already means 1 and 2 star.
    """
    stars = [_as_int(shop.get(f"rating_star_{n}")) for n in range(1, 6)]
    if any(star is not None for star in stars):
        total = _sum_present(stars)
        negative = _sum_present(stars[:2])
        return total, negative

    bad = _as_int(shop.get("rating_bad"))
    normal = _as_int(shop.get("rating_normal"))
    good = _as_int(shop.get("rating_good"))
    return _sum_present([bad, normal, good]), bad


def _catalog_features(items: list[ItemRecord]) -> dict[str, Any]:
    """Aggregate listing-level signals across a shop's observed catalog."""
    if not items:
        return {"items_observed": 0}

    prices: list[float] = []
    stocks: list[int] = []
    reviews_total = 0
    sold_total = 0
    saw_reviews = False
    saw_sold = False

    for item in items:
        fields = endpoints.extract(item.raw, endpoints.ITEM_FIELD_PATHS)

        price = _as_number(fields.get("price"))
        if price is not None and price > 0:
            prices.append(price / endpoints.PRICE_DIVISOR)

        stock = _as_int(fields.get("stock"))
        if stock is not None:
            stocks.append(stock)

        reviews = _as_int(fields.get("rating_count_total"))
        if reviews is not None:
            reviews_total += reviews
            saw_reviews = True

        sold = _as_int(fields.get("sold"))
        if sold is not None:
            sold_total += sold
            saw_sold = True

    derived: dict[str, Any] = {"items_observed": len(items)}

    if prices:
        derived["price_median"] = statistics.median(prices)
        mean = statistics.fmean(prices)
        # Coefficient of variation: spread normalised by level, so a cheap
        # catalog and an expensive one stay comparable.
        if len(prices) > 1 and mean > 0:
            derived["price_dispersion"] = statistics.pstdev(prices) / mean

    if stocks:
        derived["zero_stock_ratio"] = sum(1 for s in stocks if s <= 0) / len(stocks)

    if saw_reviews and saw_sold:
        derived["review_to_sold_ratio"] = _ratio(reviews_total, sold_total)

    return derived


def derive_features(
    record: ShopRecord,
    items: list[ItemRecord] | None = None,
    now: datetime | None = None,
) -> ShopFeatures:
    """Build the feature row for one shop from its stored raw payloads."""
    now = now or datetime.now(timezone.utc)
    items = items or []

    # Base and detail carry overlapping fields; detail wins where both exist.
    shop: dict[str, Any] = {}
    for payload in (record.raw_base, record.raw_detail):
        if payload:
            extracted = endpoints.extract(payload, endpoints.SHOP_FIELD_PATHS)
            shop.update({k: v for k, v in extracted.items() if v is not None})

    total_ratings, negative_ratings = _rating_totals(shop)
    age_days = _shop_age_days(shop.get("ctime"), now)

    return ShopFeatures(
        shop_id=record.shop_id,
        username=record.username or shop.get("username"),
        name=shop.get("name"),
        fetched_at=record.fetched_at,
        shop_age_days=age_days,
        rating_velocity=_ratio(total_ratings, age_days),
        rating_star=_as_number(shop.get("rating_star")),
        rating_count_total=total_ratings,
        rating_star_1=_as_int(shop.get("rating_star_1")),
        rating_star_2=_as_int(shop.get("rating_star_2")),
        rating_star_3=_as_int(shop.get("rating_star_3")),
        rating_star_4=_as_int(shop.get("rating_star_4")),
        rating_star_5=_as_int(shop.get("rating_star_5")),
        rating_bad=_as_int(shop.get("rating_bad")),
        rating_normal=_as_int(shop.get("rating_normal")),
        rating_good=_as_int(shop.get("rating_good")),
        bad_rating_ratio=_ratio(negative_ratings, total_ratings),
        response_rate=_as_number(shop.get("response_rate")),
        response_time_seconds=_as_number(shop.get("response_time")),
        cancellation_rate=_as_number(shop.get("cancellation_rate")),
        follower_count=_as_int(shop.get("follower_count")),
        item_count=_as_int(shop.get("item_count")),
        is_official_shop=_as_bool(shop.get("is_official_shop")),
        is_preferred_seller=_as_bool(shop.get("is_preferred_seller")),
        is_shopee_verified=_as_bool(shop.get("is_shopee_verified")),
        days_since_active=_days_since(shop.get("last_active_time"), now),
        location=shop.get("location"),
        **_catalog_features(items),
    )
