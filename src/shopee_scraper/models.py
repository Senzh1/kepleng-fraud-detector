"""Typed records passed between the fetcher, the store, and feature derivation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601 with a Z suffix, e.g. 2026-08-11T04:22:31Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FetchStatus(str, Enum):
    """Terminal outcome of trying to collect one shop.

    Only OK is skipped on a re-run; everything else is retried.
    """

    OK = "ok"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"
    ERROR = "error"


class FetchFailure(Exception):
    """A fetch that failed in a way worth recording rather than crashing on."""

    def __init__(self, status: FetchStatus, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class ShopRecord(BaseModel):
    """Raw collected payloads for one shop, before any feature derivation."""

    shop_id: int
    username: str | None = None
    raw_base: dict[str, Any] | None = None
    raw_detail: dict[str, Any] | None = None
    fetched_at: str = Field(default_factory=utc_now_iso)


class ItemRecord(BaseModel):
    """One listing belonging to a shop."""

    shop_id: int
    item_id: int
    raw: dict[str, Any]
    fetched_at: str = Field(default_factory=utc_now_iso)


class ShopFeatures(BaseModel):
    """Derived, training-ready feature row for one shop.

    Every derived field is optional. None means "not observed" and must stay
    distinguishable from a real zero when training — do not fillna(0) blindly.
    """

    shop_id: int
    username: str | None = None
    name: str | None = None
    fetched_at: str | None = None

    # Account maturity
    shop_age_days: float | None = None
    rating_velocity: float | None = None

    # Reputation
    rating_star: float | None = None
    rating_count_total: int | None = None
    rating_star_1: int | None = None
    rating_star_2: int | None = None
    rating_star_3: int | None = None
    rating_star_4: int | None = None
    rating_star_5: int | None = None
    rating_bad: int | None = None
    rating_normal: int | None = None
    rating_good: int | None = None
    bad_rating_ratio: float | None = None
    response_rate: float | None = None
    response_time_seconds: float | None = None

    # Fulfilment reliability
    cancellation_rate: float | None = None

    # Scale and standing
    follower_count: int | None = None
    item_count: int | None = None
    is_official_shop: bool | None = None
    is_preferred_seller: bool | None = None
    is_shopee_verified: bool | None = None
    days_since_active: float | None = None
    location: str | None = None

    # Catalog signals
    items_observed: int = 0
    price_median: float | None = None
    price_dispersion: float | None = None
    zero_stock_ratio: float | None = None
    review_to_sold_ratio: float | None = None

    # Supervision
    label: int | None = None
    label_source: str | None = None
