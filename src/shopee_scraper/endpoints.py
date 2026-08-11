"""Every Shopee endpoint URL and response field path lives here.

Shopee's web API is undocumented and changes without notice. This module is the
intended blast radius for those changes: when a field moves or an endpoint
disappears, fix it here and nowhere else. tests/test_endpoints.py pins the field
maps against recorded fixtures, so an upstream change fails a test instead of
silently filling your dataset with NULL columns.

Field maps list *candidate* paths per logical field, tried in order, first hit
wins. That absorbs the common case where Shopee moves a value between the top
level and a nested object without renaming it.

Verified against shopee.co.id in 2026-08. Treat as perishable.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

BASE_URL = "https://shopee.co.id"

# A single honest User-Agent. Deliberately not a rotating pool: this collector
# identifies itself consistently rather than disguising its traffic.
DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
        "shopee-scraper/0.1 (research)"
    ),
    "Accept": "application/json",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
    "X-API-SOURCE": "pc",
    "Referer": f"{BASE_URL}/",
}

# Shopee reports prices as integers scaled by 100_000.
PRICE_DIVISOR = 100_000

# Items requested per page from the shop item listing.
ITEMS_PAGE_SIZE = 30


def shop_base_url(username: str) -> str:
    """Shop profile by username slug."""
    return f"{BASE_URL}/api/v4/shop/get_shop_base?{urlencode({'username': username})}"


def shop_detail_url(shop_id: int) -> str:
    """Shop profile by numeric id."""
    return f"{BASE_URL}/api/v4/shop/get_shop_detail?{urlencode({'shopid': shop_id})}"


def shop_items_url(shop_id: int, offset: int = 0, limit: int = ITEMS_PAGE_SIZE) -> str:
    """One page of a shop's listings."""
    params = {
        "by": "pop",
        "shopid": shop_id,
        "limit": limit,
        "newest": offset,
        "order": "desc",
        "page_type": "shop",
        "version": 2,
    }
    return f"{BASE_URL}/api/v4/search/search_items?{urlencode(params)}"


def shop_page_url(shop_id: int) -> str:
    """Human-facing shop page, used by the browser fallback."""
    return f"{BASE_URL}/shop/{shop_id}"


# Logical field -> candidate dotted paths, first hit wins.
SHOP_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "shop_id": ("data.shopid", "data.account.shopid"),
    "username": ("data.account.username", "data.username"),
    "name": ("data.name",),
    "ctime": ("data.ctime",),
    "item_count": ("data.item_count",),
    "follower_count": ("data.follower_count",),
    "rating_star": ("data.account.rating_star", "data.rating_star"),
    # Shopee reports shop ratings bucketed, not per-star. Per-star paths are
    # listed in case a future response exposes them; they resolve to None today.
    "rating_star_1": ("data.rating_star_1",),
    "rating_star_2": ("data.rating_star_2",),
    "rating_star_3": ("data.rating_star_3",),
    "rating_star_4": ("data.rating_star_4",),
    "rating_star_5": ("data.rating_star_5",),
    # Present at the top level on get_shop_detail, nested under shop_rating on
    # get_shop_base. Both observed live 2026-08.
    "rating_bad": ("data.rating_bad", "data.shop_rating.rating_bad"),
    "rating_normal": ("data.rating_normal", "data.shop_rating.rating_normal"),
    "rating_good": ("data.rating_good", "data.shop_rating.rating_good"),
    "response_rate": ("data.response_rate",),
    "response_time": ("data.response_time",),
    "is_official_shop": ("data.is_official_shop",),
    "is_preferred_seller": (
        "data.is_preferred_plus_seller",
        "data.is_preferred_seller",
    ),
    "is_shopee_verified": ("data.is_shopee_verified",),
    # Order cancellations attributable to the seller. A strong fraud signal:
    # stores that take orders they cannot or will not fulfil.
    "cancellation_rate": (
        "data.cancellation_rate",
        "data.seller_metrics.cancellation_rate",
    ),
    "last_active_time": ("data.last_active_time",),
    "location": ("data.shop_location", "data.place"),
}

ITEM_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "item_id": ("itemid",),
    "shop_id": ("shopid",),
    "name": ("name",),
    "price": ("price",),
    "stock": ("stock",),
    "sold": ("historical_sold", "sold"),
    "rating_star": ("item_rating.rating_star",),
    "rating_count_total": ("item_rating.rating_count.0",),
}

# Where the item array lives in a search_items response.
ITEMS_LIST_PATHS: tuple[str, ...] = ("items", "data.items")

# Each list entry wraps the real payload; unwrap the first key that exists.
ITEM_ENTRY_UNWRAP: tuple[str, ...] = ("item_basic", "basic")

# A response carrying a non-zero code here is a hard upstream refusal.
ERROR_PATHS: tuple[str, ...] = ("error", "data.error")


def get_path(payload: Any, path: str) -> Any:
    """Read a dotted path out of a JSON-ish structure.

    Integer segments index into lists. Returns None when any segment is
    missing, so a moved field degrades to NULL rather than raising.
    """
    current = payload
    for segment in path.split("."):
        if current is None:
            return None
        if segment.isdigit() and isinstance(current, (list, tuple)):
            index = int(segment)
            if index >= len(current):
                return None
            current = current[index]
        elif isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def first_present(payload: Any, paths: tuple[str, ...]) -> Any:
    """Return the first non-None value among candidate paths."""
    for path in paths:
        value = get_path(payload, path)
        if value is not None:
            return value
    return None


def extract(payload: Any, field_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Apply a whole field map to one payload."""
    return {field: first_present(payload, paths) for field, paths in field_map.items()}


def extract_items(payload: Any) -> list[dict[str, Any]]:
    """Pull the item objects out of a search_items response.

    Unwraps the per-entry container so callers receive flat item dicts. Returns
    an empty list when the response carries no items at all.
    """
    raw_list = first_present(payload, ITEMS_LIST_PATHS)
    if not isinstance(raw_list, list):
        return []

    items: list[dict[str, Any]] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        unwrapped = entry
        for key in ITEM_ENTRY_UNWRAP:
            candidate = entry.get(key)
            if isinstance(candidate, dict):
                unwrapped = candidate
                break
        items.append(unwrapped)
    return items


def response_error(payload: Any) -> int | None:
    """Return Shopee's error code when the payload reports one."""
    value = first_present(payload, ERROR_PATHS)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value != 0:
        return value
    return None
