"""Rate-limited HTTP collector for Shopee's web API.

Politeness is structural here, not advisory. The client sleeps between requests,
backs off when Shopee pushes back, and stops the whole run after repeated blocks
rather than grinding against a wall. If Shopee refuses this traffic, the correct
response is to slow down or stop — not to disguise it.
"""

from __future__ import annotations

import random
import time
from typing import Any, NamedTuple

import httpx

from . import endpoints
from .models import FetchFailure, FetchStatus, ItemRecord, ShopRecord, utc_now_iso
from .urls import SeedRef

# Shopee returns these on rate limiting and bot suspicion.
BLOCK_STATUSES = frozenset({403, 429})

# Consecutive blocks after which the whole run aborts.
DEFAULT_BLOCK_LIMIT = 5

# Shopee's own "shop does not exist" error codes. 1000000 is what an unused
# shop id returns; verified live 2026-08 while sampling the id space. Without
# it every dead id is filed as an error, and a genuine fault — a schema change,
# a partial outage — is invisible in a run that expects thousands of misses.
NOT_FOUND_ERRORS = frozenset({4, 5, 1_000_000})


# Shopee's "this endpoint requires a logged-in session" code. Catalog endpoints
# return it to anonymous callers; shop profile endpoints do not.
LOGIN_REQUIRED_ERROR = 90309999


class RunAborted(Exception):
    """Raised when Shopee has blocked us persistently enough to stop the run."""


class ShopFetch(NamedTuple):
    """Result of collecting one shop.

    `item_note` explains why `items` is empty when it is. Without it an
    anonymous run reports "ok, 0 listings" for every shop and looks like a
    catalog problem rather than a missing session cookie.
    """

    record: ShopRecord
    items: list[ItemRecord]
    item_note: str | None = None


class RateLimiter:
    """Spaces requests apart, with jitter so the cadence is not machine-perfect."""

    def __init__(self, rate_per_second: float, jitter: float = 0.3) -> None:
        self.interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self.jitter = jitter
        self._last: float = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        delay = self.interval - elapsed
        if delay > 0:
            time.sleep(delay + random.uniform(0, self.jitter * self.interval))
        self._last = time.monotonic()


class ShopeeClient:
    """Fetches shop profiles and listings from Shopee's JSON API."""

    def __init__(
        self,
        rate: float = 1.0,
        cookie: str | None = None,
        timeout: float = 20.0,
        max_retries: int = 3,
        max_items: int = 120,
        block_limit: int = DEFAULT_BLOCK_LIMIT,
        transport: httpx.BaseTransport | None = None,
        sleeper: Any = time.sleep,
    ) -> None:
        headers = dict(endpoints.DEFAULT_HEADERS)
        if cookie:
            headers["Cookie"] = cookie
        self._http = httpx.Client(
            headers=headers, timeout=timeout, transport=transport, follow_redirects=True
        )
        self._limiter = RateLimiter(rate)
        self._sleep = sleeper
        self.max_retries = max_retries
        self.max_items = max_items
        self.block_limit = block_limit
        self.consecutive_blocks = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ShopeeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_json(self, url: str) -> dict[str, Any]:
        """GET one URL and return parsed JSON.

        Raises `FetchFailure` carrying the status the caller should record, or
        `RunAborted` once Shopee has blocked us `block_limit` times in a row.
        """
        last_detail = "no attempt made"
        # Tracked explicitly rather than inferred from the message: a 500 is a
        # server fault, not a block, and conflating them sends the operator
        # hunting for session cookies that would not have helped.
        last_status = FetchStatus.ERROR

        for attempt in range(self.max_retries):
            self._limiter.wait()
            try:
                response = self._http.get(url)
            except httpx.TimeoutException:
                last_detail, last_status = "timeout", FetchStatus.ERROR
                self._backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_detail = f"network error: {exc}"
                last_status = FetchStatus.ERROR
                self._backoff(attempt)
                continue

            if response.status_code in BLOCK_STATUSES:
                self.consecutive_blocks += 1
                if self.consecutive_blocks >= self.block_limit:
                    raise RunAborted(
                        f"blocked {self.consecutive_blocks} times in a row "
                        f"(HTTP {response.status_code}); stopping. Lower --rate, "
                        "supply session cookies, or try again later."
                    )
                last_detail = f"HTTP {response.status_code}"
                last_status = FetchStatus.BLOCKED
                self._backoff(attempt)
                continue

            if response.status_code == 404:
                self.consecutive_blocks = 0
                raise FetchFailure(FetchStatus.NOT_FOUND, "HTTP 404")

            if response.status_code >= 400:
                last_detail = f"HTTP {response.status_code}"
                last_status = FetchStatus.ERROR
                self._backoff(attempt)
                continue

            self.consecutive_blocks = 0
            try:
                payload = response.json()
            except ValueError:
                # Usually an anti-bot interstitial served as HTML.
                raise FetchFailure(
                    FetchStatus.BLOCKED, "response was not JSON"
                ) from None

            if not isinstance(payload, dict):
                raise FetchFailure(FetchStatus.ERROR, "unexpected JSON shape")

            error_code = endpoints.response_error(payload)
            if error_code is not None:
                status = (
                    FetchStatus.NOT_FOUND
                    if error_code in NOT_FOUND_ERRORS
                    else FetchStatus.ERROR
                )
                raise FetchFailure(status, f"shopee error {error_code}")

            return payload

        raise FetchFailure(
            last_status, f"{last_detail} after {self.max_retries} attempts"
        )

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter: 2s, 4s, 8s, ..."""
        self._sleep(2 ** (attempt + 1) + random.uniform(0, 1))

    def fetch_shop(self, ref: SeedRef) -> ShopFetch:
        """Collect one shop's profile and listings.

        A username-only seed is resolved to its numeric id first. Listings are
        best-effort: a shop whose profile was collected but whose catalog was
        refused still counts as a success, with `items_observed` recording what
        was actually seen.
        """
        fetched_at = utc_now_iso()
        raw_base: dict[str, Any] | None = None
        shop_id = ref.shop_id
        username = ref.username

        if username:
            raw_base = self.get_json(endpoints.shop_base_url(username))
            resolved = endpoints.first_present(
                raw_base, endpoints.SHOP_FIELD_PATHS["shop_id"]
            )
            if resolved is None and shop_id is None:
                raise FetchFailure(
                    FetchStatus.NOT_FOUND, f"no shop id for username {username!r}"
                )
            shop_id = shop_id or int(resolved)

        if shop_id is None:
            raise FetchFailure(
                FetchStatus.ERROR, "seed resolved to neither id nor name"
            )

        raw_detail = self.get_json(endpoints.shop_detail_url(shop_id))

        if username is None:
            username = endpoints.first_present(
                raw_detail, endpoints.SHOP_FIELD_PATHS["username"]
            )

        record = ShopRecord(
            shop_id=shop_id,
            username=username,
            raw_base=raw_base,
            raw_detail=raw_detail,
            fetched_at=fetched_at,
        )
        items, note = self._fetch_items(shop_id, fetched_at)
        return ShopFetch(record, items, note)

    def _fetch_items(
        self, shop_id: int, fetched_at: str
    ) -> tuple[list[ItemRecord], str | None]:
        """Page through a shop's catalog up to `max_items`.

        Item collection failing is not fatal: the profile is the labeled unit,
        so a partial catalog beats discarding the shop entirely. The reason is
        returned rather than swallowed, so an empty catalog is never silent.
        """
        collected: list[ItemRecord] = []
        offset = 0

        while offset < self.max_items:
            try:
                payload = self.get_json(endpoints.shop_items_url(shop_id, offset))
            except FetchFailure as failure:
                if collected:
                    return collected, f"partial catalog: {failure.detail}"
                if failure.status is FetchStatus.BLOCKED:
                    return [], (
                        "catalog refused — Shopee requires a logged-in session "
                        "for listings; set SHOPEE_COOKIE in .env"
                    )
                return [], f"catalog unavailable: {failure.detail}"

            items = endpoints.extract_items(payload)
            if not items:
                if collected:
                    break
                return [], "catalog returned no items"

            for item in items:
                item_id = endpoints.first_present(
                    item, endpoints.ITEM_FIELD_PATHS["item_id"]
                )
                if item_id is None:
                    continue
                collected.append(
                    ItemRecord(
                        shop_id=shop_id,
                        item_id=int(item_id),
                        raw=item,
                        fetched_at=fetched_at,
                    )
                )

            if len(items) < endpoints.ITEMS_PAGE_SIZE:
                break
            offset += endpoints.ITEMS_PAGE_SIZE

        return collected, None
