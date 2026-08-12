"""HTTP behaviour, driven entirely through a mock transport.

No test here touches the network. Backoff sleeps are stubbed so the suite stays
fast while still exercising the retry paths.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest
from conftest import load_fixture

from shopee_scraper.client import RateLimiter, RunAborted, ShopeeClient
from shopee_scraper.models import FetchFailure, FetchStatus
from shopee_scraper.urls import SeedRef


def _client(handler, **kwargs) -> ShopeeClient:
    """Client wired to a mock transport, with rate limiting and sleeps disabled."""
    defaults = {"rate": 0.0, "sleeper": lambda _seconds: None}
    return ShopeeClient(transport=httpx.MockTransport(handler), **{**defaults, **kwargs})


def _shopee_router(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("get_shop_base"):
        return httpx.Response(200, json=load_fixture("shop_base.json"))
    if path.endswith("get_shop_detail"):
        return httpx.Response(200, json=load_fixture("shop_detail.json"))
    if path.endswith("search_items"):
        return httpx.Response(200, json=load_fixture("shop_items.json"))
    return httpx.Response(404)


def test_successful_json_is_returned_verbatim() -> None:
    with _client(lambda _r: httpx.Response(200, json={"error": None, "ok": 1})) as c:
        assert c.get_json("https://shopee.co.id/x") == {"error": None, "ok": 1}


def test_404_is_not_found_and_is_not_retried() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    with _client(handler) as client:
        with pytest.raises(FetchFailure) as caught:
            client.get_json("https://shopee.co.id/x")

    assert caught.value.status is FetchStatus.NOT_FOUND
    assert len(calls) == 1


def test_rate_limiting_is_retried_then_reported_as_blocked() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429)

    with _client(handler, max_retries=2, block_limit=99) as client:
        with pytest.raises(FetchFailure) as caught:
            client.get_json("https://shopee.co.id/x")

    assert caught.value.status is FetchStatus.BLOCKED
    assert len(calls) == 2


def test_persistent_blocking_aborts_the_whole_run() -> None:
    with _client(lambda _r: httpx.Response(403), block_limit=2) as client:
        with pytest.raises(RunAborted, match="blocked 2 times"):
            client.get_json("https://shopee.co.id/x")


def test_an_html_interstitial_counts_as_a_block_not_a_parse_bug() -> None:
    with _client(lambda _r: httpx.Response(200, text="<html>captcha</html>")) as client:
        with pytest.raises(FetchFailure) as caught:
            client.get_json("https://shopee.co.id/x")

    assert caught.value.status is FetchStatus.BLOCKED


def test_an_unused_shop_id_is_not_found_rather_than_an_error() -> None:
    """Sampling the id space produces thousands of these.

    Filed as errors they would drown out a genuine fault, which is the only
    signal that tells the operator to stop the run.
    """
    with _client(lambda _r: httpx.Response(200, json={"error": 1_000_000})) as client:
        with pytest.raises(FetchFailure) as caught:
            client.get_json("https://shopee.co.id/x")

    assert caught.value.status is FetchStatus.NOT_FOUND


def test_shopee_error_codes_map_to_a_status() -> None:
    with _client(lambda _r: httpx.Response(200, json={"error": 4})) as client:
        with pytest.raises(FetchFailure) as caught:
            client.get_json("https://shopee.co.id/x")
    assert caught.value.status is FetchStatus.NOT_FOUND

    with _client(lambda _r: httpx.Response(200, json={"error": 99})) as client:
        with pytest.raises(FetchFailure) as caught:
            client.get_json("https://shopee.co.id/x")
    assert caught.value.status is FetchStatus.ERROR


def test_timeouts_are_retried_then_reported_as_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow", request=request)

    with _client(handler, max_retries=2) as client:
        with pytest.raises(FetchFailure) as caught:
            client.get_json("https://shopee.co.id/x")

    assert caught.value.status is FetchStatus.ERROR
    assert "timeout" in caught.value.detail


def test_a_successful_response_clears_the_block_counter() -> None:
    responses = [httpx.Response(429), httpx.Response(200, json={"ok": 1})]

    with _client(lambda _r: responses.pop(0), block_limit=99) as client:
        client.get_json("https://shopee.co.id/x")
        assert client.consecutive_blocks == 0


def test_fetch_shop_by_username_resolves_the_numeric_id() -> None:
    with _client(_shopee_router) as client:
        fetched = client.fetch_shop(SeedRef(raw="contoh_toko", username="contoh_toko"))

    assert fetched.record.shop_id == 123456789
    assert fetched.record.username == "contoh_toko"
    assert fetched.record.raw_base is not None
    assert fetched.record.raw_detail is not None
    assert [item.item_id for item in fetched.items] == [111, 222, 333]
    assert fetched.item_note is None


def test_fetch_shop_by_id_skips_the_username_lookup() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _shopee_router(request)

    with _client(handler) as client:
        fetched = client.fetch_shop(SeedRef(raw="123456789", shop_id=123456789))

    assert fetched.record.shop_id == 123456789
    # Username backfilled from the detail payload.
    assert fetched.record.username == "contoh_toko"
    assert not any(path.endswith("get_shop_base") for path in seen)


def test_a_short_item_page_ends_pagination() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return _shopee_router(request)

    with _client(handler) as client:
        fetched = client.fetch_shop(SeedRef(raw="123456789", shop_id=123456789))

    # Three items is under one page, so exactly one listing call is made.
    assert sum(1 for url in calls if "search_items" in url) == 1
    assert len(fetched.items) == 3


def test_a_refused_catalog_does_not_lose_the_shop() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "search_items" in str(request.url):
            return httpx.Response(500)
        return _shopee_router(request)

    with _client(handler, max_retries=1) as client:
        fetched = client.fetch_shop(SeedRef(raw="123456789", shop_id=123456789))

    assert fetched.record.shop_id == 123456789
    assert fetched.items == []
    assert "catalog unavailable" in fetched.item_note


def test_an_anonymous_catalog_refusal_says_to_supply_cookies() -> None:
    """Shopee returns 403 for listings without a session. The operator must be
    told that, not left staring at "ok, 0 listings"."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "search_items" in str(request.url):
            return httpx.Response(403, json={"error": 90309999, "is_login": False})
        return _shopee_router(request)

    with _client(handler, max_retries=1, block_limit=99) as client:
        fetched = client.fetch_shop(SeedRef(raw="123456789", shop_id=123456789))

    assert fetched.items == []
    assert "SHOPEE_COOKIE" in fetched.item_note


def test_an_empty_catalog_is_reported_rather_than_silent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "search_items" in str(request.url):
            return httpx.Response(200, json={"error": None, "items": []})
        return _shopee_router(request)

    with _client(handler) as client:
        fetched = client.fetch_shop(SeedRef(raw="123456789", shop_id=123456789))

    assert fetched.items == []
    assert fetched.item_note == "catalog returned no items"


def test_a_missing_shop_id_for_a_username_is_not_found() -> None:
    with _client(lambda _r: httpx.Response(200, json={"data": {}})) as client:
        with pytest.raises(FetchFailure) as caught:
            client.fetch_shop(SeedRef(raw="ghost", username="ghost"))

    assert caught.value.status is FetchStatus.NOT_FOUND


def test_cookies_are_sent_when_supplied() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("cookie"))
        return httpx.Response(200, json={"ok": 1})

    with _client(handler, cookie="SPC_F=abc; SPC_EC=def") as client:
        client.get_json("https://shopee.co.id/x")

    assert seen == ["SPC_F=abc; SPC_EC=def"]


def test_the_rate_limiter_holds_under_concurrent_callers() -> None:
    """Politeness has to survive two people using the app at once.

    The API runs its endpoint in a threadpool over one shared client, so two
    requests can enter `wait` together. Reading `_last` without exclusion lets
    both conclude that nothing is owed, and the configured rate is quietly
    exceeded — the one guarantee this collector makes to Shopee.
    """
    limiter = RateLimiter(20.0, jitter=0.0)  # one call per 50ms
    started = time.monotonic()

    threads = [threading.Thread(target=limiter.wait) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The first caller owes nothing; the other three each wait out an interval.
    assert time.monotonic() - started >= 0.15
