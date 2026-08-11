"""Playwright fallback fetcher.

Used only when the JSON API refuses a shop the operator still wants. Driving a
real browser survives endpoint changes that break `client.py`, at roughly twenty
times the cost per shop, so it is opt-in via `--browser-fallback`.

Playwright is an optional dependency, imported lazily so the package installs
and the test suite runs without it.
"""

from __future__ import annotations

import json
from typing import Any

from . import endpoints
from .models import FetchFailure, FetchStatus

INSTALL_HINT = (
    "Playwright is not installed. Run:\n"
    "    pip install -e .[browser]\n"
    "    playwright install chromium"
)


class BrowserFetcher:
    """Fetches Shopee JSON through a real browser session.

    Visits the Shopee home page first so that Shopee's own cookies are set, then
    issues API calls from inside that browser context.
    """

    def __init__(
        self,
        headless: bool = True,
        cookie: str | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        self.headless = headless
        self.cookie = cookie
        self.timeout_ms = timeout_ms
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None

    def start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(INSTALL_HINT) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=endpoints.DEFAULT_HEADERS["User-Agent"],
            locale="id-ID",
        )
        if self.cookie:
            self._context.add_cookies(_parse_cookie_header(self.cookie))

        page = self._context.new_page()
        page.goto(endpoints.BASE_URL, timeout=self.timeout_ms)
        page.close()

    def close(self) -> None:
        for resource in (self._context, self._browser, self._playwright):
            if resource is None:
                continue
            closer = getattr(resource, "close", None) or getattr(resource, "stop", None)
            if closer:
                closer()
        self._context = self._browser = self._playwright = None

    def __enter__(self) -> "BrowserFetcher":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_json(self, url: str) -> dict[str, Any]:
        """Issue one API request from inside the browser context."""
        if self._context is None:
            raise RuntimeError("BrowserFetcher.start() was not called")

        response = self._context.request.get(url, timeout=self.timeout_ms)

        if response.status == 404:
            raise FetchFailure(FetchStatus.NOT_FOUND, "HTTP 404 (browser)")
        if response.status >= 400:
            raise FetchFailure(FetchStatus.BLOCKED, f"HTTP {response.status} (browser)")

        try:
            payload = json.loads(response.text())
        except ValueError:
            raise FetchFailure(
                FetchStatus.BLOCKED, "response was not JSON (browser)"
            ) from None

        if not isinstance(payload, dict):
            raise FetchFailure(FetchStatus.ERROR, "unexpected JSON shape (browser)")

        error_code = endpoints.response_error(payload)
        if error_code is not None:
            raise FetchFailure(FetchStatus.ERROR, f"shopee error {error_code}")

        return payload


def _parse_cookie_header(cookie: str) -> list[dict[str, str]]:
    """Turn a raw `name=value; name=value` header into Playwright cookies."""
    cookies: list[dict[str, str]] = []
    for pair in cookie.split(";"):
        name, _, value = pair.strip().partition("=")
        if name and value:
            cookies.append(
                {
                    "name": name,
                    "value": value,
                    "domain": ".shopee.co.id",
                    "path": "/",
                }
            )
    return cookies
