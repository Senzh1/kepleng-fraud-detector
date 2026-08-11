"""Resolve Shopee Indonesia URLs and seed strings into canonical shop references.

Pure functions, no I/O. Every supported input form is pinned by tests in
tests/test_urls.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

SHOPEE_HOSTS = frozenset(
    {
        "shopee.co.id",
        "www.shopee.co.id",
        "id.shopee.com",
        "www.id.shopee.com",
    }
)

# First path segments that are Shopee features, never shop usernames. A single
# segment URL is only read as a username when it is not in this set.
RESERVED_SEGMENTS = frozenset(
    {
        "api",
        "buyer",
        "cart",
        "daily-discover",
        "find",
        "flash_sale",
        "login",
        "m",
        "mall",
        "official-shop",
        "product",
        "search",
        "seller",
        "shop",
        "signup",
        "user",
        "verify",
        "web",
    }
)

# Product permalink tail: <slug>-i.<shop_id>.<item_id>
_PRODUCT_TAIL_RE = re.compile(r"-i\.(?P<shop_id>\d+)\.(?P<item_id>\d+)$")
_DIGITS_RE = re.compile(r"^\d+$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class InvalidSeedError(ValueError):
    """A seed string could not be resolved to a Shopee shop reference."""


@dataclass(frozen=True)
class SeedRef:
    """A resolved pointer to a Shopee shop.

    At least one of `shop_id` or `username` is present. `shop_id` is preferred;
    when only a username is known the collector resolves the numeric id from the
    shop base endpoint before storing anything.
    """

    raw: str
    shop_id: int | None = None
    username: str | None = None
    item_id: int | None = None

    @property
    def is_resolved(self) -> bool:
        """True when the numeric shop id is already known."""
        return self.shop_id is not None


def resolve_seed(raw: str) -> SeedRef:
    """Turn one seed line into a `SeedRef`.

    Accepts a bare numeric shop id, a bare username, a shop URL, or a product
    URL in either the permalink or `/product/<shop>/<item>` form.

    Raises `InvalidSeedError` on anything else, including URLs pointing at a
    non-Shopee host. Callers are expected to record the failure and continue
    rather than abort the run.
    """
    text = raw.strip()
    if not text:
        raise InvalidSeedError("empty seed")

    if _DIGITS_RE.match(text):
        return SeedRef(raw=raw, shop_id=int(text))

    parsed = urlparse(text if "//" in text else f"//{text}", scheme="https")

    host = (parsed.netloc or "").lower().split(":")[0]
    segments = [seg for seg in parsed.path.split("/") if seg]

    if host and host not in SHOPEE_HOSTS:
        # A bare token with no slashes lands in netloc; treat it as a username
        # rather than rejecting it as a foreign host.
        if not segments and "." not in host and _USERNAME_RE.match(host):
            return SeedRef(raw=raw, username=host)
        raise InvalidSeedError(f"not a Shopee host: {host}")

    if not segments:
        raise InvalidSeedError(f"no shop or product path in seed: {raw!r}")

    return _from_segments(raw, segments)


def _from_segments(raw: str, segments: list[str]) -> SeedRef:
    head = segments[0].lower()

    if head == "shop" and len(segments) >= 2 and _DIGITS_RE.match(segments[1]):
        return SeedRef(raw=raw, shop_id=int(segments[1]))

    if (
        head == "product"
        and len(segments) >= 3
        and _DIGITS_RE.match(segments[1])
        and _DIGITS_RE.match(segments[2])
    ):
        return SeedRef(raw=raw, shop_id=int(segments[1]), item_id=int(segments[2]))

    tail = _PRODUCT_TAIL_RE.search(segments[-1])
    if tail:
        return SeedRef(
            raw=raw,
            shop_id=int(tail.group("shop_id")),
            item_id=int(tail.group("item_id")),
        )

    if len(segments) == 1 and head not in RESERVED_SEGMENTS:
        if _USERNAME_RE.match(segments[0]):
            return SeedRef(raw=raw, username=segments[0])

    raise InvalidSeedError(f"unrecognised Shopee path: {raw!r}")


def load_seeds(lines: list[str]) -> tuple[list[SeedRef], list[tuple[str, str]]]:
    """Resolve many seeds, partitioning successes from failures.

    Returns `(refs, failures)` where each failure is `(raw_line, reason)`. A
    malformed line never aborts the batch; the caller reports it and carries on.
    """
    refs: list[SeedRef] = []
    failures: list[tuple[str, str]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            refs.append(resolve_seed(line))
        except InvalidSeedError as exc:
            failures.append((line, str(exc)))
    return refs, failures
