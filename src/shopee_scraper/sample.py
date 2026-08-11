"""Random shop discovery by numeric id.

Shopee's search endpoints all require a logged-in session, so there is no
anonymous way to *list* shops. Shop detail, however, answers anonymously for
any numeric id. Sampling that id space uniformly gives what a seed list cannot:
an unbiased cross-section of ordinary sellers.

That property matters more than convenience. A hand-written seed list is
whatever shops the operator happened to think of — in practice large, verified
brands — and a model trained on it learns to recognise brands rather than
fraud. Uniform sampling has no such preference.

Measured against production 2026-08: roughly 10% of sampled ids resolve to a
shop, and most of those are dormant buyer accounts with no listings, hence the
activity filter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from . import endpoints
from .client import ShopeeClient
from .models import FetchFailure, FetchStatus, ShopRecord, utc_now_iso
from .store import record_status, seed_key, upsert_shop

# Id ranges bracketed by shops verified to exist: 52_635_036 through
# 555_954_448. Sampling in bands rather than one flat range keeps the draw from
# skewing towards whichever era of shop happens to be most numerous.
ID_BANDS: tuple[tuple[int, int], ...] = (
    (40_000_000, 120_000_000),
    (120_000_000, 300_000_000),
    (300_000_000, 620_000_000),
)

# A shop with no listings has nothing to model. Most sampled ids are these.
DEFAULT_MIN_ITEMS = 1

# Guards against sampling forever when the target is unreachable.
DEFAULT_ATTEMPT_MULTIPLIER = 60


@dataclass(frozen=True)
class DiscoveryStats:
    """What one discovery run actually did.

    `inactive` is reported rather than hidden: it is the cost of sampling, and
    a sudden change in the active share is the first sign that the filter or
    the id space assumption has drifted.
    """

    candidates: int = 0
    resolved: int = 0
    stored: int = 0
    inactive: int = 0
    errors: int = 0

    @property
    def resolve_rate(self) -> float | None:
        """Share of sampled ids that were real shops. None before any sample."""
        if self.candidates == 0:
            return None
        return self.resolved / self.candidates


def _in_bands(value: int, bands: Sequence[tuple[int, int]]) -> bool:
    return any(low <= value <= high for low, high in bands)


def iter_candidate_ids(
    rng: random.Random,
    bands: Sequence[tuple[int, int]] = ID_BANDS,
    seen: set[int] | None = None,
) -> Iterator[int]:
    """Yield shop ids drawn uniformly from `bands`, never repeating one.

    Bands are cycled rather than chosen at random so a short run still spans
    the whole id space instead of clustering in one era by chance.

    Generation stops once every id in `bands` has been offered. Without that
    bound, an exhausted band would spin on collisions forever, yielding
    nothing and starving the caller's own stopping conditions. `bands` are
    assumed not to overlap, which is what makes the capacity count exact.
    """
    if not bands:
        raise ValueError("at least one id band is required")

    capacity = sum(high - low + 1 for low, high in bands)
    drawn = {n for n in (seen or ()) if _in_bands(n, bands)}
    index = 0

    while len(drawn) < capacity:
        low, high = bands[index % len(bands)]
        index += 1
        candidate = rng.randint(low, high)
        if candidate in drawn:
            continue
        drawn.add(candidate)
        yield candidate


def shop_activity(payload: dict[str, Any]) -> int | None:
    """Listing count for a shop, or None when Shopee did not report one.

    None is deliberately not folded into 0: "did not say" and "has no items"
    are different claims, and only the second is evidence about the seller.
    """
    raw = endpoints.first_present(payload, endpoints.SHOP_FIELD_PATHS["item_count"])
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def discover(
    client: ShopeeClient,
    conn: Any,
    target: int,
    rng: random.Random | None = None,
    bands: Sequence[tuple[int, int]] = ID_BANDS,
    min_items: int = DEFAULT_MIN_ITEMS,
    max_candidates: int | None = None,
    on_shop: Callable[[int, str | None, int], None] | None = None,
) -> DiscoveryStats:
    """Sample ids until `target` active shops are stored.

    Each hit is written from the response that found it — discovery and
    collection are the same request, not two. Misses are not written to
    `fetch_log`: thousands of dead random ids would swamp the operator's retry
    list with entries that were never worth retrying.

    `RunAborted` from the client propagates untouched. Whatever has been stored
    stays stored, because each shop is committed as it is found.
    """
    if target <= 0:
        raise ValueError("target must be positive")

    rng = rng or random.Random()
    limit = max_candidates or target * DEFAULT_ATTEMPT_MULTIPLIER
    existing = {
        int(row["shop_id"])
        for row in conn.execute("SELECT shop_id FROM shops").fetchall()
    }

    candidates = resolved = stored = inactive = errors = 0

    for shop_id in iter_candidate_ids(rng, bands, seen=existing):
        if stored >= target or candidates >= limit:
            break
        candidates += 1

        try:
            payload = client.get_json(endpoints.shop_detail_url(shop_id))
        except FetchFailure as failure:
            # A random id pointing at nothing is the expected case, not an error.
            if failure.status is not FetchStatus.NOT_FOUND:
                errors += 1
            continue

        resolved += 1
        items = shop_activity(payload)
        if items is None or items < min_items:
            inactive += 1
            continue

        username = endpoints.first_present(
            payload, endpoints.SHOP_FIELD_PATHS["username"]
        )
        upsert_shop(
            conn,
            ShopRecord(
                shop_id=shop_id,
                username=username,
                raw_base=None,
                raw_detail=payload,
                fetched_at=utc_now_iso(),
            ),
        )
        record_status(conn, seed_key(shop_id, None), FetchStatus.OK, shop_id=shop_id)
        stored += 1
        if on_shop is not None:
            on_shop(shop_id, username, items)

    return DiscoveryStats(
        candidates=candidates,
        resolved=resolved,
        stored=stored,
        inactive=inactive,
        errors=errors,
    )
