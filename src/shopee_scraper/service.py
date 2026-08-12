"""Scoring one shop on demand — the path the app exercises.

The CLI scores a whole population at once. An app cannot: a shop pasted into
the box is a single row, and a single row has no notion of unusual. So the
reference population is loaded and the detector fitted once at startup, and
each request is one forest traversal against it. That is what keeps the
interaction synchronous, with no queue and no background job.

Every shop is judged from two independent angles — how unusual it is against
the collected population, and which known fraud patterns it matches. Neither
is trustworthy alone, which is why the verdict carries both.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import risk, store
from .features import derive_features
from .models import FetchFailure, ShopFeatures
from .urls import SeedRef, resolve_seed

# Where the score stops being a number and starts being advice. Bands exist
# because "0.63" means nothing to a seller or a buyer; "elevated" does.
HIGH_RISK = 0.66
ELEVATED_RISK = 0.40


class ShopUnavailable(Exception):
    """The shop could not be fetched live and is not in the local collection."""


class ShopFetcher(Protocol):
    """The slice of ShopeeClient this module needs.

    Narrow on purpose: the service depends on the one method it calls, so a
    test can substitute a fake without standing up an HTTP client.
    """

    def fetch_shop(self, ref: SeedRef) -> Any: ...


@dataclass(frozen=True)
class Verdict:
    """One shop's risk assessment, in the shape the UI renders.

    `source` is part of the verdict rather than a footnote: a score from a
    stored snapshot and one from a live fetch deserve different confidence, and
    hiding which happened would be dishonest.
    """

    shop_id: int
    username: str | None
    name: str | None
    risk_score: float
    band: str
    anomaly_percentile: float
    rule_score: float | None
    reasons: tuple[str, ...]
    source: str
    population_size: int
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Flat JSON for the API, rounded to what is meaningful to show."""
        return {
            "shop_id": self.shop_id,
            "username": self.username,
            "name": self.name,
            "risk_score": round(self.risk_score, 3),
            "band": self.band,
            "anomaly_percentile": round(self.anomaly_percentile, 3),
            "rule_score": (
                None if self.rule_score is None else round(self.rule_score, 3)
            ),
            "reasons": list(self.reasons),
            "source": self.source,
            "population_size": self.population_size,
            "note": self.note,
        }


def load_features(
    conn: sqlite3.Connection, labeled_only: bool = False
) -> list[ShopFeatures]:
    """Derive a feature row for every stored shop, with any label attached."""
    labels = store.shop_labels(conn)
    rows: list[ShopFeatures] = []
    for record, items in store.iter_shops(conn):
        label, source = labels.get(record.shop_id, (None, None))
        if labeled_only and label is None:
            continue
        features = derive_features(record, items)
        features.label = label
        features.label_source = source
        rows.append(features)
    return rows


def band_for(score: float) -> str:
    """Turn a score into the word a non-technical user can act on."""
    if score >= HIGH_RISK:
        return "high"
    if score >= ELEVATED_RISK:
        return "elevated"
    return "low"


def explain(scored: risk.RiskScore, population_size: int) -> tuple[str, ...]:
    """Plain-language reasons behind a score.

    A score with no reasons attached is a number the user has to trust blindly.
    Naming the rules that fired makes the verdict arguable — and a verdict a
    seller can argue with is one a marketplace can actually act on.
    """
    explanations = {rule.name: rule.explanation for rule in risk.RULES}
    reasons = [explanations[name] for name in scored.rules_fired]
    if not reasons:
        reasons.append(
            "no known fraud pattern matched; ranked on how unusual its profile "
            f"is against {population_size} collected shops"
        )
    return tuple(reasons)


class ShopScorer:
    """Holds the fitted model and the collection it was fitted on.

    Built once and reused: fitting is the slow half, and refitting per request
    would turn a millisecond lookup into a second of work for a result that
    would not change.
    """

    def __init__(
        self,
        population: list[ShopFeatures],
        weight: float = risk.DEFAULT_ANOMALY_WEIGHT,
    ) -> None:
        if not population:
            raise ValueError("cannot fit a model with no collected shops")
        self.model = risk.RiskModel.fit(population, weight=weight)
        self.population_size = len(population)
        self._by_id = {shop.shop_id: shop for shop in population}
        self._by_username = {
            shop.username.lower(): shop for shop in population if shop.username
        }

    @classmethod
    def from_db(
        cls, path: str | Path, weight: float = risk.DEFAULT_ANOMALY_WEIGHT
    ) -> "ShopScorer":
        """Fit against everything collected so far."""
        conn = store.connect(path)
        try:
            return cls(load_features(conn), weight=weight)
        finally:
            conn.close()

    def stored(self, ref: SeedRef) -> ShopFeatures | None:
        """The collected snapshot for a seed, by id or by username."""
        if ref.shop_id is not None and ref.shop_id in self._by_id:
            return self._by_id[ref.shop_id]
        if ref.username:
            return self._by_username.get(ref.username.lower())
        return None

    def score_url(self, raw: str, fetcher: ShopFetcher | None = None) -> Verdict:
        """Score whatever a user pasted: a URL, a username, or a bare id.

        A live fetch is preferred when a fetcher is supplied, but Shopee
        blocking or timing out falls back to the stored snapshot rather than
        failing outright. The demo has to survive a hostile network, and the
        fallback is reported in `source` rather than passed off as fresh.
        """
        ref = resolve_seed(raw)

        if fetcher is not None:
            try:
                fetched = fetcher.fetch_shop(ref)
            except FetchFailure as failure:
                snapshot = self.stored(ref)
                if snapshot is None:
                    raise ShopUnavailable(failure.detail) from failure
                return self._verdict(
                    snapshot,
                    source="stored",
                    note=(
                        f"live fetch failed ({failure.detail}); scored the "
                        "stored snapshot instead"
                    ),
                )
            features = derive_features(fetched.record, fetched.items)
            return self._verdict(features, source="live", note=fetched.note)

        snapshot = self.stored(ref)
        if snapshot is None:
            raise ShopUnavailable(
                "shop is not in the local collection; run with a live fetcher "
                "to score shops that have not been collected"
            )
        return self._verdict(snapshot, source="stored")

    def _verdict(
        self, features: ShopFeatures, source: str, note: str | None = None
    ) -> Verdict:
        scored = self.model.score(features)
        return Verdict(
            shop_id=scored.shop_id,
            username=scored.username,
            name=scored.name,
            risk_score=scored.risk_score,
            band=band_for(scored.risk_score),
            anomaly_percentile=scored.anomaly_percentile,
            rule_score=scored.rule_score,
            reasons=explain(scored, self.population_size),
            source=source,
            population_size=self.population_size,
            note=note,
        )
