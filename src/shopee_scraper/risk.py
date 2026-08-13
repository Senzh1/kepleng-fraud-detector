"""Unsupervised risk scoring for collected shops.

Why unsupervised: the collected shops carry no labels, and uniform id sampling —
the thing that makes the dataset representative — also makes fraud rare in it.
Labelling every shop by hand would surface only a handful of positives, far
below what a classifier can learn from. Ranking shops for human review needs no
labels at all, and the labels that review produces are what makes supervised
learning possible later.

Why not anomaly detection alone: an outlier is not a fraudster. The largest and
newest shops are unusual and mostly legitimate, so a pure Isolation Forest
ranking surfaces oddity rather than risk. The score below blends the detector
with explicit rules that encode what marketplace fraud actually looks like, so
the ranking reflects a prior about fraud instead of a prior about rarity.

Pure functions, no I/O, in the style of `features.py`: this module can be
rewritten and re-run over an existing database without re-scraping.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from .models import ShopFeatures

# Features with enough coverage across collected shops to be worth modelling.
# Excluded deliberately:
#   - rating_star_1..5, which Shopee's shop endpoints never populate
#   - price_*, zero_stock_ratio, review_to_sold_ratio, items_observed, which
#     need listings that anonymous collection cannot reach
#   - rating_bad/normal/good, which sum to rating_count_total and would weight
#     the reputation dimension three times over
# An all-null column adds a dimension carrying no information, which only
# dilutes the splits the detector makes on columns that do.
MODEL_FEATURES: tuple[str, ...] = (
    "shop_age_days",
    "rating_velocity",
    "rating_star",
    "rating_count_total",
    "bad_rating_ratio",
    "response_rate",
    "response_time_seconds",
    "follower_count",
    "item_count",
    "days_since_active",
    "is_official_shop",
    "is_preferred_seller",
    "is_shopee_verified",
)

# `cancellation_rate` is withheld until its unit is confirmed. Live responses
# show values of 0.0 and 1.0 alongside sibling thresholds that look like
# percentages, so 1.0 may mean 1% rather than 100%. Feeding it in at the wrong
# scale would rank the worst sellers as the safest. Move it into MODEL_FEATURES
# once a shop with a known cancellation history settles the question.
UNVERIFIED_FEATURES: tuple[str, ...] = ("cancellation_rate",)

# Counts spanning several orders of magnitude across real shops. Logging them
# keeps a 500k-follower outlier from dominating any later distance-based model,
# and makes the exported numbers readable as magnitudes.
LOG_FEATURES = frozenset(
    {
        "shop_age_days",
        "rating_count_total",
        "response_time_seconds",
        "follower_count",
        "item_count",
        "days_since_active",
    }
)

# Features whose absence is itself evidence: an unrated shop is not a shop with
# a rating of zero. Imputing the median would erase that distinction, so each
# imputed value is paired with a flag saying it was imputed.
MISSINGNESS_FEATURES: tuple[str, ...] = (
    "rating_star",
    "bad_rating_ratio",
    "response_rate",
    "response_time_seconds",
)

# Rule thresholds. Named rather than inline so a reviewer can argue with the
# numbers, which is the point of having rules at all.
NEW_SHOP_DAYS = 90.0
LARGE_CATALOG_ITEMS = 100
UNRATED_CATALOG_ITEMS = 20
NEGATIVE_RATING_SHARE = 0.20
MIN_RATINGS_TO_JUDGE = 10
# Percent, not a fraction: Shopee reports `response_rate` on a 0-100 scale
# (observed across 500 collected shops: min 0, median 89, max 100, always whole
# numbers). Written as 0.30 this rule fired only on a literal zero, passing
# sellers who answer 20-25% of buyers as responsive.
LOW_RESPONSE_RATE = 30.0
DORMANT_DAYS = 30.0

# Half anomaly, half rules. Neither signal is trustworthy alone: the detector
# finds unusual shops with no notion of fraud, the rules encode fraud with no
# notion of what is normal here.
DEFAULT_ANOMALY_WEIGHT = 0.5

DEFAULT_RANDOM_STATE = 0


@dataclass(frozen=True)
class FeatureMatrix:
    """A dense, imputed numeric matrix plus the ids of the rows in it.

    Dense because Isolation Forest cannot consume nulls. The information the
    imputation would destroy is preserved in the `missing_*` columns rather
    than thrown away.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    shop_ids: tuple[int, ...]
    # Kept so a shop arriving later can be imputed against the same population
    # the detector was fitted on. Re-deriving them from one shop is impossible.
    medians: dict[str, float]


@dataclass(frozen=True)
class Rule:
    """One interpretable risk heuristic.

    `test` returns None when the shop lacks the fields to judge it. That is not
    the same as returning False: a rule that cannot be evaluated must not count
    as evidence of innocence, so it is excluded from the denominator instead.
    """

    name: str
    weight: float
    explanation: str
    test: Callable[[ShopFeatures], bool | None]


class RiskScore(BaseModel):
    """One shop's place in the review queue."""

    shop_id: int
    username: str | None = None
    name: str | None = None
    risk_score: float
    # 0 when the shop was scored on its own rather than as part of a queue.
    rank: int = 0
    anomaly_percentile: float = 0.0
    rule_score: float | None = None
    rules_fired: tuple[str, ...] = Field(default_factory=tuple)
    label: int | None = None


def _numeric(value: Any) -> float | None:
    """Coerce a feature value to float, keeping None as None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _transform(name: str, value: float | None) -> float | None:
    """Apply the column's transform. log1p guards against negative inputs."""
    if value is None:
        return None
    if name in LOG_FEATURES:
        return math.log1p(value) if value > -1 else None
    return value


FEATURE_COLUMNS: tuple[str, ...] = MODEL_FEATURES + tuple(
    f"missing_{name}" for name in MISSINGNESS_FEATURES
)


def column_medians(features: Sequence[ShopFeatures]) -> dict[str, float]:
    """Median of each column across observed values only.

    Shops that never reported a field are excluded rather than counted as
    zero, so a mostly-unrated population does not drag every median down.
    A column with no observed value anywhere falls back to 0.0 — it carries no
    signal either way, and its `missing_*` flag carries whatever is left.
    """
    medians: dict[str, float] = {}
    for name in MODEL_FEATURES:
        values = [
            value
            for value in (
                _transform(name, _numeric(getattr(shop, name, None)))
                for shop in features
            )
            if value is not None
        ]
        medians[name] = statistics.median(values) if values else 0.0
    return medians


def build_row(shop: ShopFeatures, medians: dict[str, float]) -> tuple[float, ...]:
    """One dense row: imputed values followed by their missingness flags.

    Taking medians as an argument is what lets a shop scored months later be
    imputed against the population the detector was fitted on, rather than
    against itself — a single shop has no median of its own.
    """
    row = [
        medians.get(name, 0.0)
        if (value := _transform(name, _numeric(getattr(shop, name, None)))) is None
        else value
        for name in MODEL_FEATURES
    ]
    row.extend(
        1.0 if getattr(shop, name, None) is None else 0.0
        for name in MISSINGNESS_FEATURES
    )
    return tuple(row)


def build_matrix(features: Sequence[ShopFeatures]) -> FeatureMatrix:
    """Turn feature rows into a dense matrix with missingness indicators."""
    if not features:
        raise ValueError("cannot build a matrix from zero shops")

    medians = column_medians(features)
    return FeatureMatrix(
        columns=FEATURE_COLUMNS,
        rows=tuple(build_row(shop, medians) for shop in features),
        shop_ids=tuple(shop.shop_id for shop in features),
        medians=medians,
    )


def _new_shop_large_catalog(shop: ShopFeatures) -> bool | None:
    """A brand-new account already carrying a large catalog.

    Legitimate catalogs of this size take time to build. Bulk-listed
    storefronts do not.
    """
    if shop.shop_age_days is None or shop.item_count is None:
        return None
    return shop.shop_age_days < NEW_SHOP_DAYS and shop.item_count >= LARGE_CATALOG_ITEMS


def _unrated_at_scale(shop: ShopFeatures) -> bool | None:
    """Selling at volume with no reputation to show for it."""
    if shop.rating_count_total is None or shop.item_count is None:
        return None
    return shop.rating_count_total == 0 and shop.item_count >= UNRATED_CATALOG_ITEMS


def _negative_reputation(shop: ShopFeatures) -> bool | None:
    """A meaningful share of bad ratings, on enough ratings to mean something.

    The rating count guard matters: one complaint out of two is noise, and
    without it every barely-rated shop would rank as high risk.
    """
    if shop.bad_rating_ratio is None or shop.rating_count_total is None:
        return None
    if shop.rating_count_total < MIN_RATINGS_TO_JUDGE:
        return False
    return shop.bad_rating_ratio >= NEGATIVE_RATING_SHARE


def _unresponsive_seller(shop: ShopFeatures) -> bool | None:
    """Buyers cannot get an answer — common when nobody is behind the shop."""
    if shop.response_rate is None:
        return None
    return shop.response_rate < LOW_RESPONSE_RATE


def _dormant_with_stock(shop: ShopFeatures) -> bool | None:
    """Listings still up long after the seller stopped showing up."""
    if shop.days_since_active is None or shop.item_count is None:
        return None
    return shop.days_since_active >= DORMANT_DAYS and shop.item_count > 0


def _no_platform_standing(shop: ShopFeatures) -> bool | None:
    """No badge of any kind.

    A weak prior, deliberately weighted low: most ordinary sellers hold no
    badge either. It earns its place only because badged shops are almost never
    the ones under review.
    """
    badges = (
        shop.is_official_shop,
        shop.is_preferred_seller,
        shop.is_shopee_verified,
    )
    if all(badge is None for badge in badges):
        return None
    return not any(badge for badge in badges)


# Ordered by how much each pattern should move a shop up the queue. Every rule
# is per-shop and self-contained: anything needing peer context is the
# detector's job, not a rule's.
RULES: tuple[Rule, ...] = (
    Rule(
        "negative_reputation",
        3.0,
        "high share of bad ratings on enough ratings to judge",
        _negative_reputation,
    ),
    Rule(
        "new_shop_large_catalog",
        2.0,
        "large catalog on a very new account",
        _new_shop_large_catalog,
    ),
    Rule(
        "unrated_at_scale",
        2.0,
        "selling at volume with no ratings at all",
        _unrated_at_scale,
    ),
    Rule("unresponsive_seller", 1.0, "rarely answers buyers", _unresponsive_seller),
    Rule(
        "dormant_with_stock",
        1.0,
        "listings up but seller inactive",
        _dormant_with_stock,
    ),
    Rule("no_platform_standing", 0.5, "holds no platform badge", _no_platform_standing),
)


def rules_fired(shop: ShopFeatures) -> tuple[str, ...]:
    """Names of the rules that fired, for explaining a score to a reviewer."""
    return tuple(rule.name for rule in RULES if rule.test(shop) is True)


def rule_score(shop: ShopFeatures) -> float | None:
    """Fired weight as a share of evaluable weight, or None if nothing applies.

    Normalising by what could be evaluated — rather than by the total weight of
    every rule — stops a sparsely-populated shop from looking safe purely
    because most rules had nothing to read.
    """
    evaluable = 0.0
    fired = 0.0
    for rule in RULES:
        verdict = rule.test(shop)
        if verdict is None:
            continue
        evaluable += rule.weight
        if verdict:
            fired += rule.weight
    return None if evaluable == 0 else fired / evaluable


def _percentiles(values: Sequence[float]) -> list[float]:
    """Rank-normalise to [0, 1], averaging ranks across ties.

    Rank rather than min-max because Isolation Forest scores are unitless and
    unevenly spread; ranks are what make them comparable with a rule score.
    """
    count = len(values)
    if count == 1:
        return [0.5]

    order = sorted(range(count), key=lambda i: values[i])
    result = [0.0] * count
    position = 0
    while position < count:
        end = position
        while end + 1 < count and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 / (count - 1)
        for index in range(position, end + 1):
            result[order[index]] = shared
        position = end + 1
    return result


def _fit_forest(rows: Sequence[Sequence[float]], random_state: int) -> Any:
    """Fit the detector on a population.

    scikit-learn is imported here rather than at module load so that collecting
    data never depends on the modelling extra being installed.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "scoring needs scikit-learn: python -m pip install -e '.[model]'"
        ) from error

    forest = IsolationForest(random_state=random_state, contamination="auto")
    forest.fit(rows)
    return forest


def _raw_anomaly(forest: Any, rows: Sequence[Sequence[float]]) -> list[float]:
    """Unusualness, larger meaning stranger.

    `score_samples` returns higher values for more *normal* points, so the sign
    is flipped to stop every downstream comparison reading backwards.
    """
    return [-value for value in forest.score_samples(rows)]


def anomaly_percentiles(
    matrix: FeatureMatrix, random_state: int = DEFAULT_RANDOM_STATE
) -> list[float]:
    """How unusual each shop is relative to the rest, in [0, 1]."""
    forest = _fit_forest(matrix.rows, random_state)
    return _percentiles(_raw_anomaly(forest, matrix.rows))


def blend(
    anomaly_percentile: float,
    rule_score: float | None,
    weight: float = DEFAULT_ANOMALY_WEIGHT,
) -> float:
    """Combine the two signals into one score in [0, 1].

    A shop with no evaluable rule falls back to its anomaly percentile alone.
    Treating an unmeasurable rule score as 0.0 would rank exactly the shops we
    know least about as the safest.
    """
    if rule_score is None:
        return anomaly_percentile
    return weight * anomaly_percentile + (1.0 - weight) * rule_score


@dataclass(frozen=True)
class RiskModel:
    """A detector fitted on a reference population, ready to judge new shops.

    The app needs this shape rather than `score_shops`: a shop pasted into the
    UI is a single row, and a single row has no notion of unusual. Judging it
    means holding on to what the detector learned — the fitted forest, the
    imputation medians, and the population's own anomaly scores, which are what
    turn one raw number into a percentile.

    Fitting is the slow part and happens once at startup; `score` is a single
    forest traversal, which is what keeps the request synchronous.
    """

    forest: Any
    medians: dict[str, float]
    reference: tuple[float, ...]
    weight: float = DEFAULT_ANOMALY_WEIGHT

    @classmethod
    def fit(
        cls,
        population: Sequence[ShopFeatures],
        weight: float = DEFAULT_ANOMALY_WEIGHT,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> "RiskModel":
        """Learn what normal looks like from the shops already collected."""
        matrix = build_matrix(population)
        forest = _fit_forest(matrix.rows, random_state)
        return cls(
            forest=forest,
            medians=matrix.medians,
            reference=tuple(sorted(_raw_anomaly(forest, matrix.rows))),
            weight=weight,
        )

    def score(self, shop: ShopFeatures) -> RiskScore:
        """Judge one shop against the reference population."""
        row = build_row(shop, self.medians)
        raw = _raw_anomaly(self.forest, [row])[0]
        # Share of the reference population this shop is stranger than.
        percentile = bisect.bisect_left(self.reference, raw) / len(self.reference)
        rules = rule_score(shop)
        return RiskScore(
            shop_id=shop.shop_id,
            username=shop.username,
            name=shop.name,
            risk_score=blend(percentile, rules, self.weight),
            anomaly_percentile=percentile,
            rule_score=rules,
            rules_fired=rules_fired(shop),
            label=shop.label,
        )


def score_shops(
    features: Sequence[ShopFeatures],
    weight: float = DEFAULT_ANOMALY_WEIGHT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[RiskScore]:
    """Rank shops for human review, most suspicious first.

    Deterministic given the same population, so a queue can be regenerated and
    compared across runs.
    """
    if not features:
        return []

    matrix = build_matrix(features)
    percentiles = anomaly_percentiles(matrix, random_state=random_state)

    scored = []
    for shop, percentile in zip(features, percentiles):
        rules = rule_score(shop)
        scored.append(
            RiskScore(
                shop_id=shop.shop_id,
                username=shop.username,
                name=shop.name,
                risk_score=blend(percentile, rules, weight),
                rank=0,
                anomaly_percentile=percentile,
                rule_score=rules,
                rules_fired=rules_fired(shop),
                label=shop.label,
            )
        )

    # Ties broken by shop_id so the order never depends on input ordering.
    scored.sort(key=lambda row: (-row.risk_score, row.shop_id))
    return [
        row.model_copy(update={"rank": index}) for index, row in enumerate(scored, 1)
    ]


def precision_at_k(scored: Sequence[RiskScore], k: int) -> float | None:
    """Share of the labelled shops in the top `k` that are actually fraud.

    Unlabelled shops are skipped, not counted as negatives: with most of the
    queue unreviewed, counting them would report a precision near zero no
    matter how good the ranking is. Returns None when the top `k` holds no
    labels at all, since there is then nothing to measure.
    """
    head = list(scored)[:k]
    labelled = [row.label for row in head if row.label is not None]
    if not labelled:
        return None
    return sum(1 for label in labelled if label == 1) / len(labelled)


def review_sample(scored: Sequence[RiskScore], size: int) -> list[RiskScore]:
    """Pick shops to label, spread evenly across the score range.

    Labelling only the top of the queue measures precision there and tells you
    nothing about what the ranking missed. An even spread costs the same review
    effort and supports both questions.
    """
    population = sorted(scored, key=lambda row: (-row.risk_score, row.shop_id))
    count = len(population)
    size = min(size, count)
    if size <= 0:
        return []

    step = count / size
    picked: list[RiskScore] = []
    used: set[int] = set()
    for slot in range(size):
        index = min(count - 1, int(slot * step + step / 2))
        while index in used:
            index = (index + 1) % count
        used.add(index)
        picked.append(population[index])
    return picked
