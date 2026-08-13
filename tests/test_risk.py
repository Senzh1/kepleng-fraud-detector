"""Unsupervised risk scoring.

The fixtures deliberately include shops with missing reputation fields: over
half the collected shops have never been rated, and a scorer that crashes or
silently imputes zero for them would be useless on the real database.
"""

from __future__ import annotations

import pytest

from shopee_scraper import risk
from shopee_scraper.models import ShopFeatures


def make_shop(shop_id: int = 1, **overrides) -> ShopFeatures:
    """A plausible mid-sized shop, overridable field by field."""
    defaults = dict(
        shop_age_days=800.0,
        rating_velocity=0.5,
        rating_star=4.8,
        rating_count_total=400,
        bad_rating_ratio=0.02,
        # Percent, matching what Shopee reports and what the fixtures carry.
        # Written as 0.9 this read as a 0.9% responder, which is why the
        # threshold drifted to a fraction without any test objecting.
        response_rate=90.0,
        response_time_seconds=1800.0,
        follower_count=1200,
        item_count=60,
        is_official_shop=False,
        is_preferred_seller=False,
        is_shopee_verified=True,
        days_since_active=1.0,
    )
    defaults.update(overrides)
    return ShopFeatures(shop_id=shop_id, **defaults)


class TestFeatureMatrix:
    def test_includes_every_model_feature_as_a_column(self) -> None:
        matrix = risk.build_matrix([make_shop()])

        for name in risk.MODEL_FEATURES:
            assert name in matrix.columns

    def test_adds_a_missingness_indicator_for_sparse_features(self) -> None:
        matrix = risk.build_matrix([make_shop()])

        assert "missing_response_rate" in matrix.columns

    def test_missingness_indicator_marks_absent_values(self) -> None:
        # Arrange: one rated shop, one never rated.
        shops = [make_shop(1), make_shop(2, rating_star=None)]

        # Act
        matrix = risk.build_matrix(shops)
        column = matrix.columns.index("missing_rating_star")

        # Assert
        assert matrix.rows[0][column] == 0.0
        assert matrix.rows[1][column] == 1.0

    def test_imputes_the_median_of_observed_values(self) -> None:
        shops = [
            make_shop(1, bad_rating_ratio=0.10),
            make_shop(2, bad_rating_ratio=0.30),
            make_shop(3, bad_rating_ratio=None),
        ]

        matrix = risk.build_matrix(shops)
        column = matrix.columns.index("bad_rating_ratio")

        assert matrix.rows[2][column] == pytest.approx(0.20)

    def test_imputation_never_leaves_a_none_in_the_matrix(self) -> None:
        """An all-null column still has to yield a number sklearn can consume."""
        shops = [make_shop(1, response_rate=None), make_shop(2, response_rate=None)]

        matrix = risk.build_matrix(shops)

        assert all(isinstance(value, float) for row in matrix.rows for value in row)

    def test_log_transforms_heavy_tailed_counts(self) -> None:
        """Follower counts span several orders of magnitude across real shops."""
        matrix = risk.build_matrix([make_shop(follower_count=999)])
        column = matrix.columns.index("follower_count")

        assert matrix.rows[0][column] == pytest.approx(6.907755, abs=1e-5)

    def test_booleans_become_numbers(self) -> None:
        matrix = risk.build_matrix([make_shop(is_shopee_verified=True)])
        column = matrix.columns.index("is_shopee_verified")

        assert matrix.rows[0][column] == 1.0

    def test_preserves_shop_id_order(self) -> None:
        matrix = risk.build_matrix([make_shop(7), make_shop(3)])

        assert matrix.shop_ids == (7, 3)

    def test_rejects_an_empty_population(self) -> None:
        with pytest.raises(ValueError):
            risk.build_matrix([])


class TestRules:
    def test_new_shop_with_large_catalog_fires(self) -> None:
        shop = make_shop(shop_age_days=10.0, item_count=500)

        fired = risk.rules_fired(shop)

        assert "new_shop_large_catalog" in fired

    def test_established_shop_with_large_catalog_does_not_fire(self) -> None:
        shop = make_shop(shop_age_days=2000.0, item_count=500)

        assert "new_shop_large_catalog" not in risk.rules_fired(shop)

    def test_negative_reputation_needs_enough_ratings_to_be_meaningful(self) -> None:
        """One bad rating out of two is noise, not evidence."""
        noisy = make_shop(bad_rating_ratio=0.5, rating_count_total=2)

        assert "negative_reputation" not in risk.rules_fired(noisy)

    def test_negative_reputation_fires_on_a_rated_shop(self) -> None:
        shop = make_shop(bad_rating_ratio=0.4, rating_count_total=50)

        assert "negative_reputation" in risk.rules_fired(shop)

    def test_rule_score_is_the_fired_share_of_evaluable_weight(self) -> None:
        clean = make_shop()

        assert risk.rule_score(clean) == pytest.approx(0.0)

    def test_rule_score_ignores_rules_it_cannot_evaluate(self) -> None:
        """A missing input must not count as evidence of innocence."""
        blind = make_shop(
            item_count=None,
            shop_age_days=None,
            rating_count_total=None,
            bad_rating_ratio=None,
            response_rate=None,
            days_since_active=None,
        )

        # Only the badge rule survives, and it does not fire on a verified shop.
        assert risk.rule_score(blind) == pytest.approx(0.0)

    def test_rule_score_is_none_when_nothing_can_be_evaluated(self) -> None:
        blank = ShopFeatures(shop_id=1)

        assert risk.rule_score(blank) is None

    def test_a_suspicious_shop_outscores_a_clean_one(self) -> None:
        suspicious = make_shop(
            shop_age_days=5.0,
            item_count=800,
            rating_count_total=0,
            bad_rating_ratio=None,
            response_rate=5.0,
            is_shopee_verified=False,
        )

        assert risk.rule_score(suspicious) > risk.rule_score(make_shop())

    def test_unresponsive_seller_reads_response_rate_as_a_percentage(self) -> None:
        """The threshold is 30%, not 0.3%.

        Shopee reports `response_rate` on a 0-100 scale. Compared against a
        fractional threshold the rule fired only on a literal zero, so sellers
        answering a quarter of their buyers passed as responsive — the rule
        looked alive in the fire-rate report while missing everyone it exists
        to catch.
        """
        quarter = make_shop(response_rate=25.0)

        assert "unresponsive_seller" in risk.rules_fired(quarter)
        assert "unresponsive_seller" not in risk.rules_fired(make_shop())


class TestBlend:
    def test_weights_anomaly_and_rules(self) -> None:
        blended = risk.blend(anomaly_percentile=1.0, rule_score=0.0, weight=0.5)

        assert blended == pytest.approx(0.5)

    def test_falls_back_to_the_anomaly_score_when_no_rule_applies(self) -> None:
        """Dropping to zero would rank an unmeasurable shop as safest."""
        assert risk.blend(anomaly_percentile=0.8, rule_score=None) == pytest.approx(0.8)


class TestScoring:
    def test_scores_every_shop(self) -> None:
        shops = [make_shop(n) for n in range(1, 21)]

        scored = risk.score_shops(shops)

        assert len(scored) == 20

    def test_returns_shops_ranked_most_risky_first(self) -> None:
        shops = [make_shop(n) for n in range(1, 20)]
        shops.append(
            make_shop(99, shop_age_days=2.0, item_count=900, rating_count_total=0)
        )

        scored = risk.score_shops(shops)

        assert scored[0].shop_id == 99
        assert scored[0].rank == 1

    def test_is_deterministic(self) -> None:
        shops = [make_shop(n, follower_count=n * 13) for n in range(1, 31)]

        first = [s.shop_id for s in risk.score_shops(shops)]
        second = [s.shop_id for s in risk.score_shops(shops)]

        assert first == second

    def test_carries_labels_through_for_evaluation(self) -> None:
        shops = [make_shop(n) for n in range(1, 21)]
        shops[0].label = 1

        scored = risk.score_shops(shops)

        assert any(s.label == 1 for s in scored)


class TestPrecisionAtK:
    def test_measures_the_positive_share_of_labeled_shops_in_the_top_k(self) -> None:
        scored = [
            risk.RiskScore(shop_id=1, risk_score=0.9, rank=1, label=1),
            risk.RiskScore(shop_id=2, risk_score=0.8, rank=2, label=0),
            risk.RiskScore(shop_id=3, risk_score=0.1, rank=3, label=1),
        ]

        assert risk.precision_at_k(scored, k=2) == pytest.approx(0.5)

    def test_ignores_unlabeled_shops_rather_than_counting_them_as_negative(
        self,
    ) -> None:
        scored = [
            risk.RiskScore(shop_id=1, risk_score=0.9, rank=1, label=1),
            risk.RiskScore(shop_id=2, risk_score=0.8, rank=2, label=None),
        ]

        assert risk.precision_at_k(scored, k=2) == pytest.approx(1.0)

    def test_is_none_when_the_top_k_holds_no_labels(self) -> None:
        scored = [risk.RiskScore(shop_id=1, risk_score=0.9, rank=1, label=None)]

        assert risk.precision_at_k(scored, k=1) is None


class TestReviewSample:
    def test_spreads_the_sample_across_the_score_range(self) -> None:
        """Sampling the top only would measure precision and nothing else."""
        scored = [
            risk.RiskScore(shop_id=n, risk_score=n / 100, rank=101 - n)
            for n in range(1, 101)
        ]

        picked = risk.review_sample(scored, size=10)
        scores = [s.risk_score for s in picked]

        assert len(picked) == 10
        assert max(scores) > 0.8
        assert min(scores) < 0.2

    def test_never_returns_the_same_shop_twice(self) -> None:
        scored = [
            risk.RiskScore(shop_id=n, risk_score=n / 20, rank=21 - n)
            for n in range(1, 21)
        ]

        picked = risk.review_sample(scored, size=15)

        assert len({s.shop_id for s in picked}) == len(picked)

    def test_caps_at_the_population_size(self) -> None:
        scored = [risk.RiskScore(shop_id=n, risk_score=0.5, rank=n) for n in range(1, 6)]

        assert len(risk.review_sample(scored, size=50)) == 5
