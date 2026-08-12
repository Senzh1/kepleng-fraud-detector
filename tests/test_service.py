"""Single-shop scoring and the HTTP layer over it.

Both live here because the API is a thin translation of the service: testing
them side by side keeps the contract between them visible in one file.

The case that matters most is the fallback. The proof-of-work video forbids
cuts, so a live fetch failing mid-demo must degrade to the stored snapshot
rather than showing an error the recording cannot be edited to remove.
"""

from __future__ import annotations

import pytest
from conftest import load_fixture
from fastapi.testclient import TestClient
from test_risk import make_shop

from shopee_scraper import service
from shopee_scraper.api import create_app
from shopee_scraper.models import FetchFailure, FetchStatus, ShopRecord
from shopee_scraper.service import ShopScorer, ShopUnavailable
from shopee_scraper.urls import InvalidSeedError


class Fetched:
    """Stands in for client.ShopFetch."""

    def __init__(self, shop_id: int) -> None:
        self.record = ShopRecord(
            shop_id=shop_id,
            username="live_shop",
            raw_detail=load_fixture("shop_detail.json"),
            fetched_at="2026-08-11T04:22:31Z",
        )
        self.items: list = []
        self.note = None


class LiveFetcher:
    def fetch_shop(self, ref):
        return Fetched(ref.shop_id or 123456789)


class BlockedFetcher:
    def fetch_shop(self, ref):
        raise FetchFailure(FetchStatus.BLOCKED, "HTTP 429")


def population(size: int = 12) -> list:
    """A reference population with one obvious outlier."""
    shops = [make_shop(100 + n, username=f"shop{n}") for n in range(size - 1)]
    shops.append(
        make_shop(
            999,
            username="suspicious",
            shop_age_days=3.0,
            item_count=900,
            rating_count_total=0,
        )
    )
    return shops


@pytest.fixture
def scorer() -> ShopScorer:
    return ShopScorer(population())


class TestBands:
    def test_high_score_reads_as_high(self) -> None:
        assert service.band_for(0.9) == "high"

    def test_middling_score_reads_as_elevated(self) -> None:
        assert service.band_for(0.5) == "elevated"

    def test_low_score_reads_as_low(self) -> None:
        assert service.band_for(0.1) == "low"


class TestScoring:
    def test_scores_a_stored_shop_by_id(self, scorer: ShopScorer) -> None:
        verdict = scorer.score_url("999")

        assert verdict.shop_id == 999
        assert verdict.source == "stored"

    def test_scores_a_stored_shop_by_url(self, scorer: ShopScorer) -> None:
        verdict = scorer.score_url("https://shopee.co.id/shop/100")

        assert verdict.shop_id == 100

    def test_the_outlier_outscores_an_ordinary_shop(self, scorer: ShopScorer) -> None:
        assert scorer.score_url("999").risk_score > scorer.score_url("100").risk_score

    def test_explains_every_verdict(self, scorer: ShopScorer) -> None:
        """A score with no reason attached is one the user must trust blindly."""
        assert scorer.score_url("999").reasons

    def test_an_unremarkable_shop_still_gets_a_reason(self, scorer: ShopScorer) -> None:
        reasons = scorer.score_url("100").reasons

        assert any("collected shops" in reason for reason in reasons)

    def test_rejects_a_shop_it_has_never_seen(self, scorer: ShopScorer) -> None:
        with pytest.raises(ShopUnavailable):
            scorer.score_url("555555")

    def test_rejects_input_that_is_not_a_shop(self, scorer: ShopScorer) -> None:
        with pytest.raises(InvalidSeedError):
            scorer.score_url("https://tokopedia.com/whatever")

    def test_refuses_to_fit_on_nothing(self) -> None:
        with pytest.raises(ValueError):
            ShopScorer([])


class TestLiveFetch:
    def test_prefers_a_live_fetch_when_one_is_offered(self, scorer: ShopScorer) -> None:
        verdict = scorer.score_url("123456789", fetcher=LiveFetcher())

        assert verdict.source == "live"

    def test_falls_back_to_the_stored_snapshot_when_shopee_blocks(
        self, scorer: ShopScorer
    ) -> None:
        """The demo must survive a hostile network — the video cannot be cut."""
        verdict = scorer.score_url("999", fetcher=BlockedFetcher())

        assert verdict.source == "stored"
        assert "live fetch failed" in verdict.note

    def test_reports_failure_when_there_is_nothing_to_fall_back_to(
        self, scorer: ShopScorer
    ) -> None:
        with pytest.raises(ShopUnavailable):
            scorer.score_url("555555", fetcher=BlockedFetcher())


class TestVerdictShape:
    def test_rounds_to_what_is_worth_showing(self, scorer: ShopScorer) -> None:
        payload = scorer.score_url("999").as_dict()

        assert payload["risk_score"] == round(payload["risk_score"], 3)
        assert payload["band"] in {"high", "elevated", "low"}
        assert isinstance(payload["reasons"], list)


class TestApi:
    @pytest.fixture
    def client(self, scorer: ShopScorer):
        with TestClient(create_app(scorer)) as client:
            yield client

    def test_serves_the_page(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "Shopee Seller Risk" in response.text

    def test_health_reports_the_reference_population(self, client: TestClient) -> None:
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["population_size"] == 12
        assert body["live_fetch"] is False

    def test_scores_a_shop(self, client: TestClient) -> None:
        response = client.post("/api/score", json={"query": "999"})

        assert response.status_code == 200
        assert response.json()["shop_id"] == 999
        assert response.json()["reasons"]

    def test_rejects_a_url_that_is_not_shopee(self, client: TestClient) -> None:
        """A bare word is a valid username, so this needs a real foreign host."""
        response = client.post(
            "/api/score", json={"query": "https://tokopedia.com/whatever"}
        )

        assert response.status_code == 400

    def test_reports_an_uncollected_shop_as_not_found(self, client: TestClient) -> None:
        response = client.post("/api/score", json={"query": "555555"})

        assert response.status_code == 404

    def test_rejects_an_empty_query(self, client: TestClient) -> None:
        response = client.post("/api/score", json={"query": ""})

        assert response.status_code == 422
