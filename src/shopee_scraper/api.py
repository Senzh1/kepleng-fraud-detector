"""HTTP layer: one endpoint that scores one shop.

Deliberately thin. Every decision about *how* a shop is judged lives in
`risk.py` and `service.py`; this module only translates HTTP into a call and a
verdict back into JSON. Keeping it that way is what lets the same model run
from the CLI, the API, or a notebook without being reimplemented.

The model is fitted once during startup rather than per request. A request
therefore does no training and no background work — it resolves a URL, scores
one row, and returns.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .client import ShopeeClient
from .service import ShopScorer, ShopUnavailable
from .urls import InvalidSeedError

WEB_ROOT = Path(__file__).parent / "web"

DEFAULT_DB = "data/shops.db"


def _setting(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _live_fetcher() -> ShopeeClient | None:
    """A live client, only when the operator explicitly asked for one.

    Off by default. A judge running `docker compose up` should get a working
    demo without a Shopee session, without waiting on rate limits, and without
    generating traffic they did not ask for.
    """
    if _setting("SHOPEE_LIVE", "0") not in {"1", "true", "yes"}:
        return None
    return ShopeeClient(
        rate=float(_setting("SHOPEE_RATE", "1.0")),
        cookie=_setting("SHOPEE_COOKIE") or None,
    )


class ScoreRequest(BaseModel):
    """The single input the UI collects."""

    query: str = Field(
        min_length=1,
        description="Shopee shop URL, product URL, username, or numeric shop id",
    )


def create_app(scorer: ShopScorer | None = None) -> FastAPI:
    """Build the app, optionally against an already-fitted scorer.

    The injection point exists for tests: fitting from a database at import
    time would make every test depend on a collected corpus.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = (
            ShopScorer.from_db(_setting("SHOPEE_DB", DEFAULT_DB))
            if scorer is None
            else scorer
        )
        app.state.fetcher = _live_fetcher()
        yield
        fetcher = getattr(app.state, "fetcher", None)
        if fetcher is not None:
            fetcher.close()

    app = FastAPI(
        title="Shopee Seller Risk",
        description="Unsupervised fraud risk scoring for Shopee Indonesia sellers.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """The single page. Served from disk so the UI stays editable."""
        return (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Readiness plus the size of the population the model compares against."""
        return {
            "status": "ok",
            "population_size": app.state.scorer.population_size,
            "live_fetch": app.state.fetcher is not None,
        }

    @app.post("/api/score")
    def score(request: ScoreRequest) -> dict[str, Any]:
        """Score one shop and explain the verdict."""
        try:
            verdict = app.state.scorer.score_url(
                request.query, fetcher=app.state.fetcher
            )
        except InvalidSeedError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ShopUnavailable as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return verdict.as_dict()

    return app


app = create_app()
