# Shopee Indonesia Store Scraper — Design

**Date:** 2026-08-11
**Status:** Approved, pending implementation plan

## Purpose

Collect store-level data from Shopee Indonesia (`shopee.co.id`) to build a training
dataset for a fraud-detection model. The unit of analysis is the **store**: one row
per shop, labeled fraud / not-fraud.

Seeds are supplied by the operator as a list of shop URLs or IDs. The tool does not
crawl or discover shops on its own.

## Scope

**In scope**

- Resolve any Shopee shop or product URL form to a canonical `shop_id`
- Fetch shop profile data and the shop's item list
- Persist raw API responses alongside derived features in SQLite
- Derive store-level features relevant to fraud signals
- Export a training-ready JSONL / CSV dataset
- Attach labels from an operator-supplied CSV

**Out of scope**

- Shop discovery by keyword or category search
- CAPTCHA solving, browser fingerprint spoofing, proxy rotation, or any other
  block-evasion technique
- Purchasing, messaging, or any authenticated write action
- Model training itself — this tool produces the dataset, nothing more

## Operating constraints

Shopee's Terms of Service prohibit automated collection, and their endpoints are
actively defended. This tool is built to be polite and visible rather than evasive:

- Default rate limit of 1 request/second with jitter, operator-configurable
- Exponential backoff on 429 and 403, with a hard stop after repeated blocks
- A single identifying User-Agent, not a rotating pool
- `robots.txt` is fetched and logged at startup for operator awareness
- Optional session cookies read from a local `.env`, never committed

If Shopee blocks the collector, the correct response is to slow down or stop — not
to disguise the traffic. The tool is designed to degrade gracefully and report what
it could not collect.

## Architecture

```
seeds.csv ──> urls.resolve ──> client.fetch ──> store.upsert ──> features.derive ──> export
                                    │
                                    └── browser.fetch (fallback on API failure)
```

### Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `urls.py` | Parse Shopee URL forms into `(shop_id, username)`. Pure functions, no I/O. | — |
| `endpoints.py` | Every endpoint URL, required header, and response field path. Single source of truth. | — |
| `client.py` | Rate-limited httpx client. Retry, backoff, optional cookies. Returns raw JSON or a typed failure. | `endpoints` |
| `browser.py` | Playwright fallback fetcher. Same interface as `client`. | `endpoints` |
| `store.py` | SQLite schema, upserts, resume state. Owns all SQL. | `models` |
| `models.py` | Pydantic models for shop, item, and derived feature rows. | — |
| `features.py` | Raw JSON to feature row. Pure functions, no I/O. | `models` |
| `cli.py` | Typer commands: `scrape`, `export`, `label-import`, `status`. | all |

The boundary that matters most is `endpoints.py`. Shopee's internal API is
undocumented and changes without notice. Isolating every URL and field path there
means an API change is a one-file fix, and `browser.py` keeps collection running
while that fix is made.

`urls.py` and `features.py` are pure and fully testable without network or database.

## Data model

### `shops`

Canonical row per shop.

| Column | Type | Notes |
|---|---|---|
| `shop_id` | INTEGER PRIMARY KEY | Canonical Shopee shop ID |
| `username` | TEXT | Shop username slug |
| `raw_base` | TEXT | Raw JSON from shop base endpoint |
| `raw_detail` | TEXT | Raw JSON from shop detail endpoint |
| `fetched_at` | TEXT | ISO 8601 UTC, `Z` suffix |
| `label` | INTEGER NULL | 1 fraud, 0 legitimate, NULL unlabeled |
| `label_source` | TEXT NULL | Where the label came from |

Synthetic example row:

```
shop_id      = 123456789
username     = "contoh_toko"
raw_base     = "{...}"
raw_detail   = "{...}"
fetched_at   = "2026-08-11T04:22:31Z"
label        = NULL
label_source = NULL
```

### `shop_items`

Listings belonging to a shop. Used for aggregate features, not labeled directly.

| Column | Type | Notes |
|---|---|---|
| `item_id` | INTEGER | |
| `shop_id` | INTEGER | FK to `shops` |
| `raw` | TEXT | Raw listing JSON |
| `fetched_at` | TEXT | ISO 8601 UTC, `Z` suffix |

Primary key `(shop_id, item_id)`.

### `fetch_log`

Drives resume and reports failures.

| Column | Type | Notes |
|---|---|---|
| `seed_key` | TEXT PRIMARY KEY | `"<shop_id>"`, or `"u:<username>"` before resolution |
| `shop_id` | INTEGER NULL | Backfilled once the numeric id is known |
| `status` | TEXT | `ok`, `blocked`, `not_found`, `error` |
| `detail` | TEXT NULL | Error message or HTTP status |
| `attempts` | INTEGER | Retry counter |
| `updated_at` | TEXT | ISO 8601 UTC, `Z` suffix |

**Deviation from the approved design, made during implementation:** this table
was specified as keyed on `shop_id`. A username-only seed has no numeric id
until its first successful fetch, so keying on `shop_id` would have made those
seeds unresumable — every interrupted run would re-fetch them. Keying on a
stable `seed_key` fixes that while `shop_id` is still recorded once known.

**Raw JSON is retained deliberately.** Feature definitions will change repeatedly
during model development. Re-deriving features from stored raw responses takes
seconds; re-scraping takes days and risks the collector's access.

## Derived features

Computed by `features.py` from stored raw JSON. All are store-level.

**Account maturity**
- `shop_age_days` — from `ctime`
- `rating_velocity` — total ratings divided by shop age in days

**Reputation**
- `rating_count_total`
- `rating_bad`, `rating_normal`, `rating_good` — Shopee's rating buckets, where
  `rating_bad` already means 1★ and 2★
- `rating_star_1` … `rating_star_5` — raw counts per star
- `bad_rating_ratio` — (1★ + 2★) / total
- `response_rate`, `response_time_seconds`

**Deviation from the approved design, found during implementation:** the design
assumed a per-star breakdown at shop level. Shopee's shop endpoints do not
expose one — they report the three buckets above. The per-star columns are kept
in the schema and the field maps still look for them, so they populate
automatically if Shopee ever exposes them, but they are `NULL` today.
`bad_rating_ratio` prefers a genuine per-star breakdown when present and falls
back to `rating_bad / total`, so the feature is correct either way.

**Scale and standing**
- `follower_count`, `item_count`
- `is_official_shop`, `is_preferred_seller`
- `location` — shop-declared city or province

**Catalog signals**
- `price_median`, `price_dispersion` — coefficient of variation across listings
- `zero_stock_ratio` — share of listings with no stock
- `review_to_sold_ratio` — aggregate reviews divided by aggregate sold count

Features are computed defensively: a missing upstream field yields `NULL` for that
feature, never a crash and never a silently imputed zero. A `NULL` means "not
observed" and must stay distinguishable from a real zero during training.

## Error handling

Failure is expected and is not exceptional. The run continues.

- **HTTP 429 / 403** — back off exponentially, retry up to the configured limit,
  then mark `blocked` and move on
- **HTTP 404 or empty payload** — mark `not_found`, no retry
- **Schema mismatch** (endpoint changed) — store the raw response anyway, mark
  `error` with the parse failure, continue. Raw data is preserved for re-parsing
  once `endpoints.py` is corrected.
- **Network timeout** — retry, then mark `error`

A batch of 500 shops with 30 blocks yields 470 usable rows plus an actionable
retry list. `status` reports the breakdown; re-running `scrape` retries only rows
whose status is not `ok`.

## CLI

```
shopee-scrape scrape --seeds seeds.csv --db data/shops.db [--rate 1.0] [--browser-fallback]
shopee-scrape status --db data/shops.db
shopee-scrape label-import --db data/shops.db --labels labels.csv
shopee-scrape export --db data/shops.db --out dataset.jsonl [--format jsonl|csv]
```

`seeds.csv` accepts a single `url_or_id` column holding any of: a full shop URL, a
full product URL, a `shop/<id>` path, or a bare numeric shop ID.

`labels.csv` requires `shop_id` and `label` columns, with an optional `source`.

## Testing

Target 80% coverage, enforced on `urls.py` and `features.py` specifically since
they carry the parsing and computation logic.

- **Unit** — `urls.py` against every known URL form plus malformed input;
  `features.py` against fixture JSON including responses with missing fields
- **Integration** — `store.py` against a temporary SQLite file: upsert idempotency,
  resume correctness, label import
- **Contract** — recorded Shopee JSON fixtures parsed through `endpoints.py` field
  maps, so an endpoint change fails a test rather than corrupting a dataset

- **CLI** — every command driven through `typer.testing.CliRunner` with the HTTP
  client replaced by a fake: first run, resumed run, partial failure, labelling,
  and both export formats

No test touches the live network. Live connectivity is a separate opt-in smoke
check, run manually, excluded from CI.

**As built:** 87 tests, 83% overall coverage. `urls.py` 98%, `features.py` 92%,
`store.py` 100%, `endpoints.py` 97%, `client.py` 89%, `cli.py` 80%.
`browser.py` reports 0% because Playwright is an optional extra and is not
installed in the default dev environment.

## Technology

Python 3.11+, `httpx`, `playwright`, `pydantic`, `typer`, `pytest`.
SQLite via the standard library.

## Risks

| Risk | Mitigation |
|---|---|
| Shopee changes its internal API | All endpoint knowledge isolated in `endpoints.py`; contract tests catch it; browser fallback continues collection |
| Collector gets IP-blocked | Conservative default rate, backoff, hard stop. Resume means no lost work. |
| Label imbalance in training set | Out of scope for this tool, but raw retention means the dataset can be re-derived and re-balanced without re-scraping |
| Features silently degrade as fields disappear | Missing fields yield `NULL`, never imputed zeros; `status` surfaces parse errors |
