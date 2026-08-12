# Shopee Indonesia Store Scraper

Scores Shopee Indonesia sellers for fraud risk. Paste a shop link, get a risk
band and the reasons behind it.

Two halves: a **collector** that builds the dataset by uniformly sampling
Shopee's shop-id space, and an **unsupervised risk model** that ranks sellers
without needing labelled fraud cases.

## Quick start

```bash
docker compose up --build
```

Then open <http://localhost:8000> and paste a Shopee shop URL.

The app runs in **demo mode** by default: it scores against the collected
snapshot in `data/shops.db` rather than calling Shopee, so it works with no
session cookie and generates no traffic. Set `SHOPEE_LIVE=1` (with
`SHOPEE_COOKIE`) in `docker-compose.yml` to score shops that have not been
collected yet; a live fetch that fails falls back to the stored snapshot and
says so in the verdict.

Without Docker:

```powershell
python -m pip install -e ".[model,app]"
uvicorn shopee_scraper.api:app --reload
```

| Layer | Where | Responsibility |
|---|---|---|
| AI | `risk.py`, `features.py` | Feature matrix, fraud rules, anomaly blend. Pure functions |
| Backend | `api.py`, `service.py` | One synchronous endpoint, `POST /api/score` |
| Frontend | `web/index.html` | One input, one verdict |

The collector below builds the dataset the model is fitted on.

## Before you use this

Shopee's Terms of Service prohibit automated collection, and their endpoints are
actively defended. This collector is deliberately polite rather than evasive:

- Rate limited to 1 request/second by default, with jitter
- Backs off on 429/403 and **stops the run entirely** after 5 consecutive blocks
- Sends one consistent User-Agent — no rotating pool, no proxy rotation
- No CAPTCHA solving and no fingerprint spoofing

If Shopee blocks you, slow down or stop. Do not try to disguise the traffic.
Whether you may collect this data is your call to make, not this tool's.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Optional browser fallback (~20x slower per shop, survives API changes):

```powershell
python -m pip install -e ".[browser]"
playwright install chromium
```

## Use

### 1. List the shops you want

`seeds.csv`, one per line under a `url_or_id` header. Every one of these forms
works:

```csv
url_or_id
https://shopee.co.id/shop/123456789
https://shopee.co.id/contoh_toko
https://shopee.co.id/Sepatu-Pria-Keren-i.123456789.987654321
987654321
```

A header is optional. Extra columns are ignored. Lines that can't be parsed are
reported and skipped, never fatal.

### 2. Collect

```powershell
shopee-scrape scrape --seeds seeds.csv --db data/shops.db
```

Safe to interrupt with Ctrl+C and re-run — progress is committed per shop, and a
re-run skips anything already collected while retrying anything that failed.

Useful flags: `--rate 0.5` (slower), `--max-items 60` (fewer listings per shop),
`--browser-fallback`, `--skip-failed`.

### 3. Check progress

```powershell
shopee-scrape status --db data/shops.db
```

Shows the ok / blocked / not_found / error breakdown and a retry list.

### 4. Attach labels

`labels.csv`:

```csv
shop_id,label,source
123456789,1,manual_review
987654321,0,manual_review
```

`label` is 1 for fraud, 0 for legitimate.

```powershell
shopee-scrape label-import --db data/shops.db --labels labels.csv
```

Labels for shops you haven't scraped are reported, not silently inserted, so a
typo surfaces instead of creating a ghost row. Re-scraping never overwrites a
label you've already attached.

### 5. Export a dataset

```powershell
shopee-scrape export --db data/shops.db --out dataset.jsonl
shopee-scrape export --db data/shops.db --out dataset.csv --format csv --labeled-only
```

### 6. Rank shops for review

Needs the modelling extra: `python -m pip install -e ".[model]"`.

```powershell
shopee-scrape score --db data/shops.db --out risk_queue.csv
```

Ranks every collected shop most-suspicious-first, blending an Isolation Forest
anomaly score with explicit fraud rules. **Unsupervised on purpose** — it needs
no labels, which is what makes it usable before any labelling has happened.

The blend matters. An outlier is not a fraudster: the largest and newest shops
are unusual and mostly legitimate, so a pure anomaly ranking surfaces oddity
rather than risk. `--weight` sets the anomaly share (default `0.5`); the rest is
the rule score.

Output includes per-rule fire rates. A rule firing on 0% or ~100% of shops is
carrying no information on your population, and its threshold wants revisiting.

To measure the ranking, label a sample spread across the whole score range:

```powershell
shopee-scrape review-queue --db data/shops.db --out review_queue.csv --size 100
```

```csv
shop_id,label,source,shop_url,rank,risk_score,rules_fired
121814757,,manual_review,https://shopee.co.id/shop/121814757,3,0.7447,negative_reputation;dormant_with_stock
```

Open each `shop_url`, put 1 or 0 in the blank `label` column, then feed the same
file straight back to `label-import`. Re-running `score` afterwards reports
precision@k — the share of *labelled* shops in the top k that are genuinely
fraud. Unlabelled shops are skipped rather than counted as negatives, since
with most of the queue unreviewed that would report near-zero precision no
matter how good the ranking is.

Sampling the top of the queue alone would measure precision there and tell you
nothing about what the ranking missed, which is why `review-queue` spreads.

## Session cookies — required for listings

**Verified against production 2026-08:**

| Data | Anonymous | Needs cookie |
|---|---|---|
| Shop profile (ratings, age, followers, cancellation rate, location) | works | — |
| Shop listings (prices, stock, sold counts) | **403** | yes |

Every catalog endpoint (`search_items`, `rcmd_items`, `get_shop_tab`) returns
HTTP 403 with `{"error": 90309999, "is_login": false}` to anonymous callers.
Running without cookies gives you every profile-level feature and
`items_observed = 0`, and the run says so explicitly per shop.

The catalog features — `price_median`, `price_dispersion`, `zero_stock_ratio`,
`review_to_sold_ratio` — are the ones you lose without a session.

Copy `.env.example` to `.env` and paste the cookie header from a logged-in
browser session:

```
SHOPEE_COOKIE=SPC_F=...; SPC_EC=...
SHOPEE_RATE=1.0
```

`.env` is gitignored. Never commit it. Environment variables override it.

## Features produced

| Feature | Meaning |
|---|---|
| `shop_age_days` | Days since the shop was created |
| `rating_velocity` | Ratings per day since creation |
| `rating_star` | Average star rating |
| `rating_count_total` | Total ratings |
| `rating_bad` / `normal` / `good` | Shopee's rating buckets (bad = 1–2★) |
| `bad_rating_ratio` | Negative share of all ratings |
| `response_rate`, `response_time_seconds` | Seller responsiveness |
| `cancellation_rate` | Seller-attributed order cancellations — **unit unconfirmed, see below** |
| `follower_count`, `item_count` | Shop scale |
| `is_official_shop`, `is_preferred_seller`, `is_shopee_verified` | Platform badges |
| `days_since_active` | Days since the shop was last active |
| `location` | Shop-declared city or province |
| `items_observed` | Listings actually collected (0 without a session cookie) |
| `price_median` | Median listing price, IDR |
| `price_dispersion` | Coefficient of variation across listing prices |
| `zero_stock_ratio` | Share of listings with no stock |
| `review_to_sold_ratio` | Aggregate reviews over aggregate units sold |
| `label`, `label_source` | Your supervision |

`rating_star_1` … `rating_star_5` exist in the schema but are `null` today:
Shopee's shop endpoints report bucketed ratings, not a per-star breakdown. The
field maps pick them up automatically if that ever changes.

### `cancellation_rate` — verify the unit before using it

Observed live: two shops reported `0.0`, one reported `1.0`. The sibling fields
`cancellation_visibility: 10` and `cancellation_warning: 20` look like
percentage thresholds, which suggests `1.0` means **1%**, not 100%. This is
inferred, not confirmed. Check it against a shop with a known cancellation
history before treating the column as a 0–1 ratio — a wrong scale here would
make your worst-performing sellers look like your best.

### Training note

**A `null` means "not observed" and is not the same as zero.** A shop with no
ratings has `bad_rating_ratio = null`, not `0.0` — treating it as zero would tell
your model that a brand-new shop has a perfect complaint record. Encode
missingness explicitly rather than calling `fillna(0)`.

## How it's put together

```
seeds.csv ──> urls.resolve ──> client.fetch ──> store.upsert ──> features.derive ──> export
                                    │
                                    └── browser.fetch (fallback on API failure)
```

| Module | Responsibility |
|---|---|
| `urls.py` | Parse Shopee URL forms into `(shop_id, username)` |
| `endpoints.py` | Every endpoint URL and response field path |
| `client.py` | Rate-limited httpx client with retry and backoff |
| `browser.py` | Playwright fallback |
| `store.py` | SQLite schema, upserts, resume state |
| `models.py` | Typed records |
| `features.py` | Raw JSON to feature row |
| `risk.py` | Feature matrix, fraud rules, anomaly blend, review queue |
| `cli.py` | Commands |

Raw API responses are kept in SQLite alongside the derived data. Feature
definitions will change while you develop the model; re-deriving from stored raw
takes seconds, re-scraping takes days.

## When Shopee changes their API

They will, without notice. Everything you need to fix lives in `endpoints.py`.
The contract tests in `tests/test_endpoints.py` fail loudly when a field moves,
rather than letting the dataset quietly fill with nulls.

Symptoms and fixes:

| Symptom | Likely cause |
|---|---|
| All features null, shops still "ok" | Field paths moved — update `SHOP_FIELD_PATHS` |
| Everything `blocked` immediately | Rate too high, or anonymous access refused — lower `--rate`, add cookies |
| `response was not JSON` | Anti-bot interstitial — you are being challenged; stop and wait |
| `items_observed` always 0 | `search_items` shape changed — update `ITEMS_LIST_PATHS` |

## Tests

```powershell
python -m pytest
python -m pytest --cov=shopee_scraper --cov-report=term-missing
```

92 tests, no network access. Shopee responses are recorded fixtures under
`tests/fixtures/`, so the suite stays green when Shopee is down or blocking you.
`browser.py` is only covered when the optional Playwright extra is installed.
