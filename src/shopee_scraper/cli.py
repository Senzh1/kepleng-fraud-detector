"""Command line interface: scrape, status, label-import, export, score."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from . import endpoints, risk, sample, service, store
from .client import RunAborted, ShopeeClient
from .models import (
    FetchFailure,
    FetchStatus,
    ItemRecord,
    ShopFeatures,
    ShopRecord,
    utc_now_iso,
)
from .urls import SeedRef, load_seeds

app = typer.Typer(
    add_completion=False,
    help="Collect Shopee Indonesia store data for fraud-detection training.",
)

# Header tokens tolerated as the first line of a seeds file.
SEED_HEADERS = frozenset({"url", "url_or_id", "shop", "shop_id", "seed"})


def _echo(message: str) -> None:
    typer.echo(message)


def _share(part: int, whole: int) -> str:
    """`part` as a percentage of `whole`, or "n/a" when nothing was measured."""
    return f"{part / whole:.1%}" if whole else "n/a"


def _read_env_file(path: Path = Path(".env")) -> dict[str, str]:
    """Minimal .env reader. Avoids a dependency for four lines of parsing."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _setting(name: str, default: str = "") -> str:
    """Environment wins over .env, so a shell override is always possible."""
    return os.environ.get(name) or _read_env_file().get(name, default)


def _read_seed_lines(path: Path) -> list[str]:
    """First column of a seeds file, header skipped when present."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row and row[0].strip()]
    if not rows:
        return []
    if rows[0][0].strip().lower() in SEED_HEADERS:
        rows = rows[1:]
    return [row[0].strip() for row in rows]


@app.command()
def scrape(
    seeds: Path = typer.Option(..., "--seeds", help="CSV of shop URLs or IDs."),
    db: Path = typer.Option(Path("data/shops.db"), "--db", help="SQLite path."),
    rate: float = typer.Option(0.0, "--rate", help="Requests/sec. 0 uses .env or 1.0."),
    max_items: int = typer.Option(120, "--max-items", help="Listings per shop cap."),
    browser_fallback: bool = typer.Option(
        False, "--browser-fallback", help="Retry API failures through Playwright."
    ),
    retry_failed: bool = typer.Option(
        True, "--retry-failed/--skip-failed", help="Re-attempt non-ok seeds."
    ),
) -> None:
    """Collect shops listed in the seeds file. Safe to interrupt and re-run."""
    if not seeds.exists():
        raise typer.BadParameter(f"seeds file not found: {seeds}")

    refs, bad_lines = load_seeds(_read_seed_lines(seeds))
    for line, reason in bad_lines:
        _echo(f"  skip  {line!r}: {reason}")
    if not refs:
        _echo("No usable seeds. Nothing to do.")
        raise typer.Exit(code=1)

    effective_rate = rate or float(_setting("SHOPEE_RATE", "1.0"))
    cookie = _setting("SHOPEE_COOKIE") or None

    conn = store.connect(db)
    done = store.completed_keys(conn)

    pending = [
        ref for ref in refs if store.seed_key(ref.shop_id, ref.username) not in done
    ]
    if not retry_failed:
        pending = [ref for ref in pending if not _was_attempted(conn, ref)]

    _echo(
        f"{len(refs)} seeds, {len(refs) - len(pending)} skipped, "
        f"{len(pending)} to fetch at {effective_rate:g} req/s"
        f"{' (with session cookie)' if cookie else ' (anonymous)'}"
    )

    fallback: Any = None
    counts = {status: 0 for status in FetchStatus}

    with ShopeeClient(
        rate=effective_rate, cookie=cookie, max_items=max_items
    ) as client:
        try:
            for index, ref in enumerate(pending, start=1):
                key = store.seed_key(ref.shop_id, ref.username)
                name = ref.username or ref.shop_id
                record: ShopRecord | None = None
                items: list[ItemRecord] = []
                item_note: str | None = None

                try:
                    record, items, item_note = client.fetch_shop(ref)
                except FetchFailure as failure:
                    if browser_fallback:
                        fallback = fallback or _start_browser(cookie)
                        record, items = _retry_via_browser(fallback, ref)
                    if record is None:
                        store.record_status(conn, key, failure.status, failure.detail)
                        counts[failure.status] += 1
                        _echo(f"[{index}/{len(pending)}] {name}: {failure.detail}")
                        continue

                store.upsert_shop(conn, record)
                written = store.upsert_items(conn, items)
                store.record_status(
                    conn, key, FetchStatus.OK, item_note, shop_id=record.shop_id
                )
                counts[FetchStatus.OK] += 1
                suffix = f" — {item_note}" if item_note else ""
                _echo(f"[{index}/{len(pending)}] {name}: ok, {written} listings{suffix}")
        except RunAborted as abort:
            _echo(f"\nRun stopped: {abort}")
        except KeyboardInterrupt:
            _echo("\nInterrupted. Progress is saved; re-run to resume.")
        finally:
            if fallback is not None:
                fallback.close()

    conn.close()
    summary = ", ".join(f"{s.value}={counts[s]}" for s in FetchStatus if counts[s])
    _echo(f"\nDone. {summary or 'nothing collected'}")


def _was_attempted(conn: Any, ref: SeedRef) -> bool:
    key = store.seed_key(ref.shop_id, ref.username)
    return conn.execute(
        "SELECT 1 FROM fetch_log WHERE seed_key = ?", (key,)
    ).fetchone() is not None


def _start_browser(cookie: str | None) -> Any:
    from .browser import BrowserFetcher

    fetcher = BrowserFetcher(cookie=cookie)
    fetcher.start()
    return fetcher


def _retry_via_browser(
    fetcher: Any, ref: SeedRef
) -> tuple[ShopRecord | None, list[ItemRecord]]:
    """Best-effort browser retry. Returns `(None, [])` when it also fails."""
    try:
        shop_id = ref.shop_id
        raw_base = None
        if ref.username:
            raw_base = fetcher.get_json(endpoints.shop_base_url(ref.username))
            resolved = endpoints.first_present(
                raw_base, endpoints.SHOP_FIELD_PATHS["shop_id"]
            )
            shop_id = shop_id or (int(resolved) if resolved is not None else None)
        if shop_id is None:
            return None, []

        raw_detail = fetcher.get_json(endpoints.shop_detail_url(shop_id))
        fetched_at = utc_now_iso()
        record = ShopRecord(
            shop_id=shop_id,
            username=ref.username,
            raw_base=raw_base,
            raw_detail=raw_detail,
            fetched_at=fetched_at,
        )

        items: list[ItemRecord] = []
        payload = fetcher.get_json(endpoints.shop_items_url(shop_id, 0))
        for item in endpoints.extract_items(payload):
            item_id = endpoints.first_present(
                item, endpoints.ITEM_FIELD_PATHS["item_id"]
            )
            if item_id is not None:
                items.append(
                    ItemRecord(
                        shop_id=shop_id,
                        item_id=int(item_id),
                        raw=item,
                        fetched_at=fetched_at,
                    )
                )
        return record, items
    except (FetchFailure, RuntimeError):
        return None, []


@app.command()
def discover(
    target: int = typer.Option(200, "--target", help="Active shops to collect."),
    db: Path = typer.Option(Path("data/shops.db"), "--db", help="SQLite path."),
    rate: float = typer.Option(0.0, "--rate", help="Requests/sec. 0 uses .env or 1.0."),
    min_items: int = typer.Option(
        sample.DEFAULT_MIN_ITEMS, "--min-items", help="Listings a shop must have."
    ),
    seed: int = typer.Option(
        None,
        "--seed",
        help="RNG seed. Omit for a fresh draw; set it only to reproduce a run.",
    ),
    max_candidates: int = typer.Option(
        None, "--max-candidates", help="Stop after this many ids regardless."
    ),
) -> None:
    """Find shops by sampling Shopee's id space at random.

    Use this instead of a seeds file. A hand-written list is biased towards
    whichever large brands came to mind, and a model trained on it learns to
    recognise brands rather than fraud.

    Most sampled ids are dormant accounts, so a shop kept costs many requests.
    Run `status` for the rate this database has actually measured rather than
    planning against an estimate.

    Safe to interrupt: every shop is committed as it is found, and rejected ids
    are remembered so a later run does not pay to re-test them. Re-running
    without --seed continues from a fresh draw.
    """
    if target <= 0:
        raise typer.BadParameter("--target must be positive")

    effective_rate = rate or float(_setting("SHOPEE_RATE", "1.0"))
    cookie = _setting("SHOPEE_COOKIE") or None
    conn = store.connect(db)

    _echo(
        f"Sampling for {target} active shops (min {min_items} listings) "
        f"at {effective_rate:g} req/s"
        f"{' (with session cookie)' if cookie else ' (anonymous)'}"
    )

    found = 0

    def report(shop_id: int, username: str | None, items: int) -> None:
        nonlocal found
        found += 1
        _echo(f"[{found}/{target}] {username or shop_id}: {items} listings")

    with ShopeeClient(rate=effective_rate, cookie=cookie) as client:
        try:
            sample.discover(
                client,
                conn,
                target=target,
                rng=random.Random(seed),
                min_items=min_items,
                max_candidates=max_candidates,
                on_shop=report,
            )
        except RunAborted as abort:
            _echo(f"\nRun stopped: {abort}")
        except KeyboardInterrupt:
            _echo("\nInterrupted. Shops found so far are saved.")

    # Read back rather than using the return value: an interrupted run never
    # returns one, and reporting zeros for a run that did real work is worse
    # than no report at all.
    run = store.latest_run(conn)
    conn.close()

    if run is None:
        return

    _echo(
        f"\nDone. {run['stored']} stored from {run['candidates']} ids sampled "
        f"({run['resolved']} real, "
        f"{_share(run['resolved'], run['candidates'])} resolve rate; "
        f"{run['inactive']} had too few listings, {run['errors']} errored)."
    )
    if run["stored"] < target:
        _echo(
            "Short of target — re-run to continue. Stored shops and rejected "
            "ids are both remembered, so nothing is tested twice."
        )


@app.command()
def status(
    db: Path = typer.Option(Path("data/shops.db"), "--db", help="SQLite path."),
) -> None:
    """Show collection progress and the current retry list."""
    if not db.exists():
        raise typer.BadParameter(f"database not found: {db}")

    conn = store.connect(db)
    counts = store.status_counts(conn)

    shops = conn.execute("SELECT COUNT(*) AS n FROM shops").fetchone()["n"]
    items = conn.execute("SELECT COUNT(*) AS n FROM shop_items").fetchone()["n"]
    labeled = conn.execute(
        "SELECT COUNT(*) AS n FROM shops WHERE label IS NOT NULL"
    ).fetchone()["n"]

    _echo(f"Seeds attempted : {sum(counts.values())}")
    for name, count in sorted(counts.items()):
        _echo(f"  {name:<10} {count}")
    _echo(f"Shops stored    : {shops} ({labeled} labeled)")
    _echo(f"Listings stored : {items}")

    totals = store.run_totals(conn)
    if totals["runs"]:
        rejected = store.sampled_counts(conn)
        sampled = totals["candidates"]
        kept = totals["stored"]
        # The measured cost of collection, which is what sizing a run needs.
        # It lives here rather than in a docstring so it cannot go stale.
        _echo("\nDiscovery (measured):")
        _echo(f"  runs          {totals['runs']}")
        _echo(f"  ids sampled   {sampled}")
        _echo(
            f"  resolved      {totals['resolved']} "
            f"({_share(totals['resolved'], sampled)})"
        )
        _echo(f"  kept          {kept} ({_share(kept, sampled)})")
        _echo(
            f"  requests/shop {sampled / kept:.1f}"
            if kept
            else "  requests/shop n/a"
        )
        _echo(f"  errors        {totals['errors']}")
        _echo(
            f"  cached skips  {sum(rejected.values())} "
            f"(dead {rejected.get('not_found', 0)}, "
            f"dormant {rejected.get('inactive', 0)})"
        )

    problems = store.failures(conn)
    if problems:
        _echo("\nNeeds retry:")
        for row in problems[:20]:
            _echo(
                f"  {row['seed_key']:<24} {row['status']:<10} "
                f"attempts={row['attempts']}  {row['detail'] or ''}"
            )
        if len(problems) > 20:
            _echo(f"  ... and {len(problems) - 20} more")
    conn.close()


@app.command("label-import")
def label_import(
    db: Path = typer.Option(Path("data/shops.db"), "--db", help="SQLite path."),
    labels: Path = typer.Option(..., "--labels", help="CSV: shop_id,label[,source]"),
) -> None:
    """Attach fraud labels to shops already collected."""
    if not labels.exists():
        raise typer.BadParameter(f"labels file not found: {labels}")

    with labels.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        _echo("Labels file is empty.")
        raise typer.Exit(code=1)

    missing = {"shop_id", "label"} - set(rows[0].keys())
    if missing:
        raise typer.BadParameter(f"labels file missing columns: {sorted(missing)}")

    conn = store.connect(db)
    applied, unknown = store.import_labels(conn, rows)
    conn.close()

    _echo(f"Applied {applied} labels.")
    if unknown:
        _echo(
            f"{unknown} labels referenced shops not in the database — "
            "scrape them first, then re-run this command."
        )


def _collect_features(conn: Any, labeled_only: bool = False) -> list[ShopFeatures]:
    """Derive a feature row for every stored shop, with any label attached.

    Shared with the API so both score the same population in the same way.
    """
    return service.load_features(conn, labeled_only)


def _write_table(rows: list[dict[str, Any]], out: Path, fmt: str) -> None:
    """Write rows as jsonl or csv, creating the parent directory if needed."""
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scored_row(score: risk.RiskScore) -> dict[str, Any]:
    """One queue row, flattened for CSV. Rules are joined for readability."""
    return {
        "rank": score.rank,
        "shop_id": score.shop_id,
        "username": score.username,
        "name": score.name,
        "shop_url": endpoints.shop_page_url(score.shop_id),
        "risk_score": round(score.risk_score, 4),
        "anomaly_percentile": round(score.anomaly_percentile, 4),
        "rule_score": None if score.rule_score is None else round(score.rule_score, 4),
        "rules_fired": ";".join(score.rules_fired),
        "label": score.label,
    }


@app.command()
def export(
    db: Path = typer.Option(Path("data/shops.db"), "--db", help="SQLite path."),
    out: Path = typer.Option(Path("dataset.jsonl"), "--out", help="Output file."),
    fmt: str = typer.Option("jsonl", "--format", help="jsonl or csv."),
    labeled_only: bool = typer.Option(
        False, "--labeled-only", help="Export only labeled shops."
    ),
) -> None:
    """Derive features from stored raw data and write a training dataset."""
    if fmt not in {"jsonl", "csv"}:
        raise typer.BadParameter("--format must be jsonl or csv")
    if not db.exists():
        raise typer.BadParameter(f"database not found: {db}")

    conn = store.connect(db)
    rows = [features.model_dump() for features in _collect_features(conn, labeled_only)]
    conn.close()

    if not rows:
        _echo("Nothing to export.")
        raise typer.Exit(code=1)

    _write_table(rows, out, fmt)

    labeled = sum(1 for row in rows if row["label"] is not None)
    _echo(f"Wrote {len(rows)} rows ({labeled} labeled) to {out}")


@app.command()
def score(
    db: Path = typer.Option(Path("data/shops.db"), "--db", help="SQLite path."),
    out: Path = typer.Option(Path("risk_queue.csv"), "--out", help="Output file."),
    fmt: str = typer.Option("csv", "--format", help="csv or jsonl."),
    top: int = typer.Option(20, "--top", help="Rows to print."),
    at_k: int = typer.Option(50, "--at-k", help="k for precision@k."),
    weight: float = typer.Option(
        risk.DEFAULT_ANOMALY_WEIGHT,
        "--weight",
        help="Anomaly share of the blend, 0-1. The rest is rule score.",
    ),
) -> None:
    """Rank collected shops for fraud review, most suspicious first."""
    if fmt not in {"jsonl", "csv"}:
        raise typer.BadParameter("--format must be jsonl or csv")
    if not 0.0 <= weight <= 1.0:
        raise typer.BadParameter("--weight must be between 0 and 1")
    if not db.exists():
        raise typer.BadParameter(f"database not found: {db}")

    conn = store.connect(db)
    features = _collect_features(conn)
    conn.close()

    if not features:
        _echo("No shops collected yet. Nothing to score.")
        raise typer.Exit(code=1)

    scored = risk.score_shops(features, weight=weight)
    _write_table([_scored_row(row) for row in scored], out, fmt)

    _echo(f"Scored {len(scored)} shops -> {out}")

    # Per-rule counts rather than a single "fired something" tally: the weakest
    # rule fires on nearly every shop, so a combined count would read as though
    # the whole population were suspicious. A rule at 0% or ~100% is carrying no
    # information on this population and its threshold wants revisiting.
    fired = Counter(name for row in scored for name in row.rules_fired)
    _echo("\nRule fire rates:")
    for rule in risk.RULES:
        count = fired[rule.name]
        _echo(f"  {rule.name:<24} {count:>4}  {_share(count, len(scored)):>6}")

    _echo(f"\nTop {min(top, len(scored))} for review:")
    for row in scored[:top]:
        reasons = ", ".join(row.rules_fired) or "unusual profile only"
        _echo(f"  {row.rank:>3}. {row.shop_id:<12} {row.risk_score:.3f}  {reasons}")

    precision = risk.precision_at_k(scored, at_k)
    if precision is None:
        _echo(
            f"\nprecision@{at_k}: no labels yet. Run `review-queue`, label the "
            "sample, `label-import` it, then re-run this to measure the ranking."
        )
    else:
        _echo(f"\nprecision@{at_k}: {precision:.1%} of labeled shops in the top {at_k}")


@app.command("review-queue")
def review_queue(
    db: Path = typer.Option(Path("data/shops.db"), "--db", help="SQLite path."),
    out: Path = typer.Option(Path("review_queue.csv"), "--out", help="Output file."),
    size: int = typer.Option(100, "--size", help="Shops to review."),
    weight: float = typer.Option(
        risk.DEFAULT_ANOMALY_WEIGHT, "--weight", help="Anomaly share of the blend."
    ),
) -> None:
    """Pick shops to label by hand, spread across the whole score range.

    Sampling the top of the queue alone would measure precision there and say
    nothing about what the ranking missed. The output is a `label-import` file
    with the label column left blank for the reviewer to fill in.
    """
    if not db.exists():
        raise typer.BadParameter(f"database not found: {db}")

    conn = store.connect(db)
    features = _collect_features(conn)
    conn.close()

    if not features:
        _echo("No shops collected yet. Nothing to review.")
        raise typer.Exit(code=1)

    scored = risk.score_shops(features, weight=weight)
    picked = risk.review_sample(scored, size)

    rows = [
        {
            "shop_id": row.shop_id,
            "label": "",
            "source": "manual_review",
            "shop_url": endpoints.shop_page_url(row.shop_id),
            "rank": row.rank,
            "risk_score": round(row.risk_score, 4),
            "rules_fired": ";".join(row.rules_fired),
        }
        for row in picked
    ]
    _write_table(rows, out, "csv")

    _echo(f"Wrote {len(rows)} shops to {out}, spread across the score range.")
    _echo(
        "Open each shop_url, put 1 (fraud) or 0 (legitimate) in the label "
        f"column, then: shopee-scrape label-import --db {db} --labels {out}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
