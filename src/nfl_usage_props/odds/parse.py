"""Parse The Odds API event-odds payloads into flat rows.

Deliberately dumb and total: it extracts what the payload contains and does not
resolve players, compute vig, or filter anything. Every price from every book,
both sides, with timestamps.

Both sides matter and are the reason this stores rows rather than a wide table:
devigging needs the Over and Under of the same player at the same line from the
same book, and hold cannot be computed from one side alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

PROP_ROW_SCHEMA = {
    "snapshot_id": pl.Utf8,
    "snapshot_kind": pl.Utf8,
    "fetched_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "event_id": pl.Utf8,
    "commence_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "bookmaker": pl.Utf8,
    "bookmaker_title": pl.Utf8,
    "market": pl.Utf8,
    "market_last_update": pl.Datetime(time_unit="us", time_zone="UTC"),
    "player_name_raw": pl.Utf8,
    "side": pl.Utf8,
    "point": pl.Float64,
    "price": pl.Int64,
    # Filled by a later resolution pass. Nullable by design: the snapshot must
    # land on disk whether or not the resolver succeeds.
    "gsis_id": pl.Utf8,
}


@dataclass
class PropRow:
    snapshot_id: str
    snapshot_kind: str
    fetched_at: datetime
    event_id: str
    commence_time: datetime | None
    home_team: str | None
    away_team: str | None
    bookmaker: str
    bookmaker_title: str | None
    market: str
    market_last_update: datetime | None
    player_name_raw: str
    side: str
    point: float | None
    price: int | None
    gsis_id: str | None = None


def parse_iso(value: Any) -> datetime | None:
    """Parse an Odds API timestamp. They are RFC3339 with a trailing Z."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_event_odds(
    payload: dict[str, Any],
    *,
    snapshot_id: str,
    snapshot_kind: str,
    fetched_at: datetime | None = None,
    markets: list[str] | None = None,
) -> list[PropRow]:
    """Flatten one event-odds response into rows.

    Outcomes carry the player in `description` and Over/Under in `name`. An
    outcome with no `description` is not a player prop (it is a team market
    that slipped into the payload) and is skipped.
    """
    fetched_at = fetched_at or datetime.now(UTC)
    wanted = set(markets) if markets else None

    event_id = payload.get("id") or ""
    commence = parse_iso(payload.get("commence_time"))
    home = payload.get("home_team")
    away = payload.get("away_team")

    rows: list[PropRow] = []
    for book in payload.get("bookmakers") or []:
        book_key = book.get("key") or ""
        book_title = book.get("title")
        for market in book.get("markets") or []:
            market_key = market.get("key") or ""
            if wanted is not None and market_key not in wanted:
                continue
            last_update = parse_iso(market.get("last_update")) or parse_iso(book.get("last_update"))
            for outcome in market.get("outcomes") or []:
                player = outcome.get("description")
                if not player:
                    continue
                price = outcome.get("price")
                rows.append(
                    PropRow(
                        snapshot_id=snapshot_id,
                        snapshot_kind=snapshot_kind,
                        fetched_at=fetched_at,
                        event_id=event_id,
                        commence_time=commence,
                        home_team=home,
                        away_team=away,
                        bookmaker=book_key,
                        bookmaker_title=book_title,
                        market=market_key,
                        market_last_update=last_update,
                        player_name_raw=player,
                        side=(outcome.get("name") or "").strip(),
                        point=outcome.get("point"),
                        price=int(price) if isinstance(price, int | float) else None,
                    )
                )
    return rows


def rows_to_frame(rows: list[PropRow]) -> pl.DataFrame:
    """Build a frame with a stable schema, even when `rows` is empty."""
    if not rows:
        return pl.DataFrame(schema=PROP_ROW_SCHEMA)
    return pl.DataFrame([asdict(r) for r in rows], schema=PROP_ROW_SCHEMA)


def american_to_implied(price: int | float | None) -> float | None:
    """American odds to implied probability, vig included."""
    if price is None:
        return None
    price = float(price)
    if price == 0:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def two_way_hold(over_price: int | float | None, under_price: int | float | None) -> float | None:
    """Bookmaker hold on a two-way market.

    Receptions and rush-attempt props run 6-10% hold against ~4.5% on sides,
    which is most of the reason these markets are hard to beat. Surfacing it on
    every logged prop keeps that fact visible instead of buried.
    """
    over = american_to_implied(over_price)
    under = american_to_implied(under_price)
    if over is None or under is None:
        return None
    return over + under - 1.0


def add_hold(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach two-way hold per (event, book, market, player, point).

    Pairs Over with Under. Rows whose opposite side is missing from the payload
    get a null hold rather than being dropped -- a one-sided quote is still a
    real observation worth keeping.
    """
    if frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("hold"))

    keys = ["event_id", "bookmaker", "market", "player_name_raw", "point"]
    sides = (
        frame.with_columns(pl.col("side").str.to_lowercase().alias("_side"))
        .group_by(keys)
        .agg(
            pl.col("price").filter(pl.col("_side") == "over").first().alias("_over"),
            pl.col("price").filter(pl.col("_side") == "under").first().alias("_under"),
        )
    )
    sides = sides.with_columns(
        pl.struct(["_over", "_under"])
        .map_elements(
            lambda s: two_way_hold(s["_over"], s["_under"]),
            return_dtype=pl.Float64,
        )
        .alias("hold")
    ).drop("_over", "_under")

    return frame.join(sides, on=keys, how="left")
