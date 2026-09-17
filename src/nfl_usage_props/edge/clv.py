"""Stage 6: scoring the model against closing lines.

This is the only validation this project will ever have, and it does not exist
yet -- not because the code is missing but because the data is. Historical
props are paid-tier, so closing line value cannot be backfilled at any price on
the free tier. It accumulates from the moment the snapshot logger starts and
not one week earlier.

What is being measured
----------------------
Whether flagged bets beat the closing line. If the model says Over 4.5 at -115
on Wednesday and the market closes -135, that is +CLV: the market moved toward
the model. CLV is the standard proxy for edge because it is available in weeks
rather than seasons -- results need hundreds of bets before variance settles,
while CLV is informative after a few dozen.

It is a proxy, not a proof. A model can beat the close on markets nobody can
get money down on, and CLV on a stale book measures the book's slowness, not
the model's insight. Both caveats are why the report shows the opening book
alongside the number.

What this module deliberately does NOT do
-----------------------------------------
Report a win rate or an ROI. With no historical prop prices there is no
backtest, and presenting a small live sample as a track record would be the
single most misleading thing this project could produce. Results scoring lands
only once there are results.
"""

from __future__ import annotations

import polars as pl

from nfl_usage_props.edge.devig import american_to_probability

# Enough paired open/close observations for the median to mean anything. Below
# this the report says "not yet", which is the honest answer for a while.
MIN_OBSERVATIONS = 30


def pair_open_and_close(snapshots: pl.DataFrame) -> pl.DataFrame:
    """Match each opening price to the last price seen before kickoff.

    Keyed on (event, market, player, line, book, side): a price is only
    comparable to a later price on the *same* line at the same book. A book
    moving 4.5 to 5.5 has not moved the price, it has replaced the market, and
    treating that as line movement would manufacture CLV out of a different bet.
    """
    key = ["event_id", "market", "gsis_id", "point", "bookmaker", "side"]
    available = [c for c in key if c in snapshots.columns]
    ordered = snapshots.filter(pl.col("price").is_not_null()).sort("fetched_at")

    opens = (
        ordered.filter(pl.col("snapshot_kind") == "open")
        .group_by(available)
        .agg(pl.col("price").first().alias("open_price"), pl.col("fetched_at").first())
    )
    closes = (
        ordered.filter(pl.col("snapshot_kind") == "close")
        .group_by(available)
        .agg(pl.col("price").last().alias("close_price"))
    )
    return opens.join(closes, on=available, how="inner")


def closing_line_value(paired: pl.DataFrame) -> pl.DataFrame:
    """CLV in probability points, per paired price.

    Expressed as a probability difference rather than in cents, because cents
    are not comparable across price levels: twenty cents at -110 and twenty
    cents at -300 are very different amounts of edge.
    """
    if paired.is_empty():
        return paired.with_columns(pl.lit(None, dtype=pl.Float64).alias("clv"))

    open_probability = [american_to_probability(p) for p in paired["open_price"]]
    close_probability = [american_to_probability(p) for p in paired["close_price"]]
    return paired.with_columns(
        pl.Series("open_probability", open_probability),
        pl.Series("close_probability", close_probability),
    ).with_columns(
        # Positive means the market moved TOWARD the side that was taken --
        # the price got worse for anyone betting it later, which is what
        # beating the close means.
        (pl.col("close_probability") - pl.col("open_probability")).alias("clv")
    )


def score_flagged_bets(edges: pl.DataFrame, snapshots: pl.DataFrame) -> pl.DataFrame:
    """CLV restricted to the rows the model actually flagged.

    CLV across every price a book posted measures the market, not the model.
    Only the flagged side of the flagged markets says anything about whether
    the disagreement was informed.
    """
    if edges.is_empty() or snapshots.is_empty():
        return pl.DataFrame()

    flagged = edges.filter(pl.col("actionable")).select(
        "event_id", "market", "gsis_id", "point", "side", "edge", "model_probability"
    )
    if flagged.is_empty():
        return pl.DataFrame()

    paired = closing_line_value(pair_open_and_close(snapshots))
    if paired.is_empty():
        return pl.DataFrame()

    on = [c for c in ("event_id", "market", "gsis_id", "point", "side") if c in paired.columns]
    return flagged.join(paired, on=on, how="inner")


def summarise(scored: pl.DataFrame) -> dict[str, float | int | str]:
    """Headline CLV numbers, or an honest refusal when there is too little.

    The median rather than the mean: CLV is heavy-tailed, and one market that
    moved thirty cents should not be able to carry a whole week's summary.
    """
    if scored.is_empty():
        return {"status": "no scored bets yet", "n": 0}
    if scored.height < MIN_OBSERVATIONS:
        return {
            "status": (
                f"only {scored.height} scored bets; "
                f"{MIN_OBSERVATIONS} is the floor for this to mean anything"
            ),
            "n": scored.height,
        }

    clv = scored["clv"]
    return {
        "status": "ok",
        "n": scored.height,
        "median_clv": float(clv.median()),
        "mean_clv": float(clv.mean()),
        "share_positive": float((clv > 0).mean()),
    }


# --------------------------------------------------------------- settled bets


def settle(edges: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Grade flagged markets against what actually happened.

    Only possible once games are played, and only for the weeks a snapshot
    covered -- but when it is possible it beats CLV outright. CLV is a proxy
    for edge; this is the thing itself.

    Pushes are dropped rather than counted as losses. A whole-number line that
    lands exactly refunds, and scoring it as a loss would understate the model
    by however many integer lines the books happened to post.
    """
    if edges.is_empty() or panel.is_empty():
        return pl.DataFrame()

    outcomes = panel.select(
        "gsis_id",
        "game_id",
        pl.col("receptions").alias("actual_receptions"),
        pl.col("carries").alias("actual_carries"),
    )
    graded = edges.join(outcomes, on=["gsis_id", "game_id"], how="inner")
    if graded.is_empty():
        return graded

    actual = (
        pl.when(pl.col("market") == "player_receptions")
        .then(pl.col("actual_receptions"))
        .otherwise(pl.col("actual_carries"))
    )
    return (
        graded.with_columns(actual.alias("actual"))
        .filter(pl.col("actual").is_not_null() & (pl.col("actual") != pl.col("point")))
        .with_columns(
            pl.when(pl.col("side") == "Over")
            .then(pl.col("actual") > pl.col("point"))
            .otherwise(pl.col("actual") < pl.col("point"))
            .alias("won")
        )
    )


def grade(settled: pl.DataFrame) -> dict[str, float | int | str]:
    """Win rate against the break-even the offered price demands.

    Break-even is the vig-inclusive implied probability of the price actually
    taken, not 50%. Beating a coin flip on a -130 line is a losing strategy,
    and reporting a bare win rate hides that.
    """
    if settled.is_empty():
        return {"status": "nothing settled yet", "n": 0}

    win_rate = float(settled["won"].mean())
    break_even = float(
        sum(american_to_probability(p) for p in settled["offered_price"]) / settled.height
    )
    return {
        "status": "ok",
        "n": settled.height,
        "win_rate": win_rate,
        "break_even": break_even,
        "edge_realised": win_rate - break_even,
    }


def reliability(settled: pl.DataFrame, buckets: int = 5) -> pl.DataFrame:
    """Model probability against realised rate, bucketed.

    The diagnostic that matters most. A model can post a respectable overall
    win rate while being badly wrong about *which* bets it is confident in --
    and confidence is what decides where money goes. If the realised rate does
    not rise with the predicted one, the ordering is noise.
    """
    if settled.is_empty():
        return pl.DataFrame()
    return (
        settled.with_columns((pl.col("model_probability") * buckets).floor().alias("_bucket"))
        .group_by("_bucket")
        .agg(
            pl.col("model_probability").mean().alias("predicted"),
            pl.col("won").mean().alias("realised"),
            pl.len().alias("n"),
        )
        .sort("_bucket")
        .drop("_bucket")
    )
