"""Comparing the model to the market, and deciding what is worth a look.

An edge is `model probability - consensus fair probability` at the same line.
Everything hard about this module is in what it *refuses* to flag.

The filters exist because divergence is not edge
------------------------------------------------
The model disagreeing with the market is the normal state of affairs, and most
of that disagreement is the model being wrong. Each filter below removes a
class of disagreement where "the model is wrong" is more likely than "the book
is wrong":

* **Below the minimum edge.** Small disagreements are indistinguishable from
  calibration error. The threshold is set against typical prop hold, not fitted
  -- and it is an open question, to be re-derived from logged closing lines
  rather than guessed more precisely.
* **Extreme prices.** Devigging a -400 is unstable, the limits are tiny, and
  the true probability is in a region where the model was fitted on very few
  comparable outcomes.
* **Wide method spread.** When power, multiplicative and Shin disagree by more
  than a point, the price itself is strange. That is a better warning than any
  single fair value, and it costs nothing to compute.
* **Thin markets.** Two books pricing something is not a consensus. Flagged
  rather than dropped, because thin markets are where both the real edges and
  the real mistakes live and the report should say which is which.
* **Suppressed early weeks.** Role estimates before week 4 are mostly prior.
  The model will happily produce confident numbers there; they should not be
  bet.

No sizing. Deliberately, at every stage
---------------------------------------
This module reports probability, price, and edge. It does not recommend a
stake, and nothing downstream does either. Sizing depends on bankroll and risk
tolerance, which are not the model's business, and a Kelly fraction computed
from an unvalidated edge is a precise-looking number built on an unverified
one.
"""

from __future__ import annotations

import polars as pl

from nfl_usage_props.config import Config
from nfl_usage_props.edge.devig import consensus_probability, probability_to_american

# Beyond this the three devig methods are telling different stories about the
# same price, which usually means the price is stale or the market is thin.
MAX_METHOD_SPREAD = 0.01

# The market keys as The Odds API names them, mapped to what the model projects.
MARKET_TO_QUANTITY = {
    "player_receptions": "receptions",
    "player_rush_attempts": "carries",
}


def model_probabilities(projection, market: str, lines: pl.DataFrame) -> pl.DataFrame:
    """`P(over)` from the projection for each posted (player, line).

    Only lines a book actually posted are evaluated -- `output.scope` is
    `posted_line_only`, and projecting a player nobody prices produces a number
    with nothing to compare it to.
    """
    quantity = MARKET_TO_QUANTITY[market]
    draws = {
        "receptions": projection.receptions,
        "carries": projection.carries,
    }[quantity]

    index = projection.keys.with_row_index("_row").select("_row", "gsis_id", "game_id")
    joined = lines.join(index, on="gsis_id", how="inner")
    if joined.is_empty():
        return joined

    probabilities = [
        float((draws[row] > point).mean())
        for row, point in zip(joined["_row"], joined["point"], strict=True)
    ]
    return joined.with_columns(pl.Series("model_over", probabilities)).drop("_row")


def compute_edges(
    projection,
    snapshot: pl.DataFrame,
    config: Config,
    *,
    week: int | None = None,
) -> pl.DataFrame:
    """Full edge table for one snapshot: consensus, model, difference, flags.

    Every row a book posted comes back, flagged rather than filtered. The
    caller decides what to show; suppressing rows here would make a thin
    market and a missing player look identical in the report.
    """
    edge_config = config.edge
    consensus = consensus_probability(
        snapshot,
        method=edge_config.devig_method,
        weights=edge_config.consensus_book_weights,
        default_weight=edge_config.consensus_default_weight,
        min_books=edge_config.consensus_min_books,
    )
    if consensus.is_empty():
        return consensus

    frames = []
    for market in consensus["market"].unique().to_list():
        if market not in MARKET_TO_QUANTITY:
            continue
        lines = consensus.filter(pl.col("market") == market)
        priced = model_probabilities(projection, market, lines)
        if not priced.is_empty():
            frames.append(priced)
    if not frames:
        return pl.DataFrame()

    # On a two-way market the two edges are mirror images: whatever the model
    # gives the Over above consensus, it takes from the Under. So there is one
    # number, and its sign picks the side.
    edges = pl.concat(frames, how="diagonal_relaxed").with_columns(
        (pl.col("model_over") - pl.col("consensus_over")).alias("edge_over"),
    )
    edges = edges.with_columns(
        pl.when(pl.col("edge_over") >= 0)
        .then(pl.lit("Over"))
        .otherwise(pl.lit("Under"))
        .alias("side"),
        pl.col("edge_over").abs().alias("edge"),
    ).with_columns(
        pl.when(pl.col("side") == "Over")
        .then(pl.col("model_over"))
        .otherwise(1.0 - pl.col("model_over"))
        .alias("model_probability"),
        pl.when(pl.col("side") == "Over")
        .then(pl.col("mean_over_price"))
        .otherwise(pl.col("mean_under_price"))
        .alias("offered_price"),
    )

    fair = [probability_to_american(p) for p in edges["model_probability"]]
    return _flag(edges.with_columns(pl.Series("model_fair_price", fair)), config, week).sort(
        "edge", descending=True
    )


def _flag(edges: pl.DataFrame, config: Config, week: int | None) -> pl.DataFrame:
    """Attach the reasons a row should not be acted on."""
    edge_config = config.edge
    suppress_before = config.model.early_season.suppress_output_before_week

    early = week is not None and week < suppress_before
    return edges.with_columns(
        (pl.col("edge") < edge_config.min_edge).alias("below_threshold"),
        (pl.col("offered_price") <= edge_config.extreme_price_threshold).alias("extreme_price"),
        (pl.col("max_method_spread") > MAX_METHOD_SPREAD).alias("devig_disagreement"),
        pl.lit(early).alias("early_season"),
    ).with_columns(
        (
            ~pl.col("below_threshold")
            & ~pl.col("extreme_price")
            & ~pl.col("devig_disagreement")
            & ~pl.col("thin_market")
            & ~pl.col("early_season")
        ).alias("actionable")
    )


def suppression_reasons(edges: pl.DataFrame) -> pl.DataFrame:
    """Why rows were held back, counted. Read this before the edges themselves.

    A week where everything is suppressed for one reason is a signal about the
    pipeline -- a devig disagreeing everywhere means a bad snapshot, not a
    quiet slate.
    """
    flags = [
        "below_threshold",
        "extreme_price",
        "devig_disagreement",
        "thin_market",
        "early_season",
    ]
    return pl.DataFrame(
        {
            "reason": flags,
            "rows": [int(edges[f].sum()) for f in flags],
        }
    ).sort("rows", descending=True)
