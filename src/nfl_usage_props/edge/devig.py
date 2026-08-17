"""Removing the vig from a two-way price.

A book's two prices imply probabilities summing to more than one. The excess is
the hold, and stripping it out is the difference between "the book is offering
-130" and "the book thinks this happens 55% of the time". Every edge in this
project is a comparison against that second number, so how the hold is removed
is not a detail -- it changes which bets get flagged.

Three methods, because they disagree and the disagreement is informative
-----------------------------------------------------------------------
* **Multiplicative** divides both sides by the overround. Simple, and wrong in
  a specific way: it assumes the book applies its margin proportionally, so it
  moves the favourite and the longshot by the same *factor*. Real books load
  more margin onto longshots.
* **Power** raises both to a common exponent. This shifts more of the
  correction onto the longer side, which matches how prices actually behave.
* **Shin** models the hold as protection against informed money. It is the most
  principled of the three and the most sensitive to a single stale price.

On these markets the three typically land within a point of each other. When
they do not, the price is unusual -- a stale line, a thin market, a mistake --
and the spread between methods is a better warning than any single number, so
all three are computed and logged even though one drives the decision.

Why the consensus is weighted rather than averaged
--------------------------------------------------
Books are not independent estimates. Several copy a feed, so a plain average
counts one opinion five times and lets a cluster of copycats outvote a sharper
book. Weights are configurable for that reason. What this project *cannot* do
is anchor on Pinnacle, which does not appear in the free-tier feed at all -- so
the weights are a prior to be corrected once logged closing lines say which
books actually move first. Until then they are close to flat, on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl
from scipy.optimize import brentq

# A price this short is either a near-certainty nobody will take action on or a
# mistake. Either way the devig is unstable there.
_MIN_PROBABILITY = 1e-6
_MAX_PROBABILITY = 1.0 - 1e-6


def american_to_probability(price: float) -> float:
    """Raw implied probability, vig included."""
    if price < 0:
        return -price / (-price + 100.0)
    return 100.0 / (price + 100.0)


def probability_to_american(probability: float) -> float:
    """Inverse, for reporting a fair line next to the offered one."""
    probability = min(max(probability, _MIN_PROBABILITY), _MAX_PROBABILITY)
    # Strictly greater: even money is quoted +100, not -100. Both denote the
    # same price, but only one of them round-trips, and a report showing -100
    # where every book shows +100 invites a double-take every single week.
    if probability > 0.5:
        return -100.0 * probability / (1.0 - probability)
    return 100.0 * (1.0 - probability) / probability


def multiplicative_devig(over: float, under: float) -> tuple[float, float]:
    """Scale both sides down by the overround."""
    total = over + under
    return over / total, under / total


def power_devig(over: float, under: float) -> tuple[float, float]:
    """Find k with over^k + under^k = 1.

    k < 1 always (the raw probabilities sum above one), and raising to a power
    below one moves the smaller number proportionally more -- which is why this
    takes more margin off the longshot than the multiplicative method does.
    """
    if over <= 0 or under <= 0:
        raise ValueError("both implied probabilities must be positive")
    if abs(over + under - 1.0) < 1e-12:
        return over, under

    def overround(k: float) -> float:
        return over**k + under**k - 1.0

    # The function is monotone in k, so a bracket is all brentq needs. The
    # bounds are wide enough for any two-way price a book will post.
    low, high = 0.05, 20.0
    if overround(low) * overround(high) > 0:
        # No sign change: the prices are degenerate (one side ~certain). Fall
        # back rather than raise -- a single odd price should not take down a
        # whole slate's pricing.
        return multiplicative_devig(over, under)
    k = brentq(overround, low, high, xtol=1e-12)
    return over**k, under**k


def shin_devig(over: float, under: float) -> tuple[float, float]:
    """Shin's model: the hold is compensation for informed traders.

    Solves for z, the notional share of informed money, then backs out the
    probabilities a book would need to break even against it.
    """
    total = over + under
    if abs(total - 1.0) < 1e-12:
        return over, under

    def implied(z: float, q: float) -> float:
        root = math.sqrt(z * z + 4.0 * (1.0 - z) * q * q / total)
        return (root - z) / (2.0 * (1.0 - z))

    def residual(z: float) -> float:
        return implied(z, over) + implied(z, under) - 1.0

    low, high = 1e-9, 0.5
    if residual(low) * residual(high) > 0:
        return multiplicative_devig(over, under)
    z = brentq(residual, low, high, xtol=1e-12)
    return implied(z, over), implied(z, under)


DEVIG_METHODS = {
    "multiplicative": multiplicative_devig,
    "power": power_devig,
    "shin": shin_devig,
}


@dataclass(frozen=True)
class DevigResult:
    """Fair probabilities for both sides, under every method."""

    over: dict[str, float]
    under: dict[str, float]
    hold: float

    def spread(self, side: str = "over") -> float:
        """How far apart the methods are. Wide means the price is odd."""
        values = (self.over if side == "over" else self.under).values()
        return max(values) - min(values)


def devig_two_way(over_price: float, under_price: float) -> DevigResult:
    """Strip the hold from a paired price under all three methods."""
    over_raw = american_to_probability(over_price)
    under_raw = american_to_probability(under_price)
    if over_raw <= 0 or under_raw <= 0:
        raise ValueError(f"implausible prices: {over_price}, {under_price}")

    over, under = {}, {}
    for name, method in DEVIG_METHODS.items():
        over[name], under[name] = method(over_raw, under_raw)
    return DevigResult(over=over, under=under, hold=over_raw + under_raw - 1.0)


# ------------------------------------------------------------------ consensus


def consensus_probability(
    frame: pl.DataFrame,
    *,
    method: str = "power",
    weights: dict[str, float] | None = None,
    default_weight: float = 1.0,
    min_books: int = 3,
) -> pl.DataFrame:
    """Weighted consensus fair probability per (event, market, player, line).

    `frame` is a snapshot with one row per side per book, as
    `odds.parse` produces. Books that posted only one side are dropped: a
    one-sided price cannot be devigged, and guessing the other half would
    invent the very number being estimated.

    Markets with fewer than `min_books` books are kept but flagged rather than
    dropped, because "only two books price this" is itself worth seeing in the
    report -- a thin market is where both the largest edges and the largest
    mistakes live.
    """
    if method not in DEVIG_METHODS:
        raise ValueError(
            f"unknown devig method {method!r}; expected one of {sorted(DEVIG_METHODS)}"
        )
    weights = weights or {}

    # Key on gsis_id once the snapshot has been resolved, because two books can
    # spell the same player differently and grouping on the raw string would
    # split one market into two thin ones. The raw name rides along either way
    # so the report can show what the book actually posted.
    identifier = (
        "gsis_id"
        if "gsis_id" in frame.columns and frame["gsis_id"].null_count() < frame.height
        else "player_name_raw"
    )
    key = ["event_id", "market", identifier, "point"]
    # An empty snapshot is an ordinary outcome -- a pull that landed outside
    # the window, or a slate with no props posted yet. It has no columns to
    # filter on, so say "nothing priced" rather than raising about a schema.
    if frame.is_empty() or not {"price", "point", "side"} <= set(frame.columns):
        return frame.clear()

    usable = frame.filter(pl.col("price").is_not_null() & pl.col("point").is_not_null())
    if usable.is_empty():
        return usable

    paired = usable.pivot(
        on="side", index=[*key, "bookmaker"], values="price", aggregate_function="first"
    )
    # A snapshot can legitimately contain only one side -- a book pulled the
    # Under, or the pull caught the market mid-update. The pivot then has no
    # column to drop nulls on, so check before assuming both are there.
    if not {"Over", "Under"} <= set(paired.columns):
        return paired.clear()
    paired = paired.drop_nulls(["Over", "Under"])
    if paired.is_empty():
        return paired

    fair, holds, spreads = [], [], []
    for over_price, under_price in zip(paired["Over"], paired["Under"], strict=True):
        result = devig_two_way(over_price, under_price)
        fair.append(result.over[method])
        holds.append(result.hold)
        spreads.append(result.spread("over"))

    priced = paired.with_columns(
        pl.Series("fair_over", fair),
        pl.Series("hold", holds),
        pl.Series("method_spread", spreads),
        pl.col("bookmaker")
        .replace_strict(weights, default=default_weight, return_dtype=pl.Float64)
        .alias("weight"),
    )

    return (
        priced.group_by(key)
        .agg(
            ((pl.col("fair_over") * pl.col("weight")).sum() / pl.col("weight").sum()).alias(
                "consensus_over"
            ),
            pl.col("hold").mean().alias("mean_hold"),
            pl.col("method_spread").max().alias("max_method_spread"),
            pl.len().alias("n_books"),
            pl.col("Over").mean().alias("mean_over_price"),
            pl.col("Under").mean().alias("mean_under_price"),
        )
        .with_columns(
            (pl.col("n_books") < min_books).alias("thin_market"),
            (1.0 - pl.col("consensus_over")).alias("consensus_under"),
        )
        .sort(key)
    )
