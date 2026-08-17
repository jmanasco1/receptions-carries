"""Stage 8: the weekly report.

One table of what the model disagrees with the market about, and -- above it --
what it refused to say and why. The suppression counts come first on purpose:
a week where every row is held back for one reason is a broken pipeline, not a
quiet slate, and a report that only ever shows the survivors cannot tell you
which you are looking at.

The report shows probability, price and edge. It does not show a stake. Sizing
depends on bankroll and risk tolerance, which are not the model's business, and
a Kelly fraction computed from an edge nobody has validated yet is a
precise-looking number resting on an unverified one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl_usage_props.edge.devig import probability_to_american

REPORT_COLUMNS = (
    "player",
    "team",
    "market",
    "side",
    "point",
    "offered_price",
    "model_fair_price",
    "model_probability",
    "consensus_over",
    "edge",
    "n_books",
    "mean_hold",
)


def build_report(
    edges: pl.DataFrame,
    *,
    players: pl.DataFrame | None = None,
    actionable_only: bool = True,
) -> pl.DataFrame:
    """The table a human reads."""
    if edges.is_empty():
        return edges

    frame = edges.filter(pl.col("actionable")) if actionable_only else edges
    if players is not None and "gsis_id" in frame.columns:
        frame = frame.join(
            players.select("gsis_id", pl.col("display_name").alias("player")),
            on="gsis_id",
            how="left",
        )
    if "player" not in frame.columns:
        frame = frame.with_columns(
            pl.coalesce(
                [pl.col(c) for c in ("player_name_raw", "gsis_id") if c in frame.columns]
            ).alias("player")
        )
    if "team" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("team"))

    available = [c for c in REPORT_COLUMNS if c in frame.columns]
    return frame.select(available).sort("edge", descending=True)


def render_markdown(
    edges: pl.DataFrame,
    *,
    week: int | None = None,
    season: int | None = None,
    clv: dict | None = None,
    players: pl.DataFrame | None = None,
    actionable_only: bool = True,
) -> str:
    """Markdown, because it renders anywhere and diffs cleanly week to week."""
    from nfl_usage_props.edge.edges import suppression_reasons

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    label = f"{season} week {week}" if season and week else "unscheduled slate"
    lines = [f"# Usage props — {label}", "", f"_Generated {stamp}._", ""]

    if edges.is_empty():
        lines += ["No priced markets in this snapshot. Nothing to report.", ""]
        return "\n".join(lines)

    lines += [
        "## What was held back",
        "",
        "Read this first. If one reason accounts for nearly everything, the",
        "pipeline is more likely at fault than the slate.",
        "",
        "| reason | rows |",
        "| --- | ---: |",
    ]
    for row in suppression_reasons(edges).to_dicts():
        lines.append(f"| {row['reason'].replace('_', ' ')} | {row['rows']} |")
    actionable = int(edges["actionable"].sum())
    lines += ["", f"Priced markets: {edges.height}. Actionable: {actionable}.", ""]

    report = build_report(edges, players=players, actionable_only=actionable_only)
    lines += ["## Flagged", ""]
    if report.is_empty():
        lines += ["Nothing cleared the filters this week. That is a normal outcome.", ""]
    else:
        lines += [
            "| player | market | side | line | offered | fair | model | edge | books |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in report.to_dicts():
            market = str(row.get("market", "")).replace("player_", "")
            lines.append(
                f"| {row.get('player', '?')} | {market} | {row.get('side', '')} "
                f"| {row.get('point', '')} | {_price(row.get('offered_price'))} "
                f"| {_price(row.get('model_fair_price'))} "
                f"| {row.get('model_probability', 0):.1%} | {row.get('edge', 0):+.1%} "
                f"| {row.get('n_books', '')} |"
            )
        lines.append("")

    lines += ["## Closing line value", ""]
    if not clv or clv.get("status") != "ok":
        status = (clv or {}).get("status", "no scored bets yet")
        lines += [
            f"_{status}._",
            "",
            "Closing line value is the only validation available on the free tier,",
            "and it cannot be backfilled — historical props are paid-tier. It",
            "accumulates from the first logged snapshot forward.",
            "",
        ]
    else:
        lines += [
            f"- scored bets: **{clv['n']}**",
            f"- median CLV: **{clv['median_clv']:+.3f}** probability points",
            f"- share positive: **{clv['share_positive']:.1%}**",
            "",
        ]

    lines += [
        "---",
        "",
        "No stake sizes, deliberately. Sizing depends on bankroll and risk",
        "tolerance, and a Kelly fraction built on an unvalidated edge is a",
        "precise number resting on an unverified one.",
        "",
    ]
    return "\n".join(lines)


def _price(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.0f}"


def fair_line(probability: float) -> float:
    """Exposed for the report and for anyone checking the arithmetic by hand."""
    return probability_to_american(probability)
