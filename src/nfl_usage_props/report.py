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


# ---------------------------------------------------------------------- HTML


HTML_STYLE = """
:root{--bg:#fbfaf9;--fg:#1c1b19;--muted:#6b6862;--line:#e4e1dc;--card:#fff;
--good:#1a7f4b;--warn:#a8630a;--accent:#b4532a}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171614;
--fg:#eceae6;--muted:#9a958c;--line:#2e2c28;--card:#201e1b;--good:#54c98a;
--warn:#e0a355;--accent:#e08050}}
:root[data-theme=dark]{--bg:#171614;--fg:#eceae6;--muted:#9a958c;--line:#2e2c28;
--card:#201e1b;--good:#54c98a;--warn:#e0a355;--accent:#e08050}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:32px 16px 72px}
h1{font-size:1.65rem;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:1.05rem;margin:34px 0 10px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:.85rem;margin-bottom:26px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:11px 15px;min-width:118px}
.card .n{font-size:1.4rem;font-weight:650;font-variant-numeric:tabular-nums}
.card .l{color:var(--muted);font-size:.72rem;text-transform:uppercase;
letter-spacing:.05em;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:.88rem;
background:var(--card);border:1px solid var(--line);border-radius:10px;
overflow:hidden}
th{text-align:left;font-weight:600;font-size:.72rem;text-transform:uppercase;
letter-spacing:.05em;color:var(--muted);padding:9px 12px;
border-bottom:1px solid var(--line)}
td{padding:9px 12px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
th.r,td.r{text-align:right}
.edge{color:var(--good);font-weight:650}
.side{font-weight:600}
.note{color:var(--muted);font-size:.83rem;border-left:2px solid var(--line);
padding-left:13px;margin:14px 0}
.empty{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:22px;color:var(--muted)}
@media(max-width:560px){.wrap{padding:20px 16px 48px}table{font-size:.8rem}
td,th{padding:7px 8px}}
"""


def render_html(
    edges: pl.DataFrame,
    *,
    week: int | None = None,
    season: int | None = None,
    clv: dict | None = None,
    players: pl.DataFrame | None = None,
    actionable_only: bool = True,
) -> str:
    """The same report as a standalone page.

    Markdown renders fine on GitHub, but it is not what anyone wants to read on
    a Sunday morning. Everything is inline so the file can be opened straight
    from disk with no server and no assets.
    """
    from nfl_usage_props.edge.edges import suppression_reasons

    label = f"{season} · week {week}" if season and week else "unscheduled slate"
    stamp = datetime.now(UTC).strftime("%a %d %b %Y, %H:%M UTC")
    out = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width,initial-scale=1">',
        f"<title>Usage props — {label}</title><style>{HTML_STYLE}</style>",
        "</head><body><div class=wrap>",
        f"<h1>Usage props — {label}</h1>",
        f"<div class=sub>Generated {stamp}. Receptions and rush attempts.</div>",
    ]

    if edges.is_empty():
        out += [
            "<div class=empty>No priced markets in this snapshot — nothing to "
            "compare a projection against.</div></div></body></html>"
        ]
        return "".join(out)

    actionable = int(edges["actionable"].sum())
    out += [
        "<div class=cards>",
        f"<div class=card><div class=n>{edges.height}</div><div class=l>priced</div></div>",
        f"<div class=card><div class=n>{actionable}</div><div class=l>flagged</div></div>",
    ]
    if clv and clv.get("status") == "ok":
        out.append(
            f"<div class=card><div class=n>{clv['median_clv']:+.3f}</div>"
            "<div class=l>median CLV</div></div>"
        )
    out.append("</div>")

    report = build_report(edges, players=players, actionable_only=actionable_only)
    out.append("<h2>Flagged</h2>")
    if report.is_empty():
        out.append(
            "<div class=empty>Nothing cleared the filters. That is a normal "
            "outcome, not a failure — see what was held back below.</div>"
        )
    else:
        out.append(
            "<table><tr><th>player</th><th>market</th><th>side</th>"
            "<th class=r>line</th><th class=r>offered</th><th class=r>fair</th>"
            "<th class=r>model</th><th class=r>edge</th><th class=r>books</th></tr>"
        )
        for row in report.to_dicts():
            market = str(row.get("market", "")).replace("player_", "")
            out.append(
                f"<tr><td>{row.get('player', '?')}</td><td>{market}</td>"
                f"<td class=side>{row.get('side', '')}</td>"
                f"<td class=r>{row.get('point', '')}</td>"
                f"<td class=r>{_price(row.get('offered_price'))}</td>"
                f"<td class=r>{_price(row.get('model_fair_price'))}</td>"
                f"<td class=r>{row.get('model_probability', 0):.0%}</td>"
                f"<td class='r edge'>{row.get('edge', 0):+.1%}</td>"
                f"<td class=r>{row.get('n_books', '')}</td></tr>"
            )
        out.append("</table>")

    out.append("<h2>What was held back</h2>")
    out.append(
        "<div class=note>Read this before the table above. If one reason "
        "accounts for nearly everything, the pipeline is more likely at fault "
        "than the slate.</div>"
    )
    out.append("<table><tr><th>reason</th><th class=r>rows</th></tr>")
    for row in suppression_reasons(edges).to_dicts():
        out.append(
            f"<tr><td>{row['reason'].replace('_', ' ')}</td><td class=r>{row['rows']}</td></tr>"
        )
    out.append("</table>")

    out.append("<h2>Closing line value</h2>")
    if not clv or clv.get("status") != "ok":
        status = (clv or {}).get("status", "no scored bets yet")
        out.append(
            f"<div class=empty><b>{status}.</b><br><br>CLV is the only "
            "validation available on the free tier and cannot be backfilled — "
            "historical props are paid-tier. It accumulates from the first "
            "logged snapshot forward.</div>"
        )
    else:
        out.append(
            "<div class=cards>"
            f"<div class=card><div class=n>{clv['n']}</div><div class=l>scored</div></div>"
            f"<div class=card><div class=n>{clv['median_clv']:+.3f}</div>"
            "<div class=l>median CLV</div></div>"
            f"<div class=card><div class=n>{clv['share_positive']:.0%}</div>"
            "<div class=l>positive</div></div></div>"
        )

    out.append(
        "<div class=note>No stake sizes, deliberately. Sizing depends on "
        "bankroll and risk tolerance, and a Kelly fraction built on an "
        "unvalidated edge is a precise number resting on an unverified one."
        "</div>"
    )
    out.append("</div></body></html>")
    return "".join(out)
