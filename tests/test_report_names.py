"""The report has to print names, not gsis ids.

`build_report` has always accepted a `players` frame; the defect was that the
weekly report command never passed one. The consensus groups on `gsis_id` and
drops the book's raw spelling, so the fallback had nothing left to coalesce to
and six weeks of reports went out reading "00-0040667".
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from nfl_usage_props.report import build_report, render_markdown

GSIS = re.compile(r"\b\d{2}-\d{7}\b")


def edges() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["00-0040667", "00-0038797"],
            "market": ["player_receptions", "player_rush_attempts"],
            "side": ["Under", "Over"],
            "point": [4.5, 18.5],
            "offered_price": [116.0, -114.0],
            "model_fair_price": [-403.0, -686.0],
            "model_probability": [0.801, 0.873],
            "edge": [0.373, 0.372],
            "n_books": [4, 3],
            "actionable": [True, True],
            "below_threshold": [False, False],
            "extreme_price": [False, False],
            "devig_disagreement": [False, False],
            "thin_market": [False, False],
            "early_season": [False, False],
        }
    )


def players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["00-0040667", "00-0038797"],
            "display_name": ["Ricky Pearsall", "Bhayshul Tuten"],
        }
    )


def test_names_replace_ids_when_the_players_table_is_supplied():
    report = build_report(edges(), players=players())
    assert report["player"].to_list() == ["Ricky Pearsall", "Bhayshul Tuten"]


def test_rendered_markdown_carries_no_bare_gsis_id():
    text = render_markdown(edges(), season=2026, week=6, clv={}, players=players())
    assert not GSIS.search(text), "a gsis id reached the rendered report"


def test_the_weekly_report_command_passes_players():
    """The unit above passed throughout the outage -- the call site was wrong.

    Guarding the wiring rather than the function, because that is what broke.
    """
    source = Path("src/nfl_usage_props/cli.py").read_text()
    shared = source.split("shared = {", 1)[1].split("}", 1)[0]
    assert '"players"' in shared, "the weekly report no longer passes a players table"
