"""Mechanical leakage detection.

A leak does not look like a bug. It looks like a good model. The backtest
improves, every diagnostic gets better, and the failure only shows up in live
money -- by which point it is expensive and hard to attribute. So this is not
a code review checklist; it is an executable test that fails the build.

Two independent failures, caught two different ways:

**Same-game leakage** -- a feature for game g reads game g's own outcome. This
is the one that inflates backtests most and hides best, because the offending
column looks like ordinary history. Caught by corrupting the outcomes of the
games under test and rebuilding: any feature whose value moves was reading the
result it is supposed to be predicting.

**Future leakage** -- a feature for game g reads games played after it. Caught
by truncating the corpus to end at g and rebuilding: any feature whose value
moves was reading forward.

The two are genuinely different. A trailing mean computed with `rolling_mean`
and no shift leaks same-game but not future. A league average computed over
the whole season leaks future but not same-game. Only running both catches both.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_usage_props.config import Config
from nfl_usage_props.features.build import (
    OUTCOME_COLUMNS,
    build_features,
    feature_columns,
)

# Comparisons are on floats, so an exact-equality test would fail on ordinary
# reassociation from a different row order. This is loose enough to ignore that
# and far tighter than any real leak.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class LeakageFinding:
    column: str
    kind: str
    rows_changed: int
    total_rows: int
    example: str = ""

    def __str__(self) -> str:
        return (
            f"{self.kind} leak in {self.column!r}: "
            f"{self.rows_changed}/{self.total_rows} rows changed. {self.example}".strip()
        )


def check_same_game_leakage(
    panel: pl.DataFrame,
    schedules: pl.DataFrame,
    config: Config,
    *,
    season: int,
    week: int,
) -> list[LeakageFinding]:
    """Corrupt the target week's outcomes; any feature that moves is reading them.

    The corruption reverses each outcome column within the target week, which
    keeps dtypes, ranges and null patterns intact -- so a feature that changes
    changed because of *which player* got the production, not because the data
    suddenly looked strange.
    """
    target = (pl.col("season") == season) & (pl.col("week") == week)
    if panel.filter(target).is_empty():
        raise ValueError(f"no panel rows for season {season} week {week}")

    corrupted = panel.with_columns(
        [
            pl.when(target)
            .then(pl.col(column).reverse().over(pl.when(target).then(1).otherwise(0)))
            .otherwise(pl.col(column))
            .alias(column)
            for column in OUTCOME_COLUMNS
            if column in panel.columns
        ]
    )

    baseline = build_features(panel, schedules, config).filter(target)
    perturbed = build_features(corrupted, schedules, config).filter(target)
    return _compare(baseline, perturbed, kind="same-game")


def check_future_leakage(
    panel: pl.DataFrame,
    schedules: pl.DataFrame,
    config: Config,
    *,
    season: int,
    week: int,
) -> list[LeakageFinding]:
    """Delete everything after the target week; any feature that moves read forward."""
    target = (pl.col("season") == season) & (pl.col("week") == week)
    if panel.filter(target).is_empty():
        raise ValueError(f"no panel rows for season {season} week {week}")

    truncated = panel.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= week))
    )
    # Schedules must be truncated too. Leaving the full slate in place would
    # let a future-reading context feature pass by drawing on rows the panel no
    # longer has.
    truncated_schedules = schedules.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= week))
    )

    baseline = build_features(panel, schedules, config).filter(target)
    restricted = build_features(truncated, truncated_schedules, config).filter(target)
    return _compare(baseline, restricted, kind="future")


def run_leakage_checks(
    panel: pl.DataFrame,
    schedules: pl.DataFrame,
    config: Config,
    *,
    season: int,
    week: int,
) -> list[LeakageFinding]:
    """Both checks. Empty list means the feature matrix is clean for that week."""
    return [
        *check_same_game_leakage(panel, schedules, config, season=season, week=week),
        *check_future_leakage(panel, schedules, config, season=season, week=week),
    ]


def _compare(baseline: pl.DataFrame, other: pl.DataFrame, *, kind: str) -> list[LeakageFinding]:
    """Row-aligned comparison of every feature column."""
    keys = ["game_id", "gsis_id"]
    columns = feature_columns(baseline)

    left = baseline.select(*keys, *columns).sort(keys)
    right = other.select(*keys, *columns).sort(keys)
    if left.height != right.height:
        raise AssertionError(
            f"{kind} check changed the row count ({left.height} -> {right.height}); "
            "the comparison is meaningless, fix the harness before reading the result"
        )

    findings: list[LeakageFinding] = []
    for column in columns:
        a, b = left[column], right[column]
        if a.dtype.is_numeric() and b.dtype.is_numeric():
            differs = ((a.cast(pl.Float64) - b.cast(pl.Float64)).abs() > TOLERANCE).fill_null(False)
            # A null on exactly one side is a difference the subtraction hides.
            differs = differs | (a.is_null() != b.is_null())
        else:
            differs = (a != b).fill_null(a.is_null() != b.is_null())

        changed = int(differs.sum())
        if changed:
            row = left.filter(differs).head(1)
            example = (
                f"e.g. {row['game_id'][0]}/{row['gsis_id'][0]}: "
                f"{a.filter(differs)[0]} -> {b.filter(differs)[0]}"
            )
            findings.append(
                LeakageFinding(
                    column=column,
                    kind=kind,
                    rows_changed=changed,
                    total_rows=left.height,
                    example=example,
                )
            )
    return findings
