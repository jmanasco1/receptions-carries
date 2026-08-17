from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nfl_usage_props.config import (
    Config,
    CreditConfig,
    EdgeConfig,
    IngestConfig,
    ModelConfig,
    OddsConfig,
    OutputConfig,
    RetryConfig,
    StorageConfig,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A Config pointed at a temp dir, with retries fast enough for tests."""
    return Config(
        data_dir=tmp_path / "data",
        ingest=IngestConfig(
            first_season=2022,
            last_season=2024,
            tables=("schedules", "pbp"),
        ),
        storage=StorageConfig(compression="zstd", completion_lag_days=30),
        odds=OddsConfig(
            base_url="https://api.the-odds-api.com/v4",
            sport="americanfootball_nfl",
            regions="us",
            game_markets=("spreads", "totals"),
            prop_markets=("player_receptions", "player_rush_attempts"),
            credits=CreditConfig(floor=50, max_spend_per_run=120),
            retry=RetryConfig(
                max_attempts=3,
                backoff_initial_seconds=0.0,
                backoff_multiplier=1.0,
                backoff_max_seconds=0.0,
                timeout_seconds=5.0,
            ),
        ),
        edge=EdgeConfig(),
        model=ModelConfig(),
        output=OutputConfig(),
        source_path=tmp_path / "config.toml",
    )


# --------------------------------------------------------------------------
# A synthetic league, so the Stage 2 feature tests run offline and in
# milliseconds. Small enough to reason about by hand: four teams, one game each
# per week, a fixed roster with deliberately different usage profiles.
# --------------------------------------------------------------------------

SYNTHETIC_TEAMS = ("AAA", "BBB", "CCC", "DDD")
SYNTHETIC_WEEKS = 6

# (player, team, target share, carry share, snap share) -- the truth the role
# tracker is supposed to recover.
SYNTHETIC_ROSTER = [
    ("00-0000001", "AAA", 0.28, 0.00, 0.90),  # WR1
    ("00-0000002", "AAA", 0.14, 0.02, 0.55),  # WR2
    ("00-0000003", "AAA", 0.10, 0.65, 0.60),  # RB1
    ("00-0000004", "BBB", 0.24, 0.00, 0.85),
    ("00-0000005", "BBB", 0.08, 0.70, 0.62),
    ("00-0000006", "CCC", 0.30, 0.00, 0.92),
    ("00-0000007", "CCC", 0.12, 0.60, 0.58),
    ("00-0000008", "DDD", 0.22, 0.00, 0.80),
    ("00-0000009", "DDD", 0.11, 0.68, 0.64),
]

_FIRST_KICKOFF = datetime(2024, 9, 5, 20, 20, tzinfo=UTC)


def _synthetic_matchups(week: int) -> list[tuple[str, str]]:
    """Rotate opponents so every team faces a different defence each week."""
    home, away = SYNTHETIC_TEAMS[0::2], SYNTHETIC_TEAMS[1::2]
    rotated = away[week % len(away) :] + away[: week % len(away)]
    return list(zip(home, rotated, strict=True))


def make_synthetic_panel(season: int = 2024, weeks: int = SYNTHETIC_WEEKS) -> pl.DataFrame:
    """A panel whose player shares are exactly those in SYNTHETIC_ROSTER.

    Deterministic on purpose: a test that asserts the tracker converges toward
    the truth should fail because the tracker is wrong, not because a random
    draw went badly on a seed nobody looks at. Team volume still varies week to
    week -- see below for why a perfectly flat panel would be worse than useless.
    """
    rows = []
    for week in range(1, weeks + 1):
        kickoff = _FIRST_KICKOFF + timedelta(days=7 * (week - 1))
        for home, away in _synthetic_matchups(week):
            game_id = f"{season}_{week:02d}_{away}_{home}"
            for team, opponent in ((home, away), (away, home)):
                # Team volume varies week to week while every player's SHARE
                # stays fixed. Without this the panel is perfectly constant, and
                # a constant panel cannot distinguish a season-wide aggregate
                # from an as-of one -- it would let a future-reading feature
                # pass the leakage test by coincidence.
                team_plays = 58 + 2 * (week % 4)
                team_targets = 30 + (week % 5)
                team_carries = 22 + (week % 3)
                team_dropbacks = 34 + (week % 5)
                for gsis_id, roster_team, ts, cs, ss in SYNTHETIC_ROSTER:
                    if roster_team != team:
                        continue
                    targets = round(ts * team_targets)
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "season_type": "REG",
                            "game_id": game_id,
                            "gsis_id": gsis_id,
                            "team": team,
                            "opponent": opponent,
                            "kickoff_utc": kickoff,
                            "targets": targets,
                            "receptions": round(targets * 0.65),
                            "carries": round(cs * team_carries),
                            "snaps": round(ss * team_plays),
                            "pass_snaps": round(ss * team_dropbacks),
                            "team_plays": team_plays,
                            "team_dropbacks": team_dropbacks,
                            "team_rush_attempts": team_carries,
                            "team_targets": team_targets,
                            "team_carries": team_carries,
                            "team_completions": 22,
                            "team_proe": 0.01,
                            "team_xpass": 0.58,
                            "team_pass_rate": team_dropbacks / team_plays,
                        }
                    )
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        (pl.col("targets") / pl.col("team_targets")).alias("target_share"),
        (pl.col("carries") / pl.col("team_carries")).alias("carry_share"),
        (pl.col("snaps") / pl.col("team_plays")).alias("snap_share"),
        (pl.col("pass_snaps") / pl.col("team_dropbacks")).alias("route_share"),
        (pl.col("receptions") / pl.col("targets")).alias("catch_rate"),
    )


def make_synthetic_schedules(season: int = 2024, weeks: int = SYNTHETIC_WEEKS) -> pl.DataFrame:
    rows = []
    for week in range(1, weeks + 1):
        kickoff = _FIRST_KICKOFF + timedelta(days=7 * (week - 1))
        for home, away in _synthetic_matchups(week):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "home_team": home,
                    "away_team": away,
                    "kickoff_utc": kickoff,
                    # Real ids from reference/stadiums.csv -- DET00 is a fixed
                    # dome, BUF00 is open air -- so the roof join is exercised
                    # against the real table rather than against a null.
                    "stadium_id": "DET00" if home == "AAA" else "BUF00",
                    "roof": "dome" if home == "AAA" else "outdoors",
                    "wind": 0 if home == "AAA" else 9,
                    "div_game": 0,
                    "home_rest": 7,
                    "away_rest": 7,
                    "spread_line": -3.0,
                    "total_line": 44.5,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def synthetic_panel() -> pl.DataFrame:
    return make_synthetic_panel()


@pytest.fixture
def synthetic_schedules() -> pl.DataFrame:
    return make_synthetic_schedules()
