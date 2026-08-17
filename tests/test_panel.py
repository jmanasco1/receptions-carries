"""Tests for the player-game usage panel.

The panel is the foundation: a quiet error here (a share that does not sum to
one, a play universe that drifts between numerator and denominator) propagates
into every layer above it and shows up as a model that is subtly miscalibrated
for reasons nobody can find.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.features.panel import (
    build_panel,
    player_game_snaps,
    player_game_usage,
    team_game_totals,
)

# One team, one game, hand-built so every expected number can be counted by eye.
#   plays 1-3 passes (two completed), play 4 a run, play 5 a kneel,
#   play 6 a penalty-wiped pass, play 7 a two-point try.
PLAYS = [
    # play_id, type,   two_pt, dropback, rush_att, complete, receiver, rusher
    (1, "pass", 0, 1, 0, 1, "WR1", None),
    (2, "pass", 0, 1, 0, 1, "WR2", None),
    (3, "pass", 0, 1, 0, 0, "WR1", None),
    (4, "run", 0, 0, 1, 0, None, "RB1"),
    (5, "qb_kneel", 0, 0, 1, 0, None, "QB1"),
    (6, "no_play", 0, 1, 0, 0, "WR1", None),
    (7, "pass", 1, 1, 0, 1, "WR2", None),
]


@pytest.fixture
def pbp() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "season_type": "REG",
                "game_id": "G1",
                "play_id": play_id,
                "posteam": "AAA",
                "defteam": "BBB",
                "play_type": play_type,
                "two_point_attempt": two_pt,
                "qb_dropback": dropback,
                "pass_attempt": dropback,
                "rush_attempt": rush,
                "complete_pass": complete,
                "receiver_player_id": receiver,
                "rusher_player_id": rusher,
                "pass_oe": 2.0,
                "xpass": 0.6,
            }
            for play_id, play_type, two_pt, dropback, rush, complete, receiver, rusher in PLAYS
        ]
    )


@pytest.fixture
def participation() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "nflverse_game_id": "G1",
                "play_id": play_id,
                "offense_players": "WR1;WR2;RB1;QB1",
            }
            for play_id, *_ in PLAYS
        ]
    )


# -------------------------------------------------------------- play universe


def test_only_real_scrimmage_snaps_count(pbp):
    """Four plays survive: three passes and a run. The kneel, the wiped play
    and the two-point try are all excluded."""
    assert team_game_totals(pbp)["team_plays"][0] == 4


def test_kneels_are_excluded_from_team_carries(pbp):
    """A kneel is a clock play, not an opportunity anyone competes for.
    nflverse's own `player_stats.carries` counts them, which is why our QB
    carry totals differ from theirs by exactly the kneels."""
    assert team_game_totals(pbp)["team_carries"][0] == 1


def test_two_point_attempts_do_not_count(pbp):
    totals = team_game_totals(pbp)
    assert totals["team_targets"][0] == 3
    assert totals["team_dropbacks"][0] == 3


def test_penalty_wiped_plays_do_not_count(pbp):
    """Play 6 has a receiver on it, but the snap was wiped -- counting it would
    inflate the denominator and deflate every share on the team."""
    usage = player_game_usage(pbp)
    assert usage.filter(pl.col("gsis_id") == "WR1")["targets"][0] == 2


# --------------------------------------------------------------- player usage


def test_targets_receptions_and_carries(pbp):
    usage = player_game_usage(pbp).sort("gsis_id")
    by_player = {row["gsis_id"]: row for row in usage.to_dicts()}
    assert by_player["WR1"]["targets"] == 2
    assert by_player["WR1"]["receptions"] == 1
    assert by_player["WR2"]["targets"] == 1
    assert by_player["RB1"]["carries"] == 1


def test_a_player_who_caught_and_carried_lands_on_one_row():
    """Joining two aggregations naively either duplicates this player or drops
    one of his contributions."""
    frame = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "season_type": "REG",
                "game_id": "G1",
                "play_id": 1,
                "posteam": "AAA",
                "defteam": "BBB",
                "play_type": "pass",
                "two_point_attempt": 0,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "rush_attempt": 0,
                "complete_pass": 1,
                "receiver_player_id": "RB1",
                "rusher_player_id": None,
                "pass_oe": 0.0,
                "xpass": 0.5,
            },
            {
                "season": 2024,
                "week": 1,
                "season_type": "REG",
                "game_id": "G1",
                "play_id": 2,
                "posteam": "AAA",
                "defteam": "BBB",
                "play_type": "run",
                "two_point_attempt": 0,
                "qb_dropback": 0,
                "pass_attempt": 0,
                "rush_attempt": 1,
                "complete_pass": 0,
                "receiver_player_id": None,
                "rusher_player_id": "RB1",
                "pass_oe": 0.0,
                "xpass": 0.5,
            },
        ]
    )
    usage = player_game_usage(frame)
    assert usage.height == 1
    assert usage["targets"][0] == 1
    assert usage["carries"][0] == 1


def test_players_with_no_touches_still_appear_via_snaps(pbp, participation):
    """QB1 has no targets and no non-kneel carries but was on the field. He
    belongs in the panel with a zero, not missing from it."""
    panel = build_panel(pbp, participation)
    assert "QB1" in panel["gsis_id"].to_list()
    row = panel.filter(pl.col("gsis_id") == "QB1").to_dicts()[0]
    assert row["targets"] == 0
    assert row["carries"] == 0


# --------------------------------------------------------------------- snaps


def test_snaps_count_only_the_filtered_play_universe(pbp, participation):
    """Participation lists all seven plays; only the four real snaps count."""
    snaps = player_game_snaps(pbp, participation)
    assert snaps["snaps"].unique().to_list() == [4]


def test_pass_snaps_count_dropbacks_only(pbp, participation):
    snaps = player_game_snaps(pbp, participation)
    assert snaps["pass_snaps"].unique().to_list() == [3]


def test_missing_participation_yields_an_empty_frame_not_a_crash(pbp):
    """Participation is a separate feed and can be absent for a season. The
    panel should lose its snap columns, not fail to build."""
    snaps = player_game_snaps(pbp, pl.DataFrame())
    assert snaps.is_empty()


# -------------------------------------------------------------------- shares


def test_shares_sum_to_one_across_the_team(pbp, participation):
    """Layer 3 is a Dirichlet-Multinomial over the roster. If these do not sum
    to one, its shares are not a distribution and nothing above it is valid."""
    panel = build_panel(pbp, participation)
    assert panel["target_share"].sum() == pytest.approx(1.0)
    assert panel["carry_share"].sum() == pytest.approx(1.0)


def test_a_zero_denominator_gives_null_not_infinity():
    """A team with no carries in a game is rare and real. 0/0 as a float is
    silently poisonous; null propagates as 'no observation', which is correct."""
    frame = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "season_type": "REG",
                "game_id": "G1",
                "play_id": 1,
                "posteam": "AAA",
                "defteam": "BBB",
                "play_type": "pass",
                "two_point_attempt": 0,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "rush_attempt": 0,
                "complete_pass": 1,
                "receiver_player_id": "WR1",
                "rusher_player_id": None,
                "pass_oe": 0.0,
                "xpass": 0.5,
            }
        ]
    )
    panel = build_panel(frame, pl.DataFrame())
    assert panel["carry_share"][0] is None
    assert panel["target_share"][0] == pytest.approx(1.0)


def test_catch_rate_is_null_without_targets(pbp, participation):
    panel = build_panel(pbp, participation)
    assert panel.filter(pl.col("gsis_id") == "RB1")["catch_rate"][0] is None


def test_positions_are_joined_when_supplied(pbp, participation):
    positions = pl.DataFrame({"gsis_id": ["WR1"], "position": ["WR"]})
    panel = build_panel(pbp, participation, positions=positions)
    assert panel.filter(pl.col("gsis_id") == "WR1")["position"][0] == "WR"
