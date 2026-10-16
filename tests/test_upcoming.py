"""Tests for forward projection — slates that have not been played.

The panel is built from play-by-play, so before this existed the pipeline could
only ever project the past. These tests cover the seam, and one of them exists
because the seam broke in a way that produced no error at all.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from nfl_usage_props.features.upcoming import (
    DEPTH_LIMITS,
    EXPECTED_ROSTER_SIZE,
    OUTCOME_COLUMNS,
    expected_rosters,
    panel_with_upcoming,
    upcoming_panel_rows,
)

KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)


def depth_chart(teams=("AAA", "BBB"), published="2026-09-10T12:00:00Z") -> pl.DataFrame:
    """A chart deeper than any real rotation, so the caps have work to do."""
    rows = []
    for team in teams:
        for position, count in (("QB", 3), ("RB", 6), ("WR", 9), ("TE", 5), ("FB", 2)):
            for rank in range(1, count + 1):
                rows.append(
                    {
                        "dt": published,
                        "team": team,
                        "gsis_id": f"{team}-{position}{rank}",
                        "pos_abb": position,
                        "pos_rank": rank,
                    }
                )
    return pl.DataFrame(rows)


def schedule() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_BBB_AAA",
                "home_team": "AAA",
                "away_team": "BBB",
                "kickoff_utc": KICKOFF,
            }
        ]
    )


# ------------------------------------------------------------------ rosters


def test_depth_caps_produce_a_realistic_rotation():
    """The bug this guards cost 22% off every featured player's projection.

    Layer 3 normalises across whoever is on the simplex, so extra depth players
    who never see the field take share from the ones who do. Nothing errors --
    every flagged market simply comes back Under.
    """
    rosters = expected_rosters(depth_chart(), KICKOFF)
    per_team = rosters.group_by("team").len()["len"]
    low, high = EXPECTED_ROSTER_SIZE
    assert per_team.min() >= low
    assert per_team.max() <= high
    assert sum(DEPTH_LIMITS.values()) <= high


def test_caps_match_the_measured_rotation():
    """Derived from 2023-25 played games: WR 4.9, TE 3.0, RB 2.7, QB 1.2,
    FB 1.0. If these drift far from that, projections drift with them."""
    assert DEPTH_LIMITS["WR"] == 5
    assert DEPTH_LIMITS["TE"] == 3
    assert DEPTH_LIMITS["RB"] == 3
    assert 12 <= sum(DEPTH_LIMITS.values()) <= 14


def test_players_below_the_cap_are_dropped():
    rosters = expected_rosters(depth_chart(), KICKOFF)
    kept = set(rosters.filter(pl.col("team") == "AAA")["gsis_id"])
    assert "AAA-WR5" in kept
    assert "AAA-WR9" not in kept


def test_a_chart_published_after_kickoff_is_not_used():
    """Using a depth chart posted after the game is leakage, and the whole
    reason the 2025+ timestamped feed is required here."""
    late = depth_chart(published="2026-09-20T12:00:00Z")
    assert expected_rosters(late, KICKOFF).is_empty()


def test_the_undated_legacy_feed_is_refused():
    legacy = pl.DataFrame({"season": [2024], "week": [3], "club_code": ["AAA"]})
    with pytest.raises(ValueError, match="undated feed"):
        expected_rosters(legacy, KICKOFF)


# ------------------------------------------------------------ panel rows


def test_upcoming_rows_have_null_outcomes():
    """Not a placeholder -- the correct input. The tracker reads a game with
    no opportunities as 'no information', which is exactly what an unplayed
    game is, and the same path a bye takes."""
    rows = upcoming_panel_rows(schedule(), depth_chart(), season=2026, week=2)
    for column in OUTCOME_COLUMNS:
        assert rows[column].null_count() == rows.height


def test_both_teams_appear_with_the_right_opponent():
    rows = upcoming_panel_rows(schedule(), depth_chart(), season=2026, week=2)
    pairs = set(zip(rows["team"], rows["opponent"], strict=True))
    assert pairs == {("AAA", "BBB"), ("BBB", "AAA")}


def test_an_oversized_roster_raises_rather_than_biasing_quietly():
    """The failure mode is a systematic under-projection with no error. It has
    to be loud, because it is invisible in the output."""
    with pytest.raises(ValueError, match="outside the"):
        upcoming_panel_rows(
            schedule(),
            depth_chart(),
            season=2026,
            week=2,
            limits={"QB": 3, "RB": 6, "WR": 9, "TE": 5, "FB": 2},
        )


def test_an_unscheduled_week_is_an_error():
    with pytest.raises(ValueError, match="no scheduled games"):
        upcoming_panel_rows(schedule(), depth_chart(), season=2026, week=17)


def test_history_is_truncated_at_the_projected_week(synthetic_panel):
    """Anything at or after the slate would be leakage into its own features."""
    combined = panel_with_upcoming(
        synthetic_panel.with_columns(pl.lit(2026).alias("season")),
        schedule(),
        depth_chart(),
        season=2026,
        week=2,
    )
    history = combined.filter(pl.col("team_plays").is_not_null())
    assert history["week"].max() < 2


def test_upcoming_rows_carry_position_for_the_share_layer():
    """Layer 3 filters on position. Without it every upcoming player is
    ineligible and the slate silently projects nobody."""
    rows = upcoming_panel_rows(schedule(), depth_chart(), season=2026, week=2)
    assert rows["position"].null_count() == 0
    assert set(rows["position"].unique()) <= set(DEPTH_LIMITS)


# --------------------------------------------------------------- ruled out


def injury_report(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "gsis_id": pl.String,
            "report_status": pl.String,
        },
    )


def test_players_ruled_out_leave_the_simplex():
    """Layer 3 normalises across whoever is on the simplex, so a player who is
    not going to take a snap is not a harmless extra name -- he is share taken
    from the teammates who will."""
    report = injury_report(
        [{"season": 2026, "week": 2, "gsis_id": "AAA-WR1", "report_status": "Out"}]
    )
    kept = expected_rosters(depth_chart(), KICKOFF, None, report, season=2026, week=2)
    assert "AAA-WR1" not in kept["gsis_id"].to_list()
    assert "BBB-WR1" in kept["gsis_id"].to_list()


def test_doubtful_goes_but_questionable_stays():
    """Questionable players mostly play. Dropping them would move share to a
    teammate who is not going to get it either."""
    report = injury_report(
        [
            {"season": 2026, "week": 2, "gsis_id": "AAA-RB1", "report_status": "Doubtful"},
            {"season": 2026, "week": 2, "gsis_id": "AAA-TE1", "report_status": "Questionable"},
        ]
    )
    kept = expected_rosters(depth_chart(), KICKOFF, None, report, season=2026, week=2)
    ids = kept["gsis_id"].to_list()
    assert "AAA-RB1" not in ids
    assert "AAA-TE1" in ids


def test_a_designation_applies_only_to_its_own_week():
    """The trap: an unscoped anti-join deletes a player from every week of the
    season because he was out for one of them."""
    report = injury_report(
        [{"season": 2026, "week": 1, "gsis_id": "AAA-WR1", "report_status": "Out"}]
    )
    kept = expected_rosters(depth_chart(), KICKOFF, None, report, season=2026, week=2)
    assert "AAA-WR1" in kept["gsis_id"].to_list()


def test_a_designation_applies_only_to_its_own_season():
    report = injury_report(
        [{"season": 2025, "week": 2, "gsis_id": "AAA-WR1", "report_status": "Out"}]
    )
    kept = expected_rosters(depth_chart(), KICKOFF, None, report, season=2026, week=2)
    assert "AAA-WR1" in kept["gsis_id"].to_list()


def test_no_injury_report_is_not_an_error():
    """A Wednesday opener often has nothing published yet."""
    plain = expected_rosters(depth_chart(), KICKOFF)
    with_empty = expected_rosters(
        depth_chart(), KICKOFF, None, injury_report([]), season=2026, week=2
    )
    assert plain.height == with_empty.height


def test_rosters_still_pass_the_size_guard_after_trimming():
    """Trimming must not push a team under the band the guard enforces --
    that guard is what catches a share denominator going wrong."""
    report = injury_report(
        [
            {"season": 2026, "week": 2, "gsis_id": f"AAA-WR{r}", "report_status": "Out"}
            for r in (1, 2)
        ]
    )
    rows = upcoming_panel_rows(schedule(), depth_chart(), season=2026, week=2, injuries=report)
    per_team = rows.group_by("game_id", "team").len()["len"]
    low, high = EXPECTED_ROSTER_SIZE
    assert low <= float(per_team.median()) <= high
    assert "AAA-WR1" not in rows["gsis_id"].to_list()


# ------------------------------------------------- partially played weeks


def played_rows(game_id: str, team: str = "AAA") -> pl.DataFrame:
    """Real panel rows for a game that has already finished."""
    return pl.DataFrame(
        {
            "season": [2026],
            "week": [2],
            "game_id": [game_id],
            "team": [team],
            "opponent": ["BBB"],
            "gsis_id": [f"{team}-WR1"],
            "kickoff_utc": [KICKOFF],
            "season_type": ["REG"],
            "targets": [7.0],
            "receptions": [5.0],
            "carries": [0.0],
        }
    )


def two_game_schedule() -> pl.DataFrame:
    """Thursday night plus a Sunday game — a normal NFL week."""
    return pl.concat(
        [
            schedule(),
            pl.DataFrame(
                [
                    {
                        "season": 2026,
                        "week": 2,
                        "game_id": "2026_02_AAA_BBB",
                        "home_team": "BBB",
                        "away_team": "AAA",
                        "kickoff_utc": KICKOFF,
                    }
                ]
            ),
        ]
    )


def test_an_already_played_game_is_not_re_projected():
    """The week 6 bug: one Thursday game had panel rows, so the week was not
    empty, so the forward path was skipped and the other thirteen games were
    silently dropped from the report."""
    panel = played_rows("2026_02_BBB_AAA")
    combined = panel_with_upcoming(panel, two_game_schedule(), depth_chart(), season=2026, week=2)
    upcoming = combined.filter(pl.col("targets").is_null())
    assert upcoming["game_id"].unique().to_list() == ["2026_02_AAA_BBB"]


def test_the_unplayed_half_of_the_week_is_still_projected():
    panel = played_rows("2026_02_BBB_AAA")
    combined = panel_with_upcoming(panel, two_game_schedule(), depth_chart(), season=2026, week=2)
    assert not combined.filter(
        (pl.col("game_id") == "2026_02_AAA_BBB") & pl.col("targets").is_null()
    ).is_empty()


def test_a_fully_played_week_says_so_rather_than_inventing_rows():
    with pytest.raises(ValueError, match="nothing left to project"):
        upcoming_panel_rows(
            schedule(),
            depth_chart(),
            season=2026,
            week=2,
            skip_game_ids=["2026_02_BBB_AAA"],
        )
