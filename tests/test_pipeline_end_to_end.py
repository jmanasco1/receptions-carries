"""The whole chain on real data: features -> projection -> price -> report.

Every other test exercises one stage. This one checks the seams, and the seams
are where this will actually break -- the model keys players on `gsis_id` while
a book posts a name, and if that join silently produces nothing the report is
empty and looks merely quiet.

Prop snapshots cannot be backfilled, so the market side is synthesised: a line
at the model's own median, priced -110 both ways. That is enough to prove the
joins line up and the arithmetic flows. It proves nothing whatsoever about
edges, and is not trying to.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_usage_props.config import load_config
from nfl_usage_props.edge.clv import score_flagged_bets, summarise
from nfl_usage_props.edge.edges import compute_edges
from nfl_usage_props.features.build import build_features
from nfl_usage_props.features.dataset import load_panel, load_schedules
from nfl_usage_props.model.player_share import PlayerShareModel, fit_catch_rate_dispersion
from nfl_usage_props.model.projection import project_week
from nfl_usage_props.model.team_volume import TeamVolumeModel, split_by_season, team_game_frame
from nfl_usage_props.report import render_markdown
from nfl_usage_props.storage import ParquetStore

pytestmark = pytest.mark.network

SEASONS = [2022, 2023, 2024]
SLATE_SEASON, SLATE_WEEK = 2024, 10


@pytest.fixture(scope="module")
def projected():
    config = load_config(load_env=False)
    store = ParquetStore(config.raw_dir)
    for table in ("pbp", "participation", "schedules"):
        if any(not store.exists(table, s) for s in SEASONS):
            pytest.skip(f"{table} not ingested; run `nfl-props ingest run`")

    panel = load_panel(config, SEASONS)
    features = build_features(panel, load_schedules(config, SEASONS), config).filter(
        pl.col("season_type") == "REG"
    )
    history, _ = split_by_season(features, [SLATE_SEASON])
    slate = features.filter((pl.col("season") == SLATE_SEASON) & (pl.col("week") == SLATE_WEEK))
    slate_teams = team_game_frame(features).filter(
        (pl.col("season") == SLATE_SEASON) & (pl.col("week") == SLATE_WEEK)
    )

    volume = TeamVolumeModel(seed=3)
    volume.fit(team_game_frame(history), fit_wind=False)
    targets = PlayerShareModel("targets", seed=3)
    targets.fit(history)
    carries = PlayerShareModel("carries", seed=3)
    carries.fit(history)
    catch = fit_catch_rate_dispersion(history)

    projection = project_week(
        slate, slate_teams, volume, targets, carries, catch.concentration, draws=1500, seed=3
    )
    return config, projection, slate


def synthetic_snapshot(projection, n_players: int = 60, min_projection: float = 2.0):
    """A market posted around each player's projected usage.

    Only players projected for `min_projection` receptions get a line, because
    that is what books do. Posting a 1.5 line on someone projected for 0.2 --
    which an unfiltered fixture happily does -- manufactures a 47-point edge
    out of a market nobody would offer, and then the test is measuring the
    fixture rather than the pipeline.
    """
    means = projection.receptions.mean(axis=1)
    keep = np.flatnonzero(means >= min_projection)[:n_players]
    keys = projection.keys[keep]
    rows = []
    for i, row in zip(keep, keys.to_dicts(), strict=True):
        point = round(means[i] * 2) / 2
        point = point + 0.5 if float(point).is_integer() else point
        for book in ("alpha", "beta", "gamma"):
            for side in ("Over", "Under"):
                rows.append(
                    {
                        "event_id": row["game_id"],
                        "market": "player_receptions",
                        "player_name_raw": row["gsis_id"],
                        "gsis_id": row["gsis_id"],
                        "point": point,
                        "bookmaker": book,
                        "side": side,
                        "price": -110,
                        "snapshot_kind": "open",
                        "fetched_at": "2024-11-06T18:00:00Z",
                    }
                )
    return pl.DataFrame(rows)


def test_the_projection_covers_the_slate(projected):
    _, projection, slate = projected
    assert projection.keys.height > 200
    assert set(projection.keys["game_id"].unique()) <= set(slate["game_id"].unique())


def test_the_model_joins_to_a_snapshot_on_gsis_id(projected):
    """The seam most likely to fail in production. A broken join here produces
    an empty report that looks like a quiet week."""
    config, projection, _ = projected
    edges = compute_edges(projection, synthetic_snapshot(projection), config, week=SLATE_WEEK)
    assert not edges.is_empty()
    assert edges.height >= 20


def test_every_priced_row_gets_a_model_probability(projected):
    config, projection, _ = projected
    edges = compute_edges(projection, synthetic_snapshot(projection), config, week=SLATE_WEEK)
    assert edges["model_probability"].null_count() == 0
    assert edges["model_probability"].min() >= 0.0
    assert edges["model_probability"].max() <= 1.0


def test_the_model_finds_no_large_edge_against_its_own_median(projected):
    """Lines set at the model's own mean, priced -110 both ways, is close to a
    market the model agrees with. Large systematic edges here would mean the
    devig or the probability lookup is wired wrong, not that money is free."""
    config, projection, _ = projected
    edges = compute_edges(projection, synthetic_snapshot(projection), config, week=SLATE_WEEK)
    assert edges["edge"].median() < 0.15


def test_the_report_renders_from_real_projections(projected):
    config, projection, _ = projected
    edges = compute_edges(projection, synthetic_snapshot(projection), config, week=SLATE_WEEK)
    rendered = render_markdown(edges, week=SLATE_WEEK, season=SLATE_SEASON)
    assert "Usage props" in rendered
    assert "What was held back" in rendered


def test_clv_reports_honestly_with_no_closing_prices(projected):
    """Only opening prices exist here, so there is nothing to score against.
    The summary has to say so rather than inventing a number."""
    config, projection, _ = projected
    snapshot = synthetic_snapshot(projection)
    edges = compute_edges(projection, snapshot, config, week=SLATE_WEEK)
    assert summarise(score_flagged_bets(edges, snapshot))["n"] == 0
