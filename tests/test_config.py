from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nfl_usage_props.config import DEFAULT_CONFIG_PATH, current_nfl_season, load_config


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (datetime(2024, 9, 8, tzinfo=UTC), 2024),
        (datetime(2024, 12, 31, tzinfo=UTC), 2024),
        (datetime(2025, 1, 15, tzinfo=UTC), 2024),  # playoffs belong to 2024
        (datetime(2025, 2, 9, tzinfo=UTC), 2024),  # Super Bowl
        (datetime(2025, 3, 1, tzinfo=UTC), 2025),  # league year rolls over
    ],
)
def test_current_nfl_season(when, expected):
    assert current_nfl_season(when) == expected


def test_repo_config_loads():
    config = load_config(DEFAULT_CONFIG_PATH, load_env=False)
    assert config.ingest.first_season == 2016
    assert "pbp" in config.ingest.tables
    assert config.odds.prop_markets == ("player_receptions", "player_rush_attempts")


def test_credit_floor_exceeds_a_full_slate_pull():
    """A floor below ~32 lets a run abort mid-slate, stranding credits."""
    config = load_config(DEFAULT_CONFIG_PATH, load_env=False)
    assert config.odds.credits.floor >= 32


def test_seasons_track_the_current_season_when_unpinned():
    """`last_season` is commented out in config.toml, so the range must follow
    the calendar rather than needing a yearly edit."""
    config = load_config(DEFAULT_CONFIG_PATH, load_env=False)
    assert config.ingest.last_season is None
    seasons = config.ingest.seasons(datetime(2025, 6, 1, tzinfo=UTC))
    assert seasons[0] == 2016
    assert seasons[-1] == 2025


def test_data_dir_resolves_relative_to_config_file():
    config = load_config(DEFAULT_CONFIG_PATH, load_env=False)
    assert config.data_dir.is_absolute()
    assert config.raw_dir == config.data_dir / "raw"
    assert config.metadata_db == config.data_dir / "metadata.sqlite"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.toml", load_env=False)


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NFL_PROPS_DATA_DIR", str(tmp_path / "elsewhere"))
    assert load_config(DEFAULT_CONFIG_PATH, load_env=False).data_dir == tmp_path / "elsewhere"


def test_api_key_read_from_env_not_toml(monkeypatch):
    config = load_config(DEFAULT_CONFIG_PATH, load_env=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert config.odds_api_key() is None
    monkeypatch.setenv("ODDS_API_KEY", "  secret  ")
    assert config.odds_api_key() == "secret"


def test_blank_api_key_is_none(monkeypatch):
    """.env.example ships `ODDS_API_KEY=` — that must read as unset."""
    monkeypatch.setenv("ODDS_API_KEY", "   ")
    assert load_config(DEFAULT_CONFIG_PATH, load_env=False).odds_api_key() is None


def test_config_toml_contains_no_secrets():
    text = DEFAULT_CONFIG_PATH.read_text().lower()
    assert "api_key" not in text.replace("odds_api_key", "")
