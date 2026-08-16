from __future__ import annotations

import json
from pathlib import Path

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
