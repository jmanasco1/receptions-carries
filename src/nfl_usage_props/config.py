"""Configuration loading.

Config lives in a single TOML file. Secrets live in `.env` and are never read
from the TOML. `load_config()` is the only entry point.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.toml"


def current_nfl_season(now: datetime | None = None) -> int:
    """The season currently in progress.

    NFL seasons are labelled by the calendar year they start in, and run from
    September into February. Anything before March belongs to the prior
    season's label.
    """
    now = now or datetime.now(UTC)
    return now.year - 1 if now.month < 3 else now.year


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 5
    backoff_initial_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 32.0
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class CreditConfig:
    floor: int = 50
    max_spend_per_run: int = 120


@dataclass(frozen=True)
class OddsConfig:
    base_url: str = "https://api.the-odds-api.com/v4"
    sport: str = "americanfootball_nfl"
    regions: str = "us"
    odds_format: str = "american"
    game_markets: tuple[str, ...] = ("spreads", "totals")
    prop_markets: tuple[str, ...] = ("player_receptions", "player_rush_attempts")
    credits: CreditConfig = field(default_factory=CreditConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass(frozen=True)
class IngestConfig:
    first_season: int = 2016
    last_season: int | None = None
    tables: tuple[str, ...] = ()

    def seasons(self, now: datetime | None = None) -> list[int]:
        last = self.last_season or current_nfl_season(now)
        return list(range(self.first_season, last + 1))


@dataclass(frozen=True)
class StorageConfig:
    compression: str = "zstd"
    completion_lag_days: int = 30


@dataclass(frozen=True)
class ModelConfig:
    share_half_life_games: float = 6.0
    monte_carlo_draws: int = 10000
    random_seed: int = 20240901


@dataclass(frozen=True)
class Config:
    data_dir: Path
    ingest: IngestConfig
    storage: StorageConfig
    odds: OddsConfig
    model: ModelConfig
    source_path: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"

    @property
    def odds_dir(self) -> Path:
        return self.data_dir / "odds"

    @property
    def metadata_db(self) -> Path:
        return self.data_dir / "metadata.sqlite"

    def odds_api_key(self) -> str | None:
        """Read the API key from the environment. Never from the TOML."""
        key = os.environ.get("ODDS_API_KEY", "").strip()
        return key or None


def load_config(path: str | Path | None = None, *, load_env: bool = True) -> Config:
    """Load configuration from TOML, applying environment overrides."""
    if load_env:
        load_dotenv(REPO_ROOT / ".env")

    if path is None:
        path = os.environ.get("NFL_PROPS_CONFIG") or DEFAULT_CONFIG_PATH
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    project = raw.get("project", {})
    data_dir = os.environ.get("NFL_PROPS_DATA_DIR") or project.get("data_dir", "data")
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = (path.parent / data_dir).resolve()

    ingest_raw = raw.get("ingest", {})
    ingest = IngestConfig(
        first_season=int(ingest_raw.get("first_season", 2016)),
        last_season=ingest_raw.get("last_season"),
        tables=tuple(ingest_raw.get("tables", ())),
    )

    storage_raw = raw.get("storage", {})
    storage = StorageConfig(
        compression=storage_raw.get("compression", "zstd"),
        completion_lag_days=int(storage_raw.get("completion_lag_days", 30)),
    )

    odds_raw = raw.get("odds", {})
    odds = OddsConfig(
        base_url=odds_raw.get("base_url", "https://api.the-odds-api.com/v4").rstrip("/"),
        sport=odds_raw.get("sport", "americanfootball_nfl"),
        regions=odds_raw.get("regions", "us"),
        odds_format=odds_raw.get("odds_format", "american"),
        game_markets=tuple(odds_raw.get("game_markets", ("spreads", "totals"))),
        prop_markets=tuple(odds_raw.get("prop_markets", ())),
        credits=CreditConfig(**odds_raw.get("credits", {})),
        retry=RetryConfig(**odds_raw.get("retry", {})),
    )

    model = ModelConfig(**raw.get("model", {}))

    return Config(
        data_dir=data_dir,
        ingest=ingest,
        storage=storage,
        odds=odds,
        model=model,
        source_path=path,
    )
