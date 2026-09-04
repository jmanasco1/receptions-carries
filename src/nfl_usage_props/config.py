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
class SnapshotConfig:
    """Odds snapshot cadence. See config.toml for the credit budget."""

    close_cutoff_minutes: int = 25
    horizon_hours: int = 192
    max_repoll_events: int = 4


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
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)


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
class EdgeConfig:
    devig_method: str = "power"
    devig_methods_logged: tuple[str, ...] = ("power", "multiplicative", "shin")
    consensus_book_weights: dict[str, float] = field(default_factory=dict)
    consensus_default_weight: float = 1.0
    consensus_min_books: int = 3
    min_edge: float = 0.03
    stale_price_confirmations: int = 2
    extreme_price_threshold: int = -300

    def book_weight(self, book_key: str) -> float:
        return self.consensus_book_weights.get(book_key, self.consensus_default_weight)


@dataclass(frozen=True)
class RoleTrackerConfig:
    """State-space role tracker. Replaces fixed exponential decay entirely."""

    # Derived, not guessed: q = R / (m(m-1)) at each layer's target effective
    # memory. See `model.role_tracker.calibrated_process_noise`, which a test
    # holds these to.
    process_noise_snap_share: float = 0.00599
    process_noise_carry_share: float = 0.00625
    process_noise_target_share: float = 0.00355
    process_noise_event_multiplier: float = 6.0
    partial_game_exposure_ratio: float = 0.55
    changepoint_lookback_games: int = 2
    changepoint_threshold_sd: float = 2.5

    def process_noise(self, layer: str) -> float:
        try:
            return getattr(self, f"process_noise_{layer}")
        except AttributeError as exc:
            raise KeyError(
                f"no process noise configured for layer {layer!r}; "
                "expected one of snap_share, carry_share, target_share"
            ) from exc


@dataclass(frozen=True)
class CatchRateConfig:
    prior_strength_targets: float = 50.0


@dataclass(frozen=True)
class EarlySeasonConfig:
    prior_season_weight_week1: float = 0.75
    prior_season_zero_by_week: int = 6
    context_change_discount: float = 0.35
    rookie_preseason_weight: float = 0.0
    # Week 1 only. Weeks 2-3 measured as calibrated on held-out data and no
    # less accurate than mid-season; see the table in config.toml.
    suppress_output_before_week: int = 2


@dataclass(frozen=True)
class ModelConfig:
    monte_carlo_draws: int = 10000
    random_seed: int = 20240901
    # Weeks to lag the LEGACY (<=2024) depth chart feed. The 2025+ feed has a
    # `dt` timestamp and is filtered exactly instead. 0 accepts the leak.
    depth_chart_lag_weeks: int = 1
    role_tracker: RoleTrackerConfig = field(default_factory=RoleTrackerConfig)
    catch_rate: CatchRateConfig = field(default_factory=CatchRateConfig)
    early_season: EarlySeasonConfig = field(default_factory=EarlySeasonConfig)


@dataclass(frozen=True)
class OutputConfig:
    """What lands in the weekly report.

    `scope` filters the REPORT only. Layer 3 is a Dirichlet-Multinomial over the
    full roster whose shares must sum to one, so estimation always covers every
    rostered player regardless of this setting.
    """

    scope: str = "posted_line_only"

    VALID_SCOPES = ("posted_line_only", "all_rostered")

    def __post_init__(self) -> None:
        if self.scope not in self.VALID_SCOPES:
            raise ValueError(f"output.scope must be one of {self.VALID_SCOPES}, got {self.scope!r}")

    @property
    def posted_line_only(self) -> bool:
        return self.scope == "posted_line_only"


@dataclass(frozen=True)
class Config:
    data_dir: Path
    ingest: IngestConfig
    storage: StorageConfig
    odds: OddsConfig
    edge: EdgeConfig
    model: ModelConfig
    output: OutputConfig
    source_path: Path

    @property
    def props_dir(self) -> Path:
        """Parsed prop snapshots. The CLV record — never overwritten."""
        return self.data_dir / "odds" / "props"

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
        snapshots=SnapshotConfig(**odds_raw.get("snapshots", {})),
    )

    edge_raw = dict(raw.get("edge", {}))
    if "devig_methods_logged" in edge_raw:
        edge_raw["devig_methods_logged"] = tuple(edge_raw["devig_methods_logged"])
    edge = EdgeConfig(**edge_raw)

    model_raw = dict(raw.get("model", {}))
    model = ModelConfig(
        **{k: v for k, v in model_raw.items() if not isinstance(v, dict)},
        role_tracker=RoleTrackerConfig(**model_raw.get("role_tracker", {})),
        catch_rate=CatchRateConfig(**model_raw.get("catch_rate", {})),
        early_season=EarlySeasonConfig(**model_raw.get("early_season", {})),
    )
    output = OutputConfig(**raw.get("output", {}))

    return Config(
        data_dir=data_dir,
        ingest=ingest,
        storage=storage,
        odds=odds,
        edge=edge,
        model=model,
        output=output,
        source_path=path,
    )
