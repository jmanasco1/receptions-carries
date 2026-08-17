"""Layers 1 and 2: how many plays a team runs, and how many of them are passes.

These two layers are where market information belongs. The book knows the game
environment — it prices the total and the spread, and those are the best public
estimates of pace and script that exist. It does *not* know the target share.
So the spread and total enter here and never at Layer 3, which is the whole
argument for a hierarchy over a flat regression that would let the closing line
explain a receiver's role.

    Layer 1   team offensive plays        Negative Binomial, log link
    Layer 2   dropbacks out of those      Beta-Binomial, logit link

Layer 2 is modelled as a *rate*, not a count, because Layer 3 samples player
shares out of the resulting opportunities. Sampling counts here and again
downstream would apply the binomial noise twice and inflate every tail.

How much these layers are actually worth
---------------------------------------
Not much, and that is worth stating plainly rather than discovering later.
Held out on 2024-25, Layer 1 predicts team plays at MAE 6.70 against 6.83 for
simply predicting the league mean -- a 2% improvement. Team play counts are
close to unpredictable at the game level. Layer 2 does better, MAE 0.0784 on
the pass rate against 0.0838 for a constant rate, roughly 7%.

That is not a reason to drop them. Both are well calibrated out of sample
(50/80/95 intervals covering 0.51/0.81/0.95), and a calibrated distribution
centred near the mean is exactly what the Monte Carlo needs -- the variance is
the contribution, not the mean. But it does mean the edge in this project lives
at Layer 3, where the role tracker predicts target share at r=0.81. Anyone
tempted to spend a week tuning pace features should read these numbers first.

On wind
-------
`wind_observed` is measured during the game, so it can be fitted against but
never read on a Sunday morning. The model therefore fits with and without it:
the wind coefficient is reported so its size is known, and the production
specification excludes it until Stage 3 supplies a pre-kickoff forecast.
Fitting a coefficient you cannot feed at prediction time is not a model, it is
a number that makes the backtest look better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_usage_props.model.glm import BetaBinomialGLM, FitResult, NegativeBinomialGLM

# Pace. A team's own trailing pace and its opponent's, plus what the market
# says about the game. `abs(spread)` rather than `spread`: a blowout shortens
# the game for both teams, and which side is ahead does not change that.
PLAYS_FEATURES: tuple[str, ...] = (
    "trailing_team_plays",
    "def_plays_per_game_allowed",
    "total_line",
    "abs_spread",
    "rest",
    "is_home",
)

# Script and tendency. `spread` signed this time -- a trailing team throws, and
# which side is trailing is exactly the point.
PASS_RATE_FEATURES: tuple[str, ...] = (
    "spread",
    "trailing_team_proe",
    "def_pass_rate_allowed",
    "trailing_pass_rate",
    "total_line",
)

# Fitted for its coefficient, not used in production until a forecast exists.
WIND_FEATURE = "wind_observed"


@dataclass
class TeamVolumeFit:
    plays: FitResult
    pass_rate: FitResult
    wind_pass_rate: FitResult | None = None

    @property
    def wind_coefficient(self) -> float | None:
        """Log-odds change in pass rate per mph. Expected to be negative."""
        if self.wind_pass_rate is None:
            return None
        names = self.wind_pass_rate.feature_names
        return float(self.wind_pass_rate.coefficients[names.index(WIND_FEATURE)])


def team_game_frame(features: pl.DataFrame) -> pl.DataFrame:
    """Collapse the player-game feature matrix to one row per team-game."""
    columns = [
        "season",
        "week",
        "season_type",
        "game_id",
        "team",
        "opponent",
        "kickoff_utc",
        "team_plays",
        "team_dropbacks",
        *PLAYS_FEATURES[:-2],
        *[c for c in PASS_RATE_FEATURES if c not in PLAYS_FEATURES],
        "rest",
        "is_home",
        "spread",
        WIND_FEATURE,
    ]
    available = [c for c in dict.fromkeys(columns) if c in features.columns]

    frame = (
        features.with_columns(
            pl.col("spread").abs().alias("abs_spread"),
            (pl.col("trailing_team_dropbacks") / pl.col("trailing_team_plays")).alias(
                "trailing_pass_rate"
            ),
        )
        .select(*available, "abs_spread", "trailing_pass_rate")
        .unique(subset=["game_id", "team"])
        .sort("season", "week", "game_id", "team")
    )
    return frame


class TeamVolumeModel:
    """Layers 1 and 2, fitted together and sampled together."""

    def __init__(self, *, seed: int = 20240901):
        self.seed = seed
        self.fit_result: TeamVolumeFit | None = None
        # The fitted estimators are kept, not rebuilt from their coefficients:
        # each one also carries the feature scaling it was fitted with, and
        # reconstructing one without that silently predicts on raw inputs.
        self._plays_model: NegativeBinomialGLM | None = None
        self._rate_model: BetaBinomialGLM | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, team_games: pl.DataFrame, *, fit_wind: bool = True) -> TeamVolumeFit:
        usable = self._usable(team_games, PLAYS_FEATURES)
        self._plays_model = NegativeBinomialGLM(PLAYS_FEATURES)
        plays = self._plays_model.fit(
            _matrix(usable, PLAYS_FEATURES), usable["team_plays"].to_numpy()
        )

        usable_rate = self._usable(team_games, PASS_RATE_FEATURES)
        self._rate_model = BetaBinomialGLM(PASS_RATE_FEATURES)
        pass_rate = self._rate_model.fit(
            _matrix(usable_rate, PASS_RATE_FEATURES),
            usable_rate["team_dropbacks"].to_numpy(),
            usable_rate["team_plays"].to_numpy(),
        )

        wind_fit = None
        if fit_wind and WIND_FEATURE in team_games.columns:
            with_wind = (*PASS_RATE_FEATURES, WIND_FEATURE)
            usable_wind = self._usable(team_games, with_wind)
            if usable_wind.height > 100:
                wind_fit = BetaBinomialGLM(with_wind).fit(
                    _matrix(usable_wind, with_wind),
                    usable_wind["team_dropbacks"].to_numpy(),
                    usable_wind["team_plays"].to_numpy(),
                )

        self.fit_result = TeamVolumeFit(plays=plays, pass_rate=pass_rate, wind_pass_rate=wind_fit)
        return self.fit_result

    @staticmethod
    def _usable(frame: pl.DataFrame, features: tuple[str, ...]) -> pl.DataFrame:
        """Rows with every feature and both outcomes present.

        Week 1 of the very first season has no trailing history and is dropped
        rather than imputed: one game-week of missing pace is not worth an
        imputation rule that would then apply silently everywhere else.
        """
        needed = [*features, "team_plays", "team_dropbacks"]
        return frame.drop_nulls([c for c in needed if c in frame.columns])

    # -------------------------------------------------------------- predict

    def predict(self, team_games: pl.DataFrame) -> pl.DataFrame:
        """Expected plays, dropbacks and pass rate per team-game."""
        self._require_fit()
        frame = self._usable(team_games, PLAYS_FEATURES)
        plays = self._plays_model.predict_mean(_matrix(frame, PLAYS_FEATURES))
        rate = self._rate_model.predict_rate(_matrix(frame, PASS_RATE_FEATURES))
        return frame.select("game_id", "team").with_columns(
            pl.Series("expected_plays", plays),
            pl.Series("expected_pass_rate", rate),
            pl.Series("expected_dropbacks", plays * rate),
            pl.Series("expected_rush_attempts", plays * (1.0 - rate)),
        )

    def sample(self, team_games: pl.DataFrame, draws: int = 10_000) -> dict[str, np.ndarray]:
        """Joint draws of plays, dropbacks and rush attempts.

        Plays and the pass rate are drawn independently and then combined, so
        the dropback count inherits both sources of variance -- pace and script
        -- rather than only one.
        """
        self._require_fit()
        rng = np.random.default_rng(self.seed)
        frame = self._usable(team_games, PLAYS_FEATURES)

        plays = self._plays_model.sample(_matrix(frame, PLAYS_FEATURES), draws, rng)
        rate = self._rate_model.sample_rate(_matrix(frame, PASS_RATE_FEATURES), draws, rng)
        dropbacks = rng.binomial(plays, rate)
        return {
            "game_id": frame["game_id"].to_numpy(),
            "team": frame["team"].to_numpy(),
            "plays": plays,
            "dropbacks": dropbacks,
            "rush_attempts": plays - dropbacks,
        }

    def _require_fit(self) -> None:
        if self.fit_result is None or self._plays_model is None or self._rate_model is None:
            raise RuntimeError("model is not fitted; call fit() first")


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    """Feature columns as a float matrix, booleans cast to 0/1."""
    return np.column_stack([frame[name].cast(pl.Float64).to_numpy() for name in features])


def split_by_season(
    frame: pl.DataFrame, holdout_seasons: list[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train/test split on whole seasons.

    Whole seasons, not random rows: rows within a season share team-level
    history, so a random split lets the training set see games that inform the
    test set's features and reports a score the model will not reproduce.
    """
    holdout = set(holdout_seasons)
    return (
        frame.filter(~pl.col("season").is_in(holdout)),
        frame.filter(pl.col("season").is_in(holdout)),
    )
