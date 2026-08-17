"""Tests for Layer 3 (Dirichlet-Multinomial shares) and Layer 4 (catch rate)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.special import gammaln

from nfl_usage_props.model.player_share import (
    ELIGIBLE_POSITIONS,
    PlayerShareModel,
    dirichlet_multinomial_loglik,
    eligible,
    fit_catch_rate_dispersion,
)
from nfl_usage_props.model.role_tracker import logit


def make_features(
    n_team_games: int = 400,
    shares=(0.30, 0.22, 0.16, 0.12, 0.10, 0.06, 0.04),
    concentration: float = 40.0,
    seed: int = 5,
    prior_sd: float = 0.05,
) -> pl.DataFrame:
    """Team-games generated from a known Dirichlet concentration."""
    rng = np.random.default_rng(seed)
    truth = np.array(shares)
    rows = []
    for g in range(n_team_games):
        drawn = rng.dirichlet(truth * concentration)
        total = int(rng.integers(28, 40))
        counts = rng.multinomial(total, drawn)
        carries = rng.multinomial(int(rng.integers(18, 30)), drawn)
        for i, share in enumerate(truth):
            rows.append(
                {
                    "game_id": f"G{g:04d}",
                    "team": "AAA",
                    "gsis_id": f"P{i}",
                    "position": "WR" if i < 4 else "RB",
                    "targets": int(counts[i]),
                    "carries": int(carries[i]),
                    "receptions": int(rng.binomial(counts[i], 0.65)),
                    "role_target_share_mean": logit(share),
                    "role_target_share_sd": prior_sd,
                    "role_carry_share_mean": logit(share),
                    "role_carry_share_sd": prior_sd,
                    "catch_rate_prior": 0.65,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def features() -> pl.DataFrame:
    return make_features()


# ------------------------------------------------------------- eligibility


def test_linemen_are_excluded_from_the_simplex():
    """participation lists all eleven. Five linemen with a receiver's prior
    each would dilute every real share by about a third."""
    frame = pl.DataFrame({"position": ["WR", "TE", "RB", "OT", "G", "C", None]})
    assert set(eligible(frame)["position"].to_list()) == {"WR", "TE", "RB"}


def test_quarterbacks_are_eligible():
    """They carry -- 13.6% of all carries. Excluding them would redistribute
    that onto the backs."""
    assert "QB" in ELIGIBLE_POSITIONS


# ---------------------------------------------------------------- the likelihood


def test_dirichlet_multinomial_matches_the_closed_form():
    counts = np.array([5.0, 3.0, 2.0])
    alpha = np.array([2.0, 1.5, 1.0])
    starts = np.array([0, 3])
    expected = (
        gammaln(alpha.sum())
        - gammaln(counts.sum() + alpha.sum())
        + np.sum(gammaln(counts + alpha) - gammaln(alpha))
    )
    assert dirichlet_multinomial_loglik(counts, alpha, starts)[0] == pytest.approx(expected)


def test_likelihood_splits_ragged_groups_independently():
    """Rosters run 8 to 20 players. Padding to a rectangle would put phantom
    players on every simplex."""
    counts = np.array([5.0, 3.0, 4.0, 4.0, 4.0])
    alpha = np.array([2.0, 2.0, 1.0, 1.0, 1.0])
    both = dirichlet_multinomial_loglik(counts, alpha, np.array([0, 2, 5]))
    first = dirichlet_multinomial_loglik(counts[:2], alpha[:2], np.array([0, 2]))
    second = dirichlet_multinomial_loglik(counts[2:], alpha[2:], np.array([0, 3]))
    assert both[0] == pytest.approx(first[0])
    assert both[1] == pytest.approx(second[0])


def test_likelihood_vectorises_over_draws():
    counts = np.array([5.0, 3.0, 2.0])
    alpha = np.array([[2.0, 4.0], [1.5, 3.0], [1.0, 2.0]])
    stacked = dirichlet_multinomial_loglik(counts, alpha, np.array([0, 3]))
    assert stacked.shape == (1, 2)
    for column in range(2):
        single = dirichlet_multinomial_loglik(counts, alpha[:, column], np.array([0, 3]))
        assert stacked[0, column] == pytest.approx(single[0])


# ----------------------------------------------------------------------- fit


def test_recovers_a_known_concentration():
    """Generated at K=40 with a near-certain role prior, so K carries all the
    spread and the fit has one right answer."""
    frame = make_features(n_team_games=600, concentration=40.0, prior_sd=0.01, seed=11)
    fit = PlayerShareModel("targets").fit(frame)
    assert fit.converged
    assert fit.concentration == pytest.approx(40.0, rel=0.25)


def test_a_tighter_distribution_fits_a_higher_concentration():
    loose = PlayerShareModel("targets").fit(
        make_features(concentration=15.0, prior_sd=0.01, seed=2)
    )
    tight = PlayerShareModel("targets").fit(
        make_features(concentration=120.0, prior_sd=0.01, seed=2)
    )
    assert tight.concentration > loose.concentration


def test_role_uncertainty_is_integrated_out_not_double_counted(features):
    """The bug this guards: fitting K against the tracker's MEAN makes K absorb
    role uncertainty, and the sampler then adds it again. A wider role prior
    must therefore produce a HIGHER fitted K, not the same one -- more of the
    observed spread is already explained."""
    certain = make_features(prior_sd=0.01, seed=7)
    uncertain = make_features(prior_sd=0.60, seed=7)
    k_certain = PlayerShareModel("targets").fit(certain).concentration
    k_uncertain = PlayerShareModel("targets").fit(uncertain).concentration
    assert k_uncertain > k_certain


def test_layer_name_is_validated():
    with pytest.raises(ValueError, match="targets"):
        PlayerShareModel("receptions")


def test_sampling_before_fitting_is_an_error(features):
    with pytest.raises(RuntimeError, match="not fitted"):
        PlayerShareModel("targets").sample_shares(features, 10)


# -------------------------------------------------------------------- sample


@pytest.fixture(scope="module")
def fitted(features):
    model = PlayerShareModel("targets")
    model.fit(features)
    return model


def test_sampled_shares_sum_to_one_within_every_team_game(fitted, features):
    """The simplex constraint is the entire reason for a Dirichlet. Without it
    a team can be projected throwing 130% of its passes."""
    keys, shares = fitted.sample_shares(features, 200)
    totals = (
        keys.with_row_index("_i")
        .group_by("game_id", "team")
        .agg(pl.col("_i").min().alias("start"), pl.len().alias("size"))
    )
    for row in totals.head(20).to_dicts():
        block = shares[row["start"] : row["start"] + row["size"]]
        assert block.sum(axis=0) == pytest.approx(np.ones(block.shape[1]))


def test_sampled_shares_centre_on_the_priors(fitted, features):
    keys, shares = fitted.sample_shares(features, 500)
    mean_share = shares.mean(axis=1)
    by_player = (
        keys.with_columns(pl.Series("sampled", mean_share))
        .group_by("gsis_id")
        .agg(pl.col("sampled").mean())
    )
    lookup = dict(zip(by_player["gsis_id"], by_player["sampled"], strict=True))
    assert lookup["P0"] == pytest.approx(0.30, abs=0.04)
    assert lookup["P6"] == pytest.approx(0.04, abs=0.03)


def test_a_wider_role_prior_produces_wider_shares(fitted):
    """Role uncertainty must reach the projection. If it does not, a receiver
    whose situation just changed gets a confident wrong number."""
    certain = make_features(prior_sd=0.01, seed=3)
    uncertain = make_features(prior_sd=1.20, seed=3)
    model = PlayerShareModel("targets")
    model.fit(certain)
    _, tight = model.sample_shares(certain, 400)
    _, loose = model.sample_shares(uncertain, 400)
    assert loose.std(axis=1).mean() > tight.std(axis=1).mean()


def test_keys_carry_what_downstream_layers_need(fitted, features):
    keys, _ = fitted.sample_shares(features, 20)
    for column in ("game_id", "team", "gsis_id", "position", "catch_rate_prior"):
        assert column in keys.columns


# ------------------------------------------------------------- catch rate


def test_binomial_catch_data_reports_no_finite_dispersion(features):
    """The fixture draws receptions as a pure Binomial, so there IS no finite
    Beta-Binomial dispersion. Hitting the bound and saying so is correct --
    reporting a confident number there would be the bug."""
    fit = fit_catch_rate_dispersion(features)
    assert not fit.converged


def test_overdispersed_catch_data_fits_cleanly():
    rng = np.random.default_rng(9)
    rows = []
    for _ in range(4000):
        targets = int(rng.integers(4, 12))
        rate = rng.beta(0.65 * 25, 0.35 * 25)
        rows.append(
            {
                "position": "WR",
                "targets": targets,
                "receptions": int(rng.binomial(targets, rate)),
                "catch_rate_prior": 0.65,
            }
        )
    fit = fit_catch_rate_dispersion(pl.DataFrame(rows))
    assert fit.converged
    assert fit.concentration == pytest.approx(25.0, rel=0.5)


def test_catch_rate_dispersion_rises_with_a_tighter_distribution():
    """A player whose catch rate never varies should fit a large phi."""
    rng = np.random.default_rng(4)
    rows = []
    for _ in range(3000):
        targets = int(rng.integers(4, 12))
        rows.append(
            {
                "position": "WR",
                "targets": targets,
                "receptions": int(rng.binomial(targets, 0.65)),
                "catch_rate_prior": 0.65,
            }
        )
    binomial = fit_catch_rate_dispersion(pl.DataFrame(rows))

    noisy = []
    for _ in range(3000):
        targets = int(rng.integers(4, 12))
        rate = rng.beta(0.65 * 8, 0.35 * 8)
        noisy.append(
            {
                "position": "WR",
                "targets": targets,
                "receptions": int(rng.binomial(targets, rate)),
                "catch_rate_prior": 0.65,
            }
        )
    overdispersed = fit_catch_rate_dispersion(pl.DataFrame(noisy))
    assert binomial.concentration > overdispersed.concentration
