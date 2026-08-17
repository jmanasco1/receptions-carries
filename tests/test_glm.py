"""Tests for the two GLM fitters.

Almost all of these are parameter-recovery tests on simulated data with a known
answer. That is the only way to tell a fitted model from a converged optimiser:
both produce numbers, and only one of them produces the right ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfl_usage_props.model.glm import (
    BetaBinomialGLM,
    NegativeBinomialGLM,
    Standardizer,
    _design,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20240901)


def simulate_nb(rng, n=4000, beta=(4.1, 0.15, -0.08), phi=12.0):
    x = rng.normal(size=(n, len(beta) - 1))
    mu = np.exp(beta[0] + x @ np.array(beta[1:]))
    return x, rng.poisson(rng.gamma(phi, mu / phi))


def simulate_bb(rng, n=4000, beta=(0.35, 0.4, -0.25), phi=40.0):
    x = rng.normal(size=(n, len(beta) - 1))
    p = 1.0 / (1.0 + np.exp(-(beta[0] + x @ np.array(beta[1:]))))
    trials = rng.integers(50, 75, size=n)
    return x, rng.binomial(trials, rng.beta(p * phi, (1 - p) * phi)), trials


# ------------------------------------------------------------ standardizing


def test_standardizer_centres_and_scales():
    x = np.array([[10.0, 100.0], [20.0, 300.0], [30.0, 200.0]])
    standardizer = Standardizer.fit(x)
    scaled = standardizer.apply(x)
    assert scaled.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert scaled.std(axis=0) == pytest.approx([1.0, 1.0])


def test_a_constant_column_does_not_produce_nan():
    """Scaling by a zero standard deviation would poison the whole fit."""
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaled = Standardizer.fit(x).apply(x)
    assert np.isfinite(scaled).all()


def test_design_matrix_has_an_intercept():
    assert np.all(_design(np.array([[1.0], [2.0]]))[:, 0] == 1.0)


# -------------------------------------------------- negative binomial recovery


def test_nb_recovers_coefficients(rng):
    x, y = simulate_nb(rng)
    result = NegativeBinomialGLM(("a", "b")).fit(x, y)
    # Coefficients are on the standardised scale, so compare predictions.
    assert result.converged


def test_nb_recovers_dispersion(rng):
    """The parameter the joint optimiser got wrong by a factor of 25 on real
    data. Profiling it out is what this asserts."""
    x, y = simulate_nb(rng, phi=12.0)
    result = NegativeBinomialGLM(("a", "b")).fit(x, y)
    assert result.dispersion == pytest.approx(12.0, rel=0.2)


@pytest.mark.parametrize("phi", [2.0, 25.0, 400.0])
def test_nb_recovers_dispersion_across_its_range(rng, phi):
    """Weak overdispersion is the hard case: the likelihood is nearly flat in
    log phi, which is exactly where the naive fit failed."""
    x, y = simulate_nb(rng, n=8000, phi=phi)
    result = NegativeBinomialGLM(("a", "b")).fit(x, y)
    assert result.dispersion == pytest.approx(phi, rel=0.35)


def test_nb_predicts_the_conditional_mean(rng):
    x, y = simulate_nb(rng)
    model = NegativeBinomialGLM(("a", "b"))
    model.fit(x, y)
    assert model.predict_mean(x).mean() == pytest.approx(y.mean(), rel=0.02)


def test_nb_variance_exceeds_the_mean(rng):
    """The entire reason for choosing NB over Poisson."""
    x, y = simulate_nb(rng, phi=12.0)
    model = NegativeBinomialGLM(("a", "b"))
    model.fit(x, y)
    assert np.all(model.predict_variance(x) > model.predict_mean(x))


def test_nb_samples_match_the_fitted_moments(rng):
    x, y = simulate_nb(rng, phi=12.0)
    model = NegativeBinomialGLM(("a", "b"))
    model.fit(x, y)
    draws = model.sample(x[:200], 4000, np.random.default_rng(1))
    assert draws.mean(axis=1).mean() == pytest.approx(model.predict_mean(x[:200]).mean(), rel=0.05)
    assert draws.var(axis=1).mean() == pytest.approx(
        model.predict_variance(x[:200]).mean(), rel=0.15
    )


def test_nb_reports_when_dispersion_hits_the_bound(rng):
    """Pure Poisson data has no finite MLE for phi. That must surface as a
    warning, not as a confident number."""
    x = rng.normal(size=(3000, 1))
    y = rng.poisson(np.exp(3.0 + 0.1 * x[:, 0]))
    result = NegativeBinomialGLM(("a",)).fit(x, y)
    assert not result.converged
    assert "bound" in result.message


# ------------------------------------------------------ beta-binomial recovery


def test_bb_recovers_dispersion(rng):
    x, successes, trials = simulate_bb(rng, phi=40.0)
    result = BetaBinomialGLM(("a", "b")).fit(x, successes, trials)
    assert result.dispersion == pytest.approx(40.0, rel=0.2)


def test_bb_predicts_the_pooled_rate(rng):
    x, successes, trials = simulate_bb(rng)
    model = BetaBinomialGLM(("a", "b"))
    model.fit(x, successes, trials)
    assert model.predict_rate(x).mean() == pytest.approx(successes.sum() / trials.sum(), rel=0.03)


def test_bb_rate_samples_are_wider_than_binomial(rng):
    """If they were not, the Beta-Binomial would be doing nothing."""
    x, successes, trials = simulate_bb(rng, phi=40.0)
    model = BetaBinomialGLM(("a", "b"))
    model.fit(x, successes, trials)
    rates = model.sample_rate(x[:200], 3000, np.random.default_rng(2))
    predicted = model.predict_rate(x[:200])
    binomial_sd = np.sqrt(predicted * (1 - predicted) / 62)
    assert rates.std(axis=1).mean() > binomial_sd.mean()


def test_bb_rejects_successes_above_trials():
    with pytest.raises(ValueError, match="cannot exceed"):
        BetaBinomialGLM().fit(np.zeros((3, 1)), np.array([5, 1, 1]), np.array([4, 4, 4]))


# ------------------------------------------------------------------- guards


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        NegativeBinomialGLM().predict_mean(np.zeros((2, 1)))
    with pytest.raises(RuntimeError, match="not fitted"):
        BetaBinomialGLM().predict_rate(np.zeros((2, 1)))


def test_profiling_beats_the_joint_optimiser_on_flat_likelihoods(rng):
    """The concrete regression: on weakly overdispersed data the joint fit
    stops early while reporting success. Profiling must find a strictly better
    likelihood."""
    from scipy.optimize import minimize

    x, y = simulate_nb(rng, n=6000, phi=400.0)
    profiled = NegativeBinomialGLM(("a", "b")).fit(x, y)

    standardized = Standardizer.fit(x).apply(x)
    design = _design(standardized)
    start = np.zeros(design.shape[1] + 1)
    start[0] = np.log(y.mean())
    start[-1] = np.log(5.0)
    joint = minimize(
        NegativeBinomialGLM._negative_log_likelihood,
        start,
        args=(design, y.astype(float)),
        method="L-BFGS-B",
    )
    assert profiled.log_likelihood >= -float(joint.fun) - 1e-6
