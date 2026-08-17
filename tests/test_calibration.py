"""Tests for the calibration diagnostics.

These tests are all of the form "feed it a distribution whose calibration we
know, check it says so". A diagnostic that reports good calibration for a
badly calibrated model is the same failure mode as a leakage test that cannot
detect leaks.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfl_usage_props.model.calibration import (
    assess,
    coverage,
    log_score,
    pit_deviation,
    randomised_pit,
)


@pytest.fixture
def rng():
    return np.random.default_rng(11)


def poisson_samples(rng, mu, n=1500, draws=3000):
    """Draws from a correctly specified predictive, plus matching outcomes."""
    means = rng.uniform(mu * 0.7, mu * 1.3, size=n)
    samples = rng.poisson(means[:, None], size=(n, draws))
    actual = rng.poisson(means)
    return samples, actual


# ---------------------------------------------------------------------- PIT


def test_pit_is_uniform_for_a_correct_model(rng):
    samples, actual = poisson_samples(rng, 60)
    below = np.mean(samples < actual[:, None], axis=1)
    at_or_below = np.mean(samples <= actual[:, None], axis=1)
    assert pit_deviation(randomised_pit(at_or_below, below)) < 0.15


def test_pit_detects_an_overconfident_model(rng):
    """Too-narrow intervals push outcomes into the tails: a U-shaped PIT."""
    samples, actual = poisson_samples(rng, 60)
    narrow = (samples - samples.mean(axis=1, keepdims=True)) * 0.3 + samples.mean(
        axis=1, keepdims=True
    )
    below = np.mean(narrow < actual[:, None], axis=1)
    at_or_below = np.mean(narrow <= actual[:, None], axis=1)
    assert pit_deviation(randomised_pit(at_or_below, below)) > 0.4


def test_pit_detects_a_biased_model(rng):
    samples, actual = poisson_samples(rng, 60)
    below = np.mean(samples + 15 < actual[:, None], axis=1)
    at_or_below = np.mean(samples + 15 <= actual[:, None], axis=1)
    assert pit_deviation(randomised_pit(at_or_below, below)) > 0.4


def test_pit_deviation_is_zero_for_a_perfectly_flat_histogram():
    assert pit_deviation(np.linspace(0, 1, 10_000)) < 0.01


# ----------------------------------------------------------------- coverage


def test_coverage_matches_the_nominal_level(rng):
    samples, actual = poisson_samples(rng, 60)
    assert coverage(samples, actual, 0.80) == pytest.approx(0.80, abs=0.04)
    assert coverage(samples, actual, 0.95) == pytest.approx(0.95, abs=0.03)


def test_discrete_intervals_over_cover_at_the_50_percent_level(rng):
    """Not a defect -- a property of counts. A central interval runs between
    two integers and includes both, so it cannot carve out exactly 50%. The
    tolerances in `well_calibrated` exist to absorb this, and this test pins
    the size of the effect so those tolerances stay justified."""
    samples, actual = poisson_samples(rng, 60)
    observed = coverage(samples, actual, 0.50)
    assert observed > 0.50
    assert observed < 0.50 + 0.09


def test_coverage_falls_when_intervals_are_too_narrow(rng):
    samples, actual = poisson_samples(rng, 60)
    narrow = (samples - samples.mean(axis=1, keepdims=True)) * 0.25 + samples.mean(
        axis=1, keepdims=True
    )
    assert coverage(narrow, actual, 0.80) < 0.5


# ---------------------------------------------------------------- log score


def test_log_score_prefers_the_correct_model(rng):
    """Proper scoring: a model cannot win by widening until coverage looks
    right. This is what makes coverage and log score complementary."""
    samples, actual = poisson_samples(rng, 20)
    wide = samples + rng.integers(-12, 13, size=samples.shape)
    assert log_score(samples, actual).mean() < log_score(wide, actual).mean()


def test_log_score_is_floored_not_infinite(rng):
    """An outcome with no sampled mass must not send the mean to infinity."""
    samples = np.zeros((3, 100), dtype=int)
    scores = log_score(samples, np.array([50, 50, 50]))
    assert np.all(np.isfinite(scores))


# ------------------------------------------------------------------ assess


def test_assess_reports_a_correct_model_as_well_calibrated(rng):
    samples, actual = poisson_samples(rng, 60)
    report = assess(samples, actual)
    assert report.well_calibrated
    assert report.n == len(actual)


def test_assess_rejects_an_overconfident_model(rng):
    samples, actual = poisson_samples(rng, 60)
    narrow = (samples - samples.mean(axis=1, keepdims=True)) * 0.25 + samples.mean(
        axis=1, keepdims=True
    )
    assert not assess(narrow, actual).well_calibrated


def test_assess_rejects_a_biased_model(rng):
    """Accurate spread, wrong centre. Coverage catches it even though the
    distribution's shape is fine."""
    samples, actual = poisson_samples(rng, 60)
    assert not assess(samples + 20, actual).well_calibrated


def test_summary_is_readable(rng):
    samples, actual = poisson_samples(rng, 60)
    summary = assess(samples, actual).summary()
    for token in ("PIT", "cover", "logscore", "MAE"):
        assert token in summary
