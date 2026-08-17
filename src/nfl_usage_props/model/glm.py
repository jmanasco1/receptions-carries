"""Two overdispersed GLMs, fitted by maximum likelihood.

Both layers of the team volume model need a distribution, not a point estimate,
and both need a distribution wider than the textbook one:

* **Negative Binomial** for team plays. Poisson would assert var = mean. On
  2016-2025 regular season team-games the mean is 62.2 with a standard
  deviation of 8.36, against the 7.88 Poisson would require -- so the
  overdispersion is real but mild, and the fitted phi lands near 700 rather
  than the double digits the term usually suggests. Mild is not zero:
  understating it understates every player's reception variance downstream,
  and the product is a half-point probability, so the tails are the deliverable.
* **Beta-Binomial** for the pass/rush split. Binomial would assert that every
  dropback within a game is an independent coin flip at the same rate. They are
  not independent: a team that falls behind throws more for the rest of the
  afternoon. The extra-binomial variation is the game script, and it is real.

Both are fitted by **profiling out the dispersion**: an outer 1-D search over
log phi, with the coefficients refitted at each candidate. Jointly optimising
everything at once is the obvious approach and it silently fails here. The
likelihood is very flat in log phi — team play counts are only mildly
overdispersed — so a finite-difference gradient in that direction is mostly
rounding error, and L-BFGS-B stops early while reporting success. On this data
the joint fit returned phi = 19,000 (effectively Poisson) when the true MLE is
near 800, understating the predictive variance by about 12% and reporting
`converged=True` the whole time.

Features are standardised before fitting for the same reason. Raw inputs here
span `is_home` at 0/1 and `total_line` near 45, which makes the Hessian badly
conditioned and the coefficients unreadable. The scaling is stored and applied
at prediction, so callers still pass raw values.

Deliberately small: these are two specific likelihoods, not a modelling
framework, and a hand-rolled framework is how you end up with a subtly wrong
link function nobody notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, expit, gammaln

# The optimiser explores; the likelihood must not return NaN when it does.
_MIN_MU = 1e-6
_MAX_LOG_MU = 12.0
_MIN_PHI = 1e-4
_MAX_PHI = 1e6


@dataclass
class FitResult:
    """Coefficients plus enough diagnostics to tell a fit from a shrug."""

    coefficients: np.ndarray
    dispersion: float
    log_likelihood: float
    n_observations: int
    n_parameters: int
    converged: bool
    message: str = ""
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def aic(self) -> float:
        return 2 * self.n_parameters - 2 * self.log_likelihood

    def describe(self) -> list[tuple[str, float]]:
        names = self.feature_names or tuple(f"x{i}" for i in range(len(self.coefficients)))
        return list(zip(names, self.coefficients.tolist(), strict=True))


def _design(x: np.ndarray) -> np.ndarray:
    """Prepend an intercept column."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack([np.ones(len(x)), x])


@dataclass
class Standardizer:
    """Centre and scale, fitted once on training data and reused at predict."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        scale = x.std(axis=0)
        # A constant column carries no information; scaling it by zero would
        # produce NaN and take the whole fit with it.
        scale[scale < 1e-12] = 1.0
        return cls(mean=x.mean(axis=0), scale=scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return (x - self.mean) / self.scale


def _profile_fit(
    objective,
    design: np.ndarray,
    start_beta: np.ndarray,
    log_phi_bounds: tuple[float, float],
) -> tuple[np.ndarray, float, float, bool, str]:
    """Optimise coefficients and dispersion by profiling out log phi.

    `objective(beta, phi)` returns the negative log-likelihood. Returns
    (beta, phi, nll, converged, message).
    """
    cache: dict[float, tuple[np.ndarray, float]] = {}

    def inner(log_phi: float) -> float:
        phi = float(np.exp(log_phi))
        seed = cache[max(cache)][0] if cache else start_beta
        fit = minimize(
            lambda beta: objective(beta, phi),
            seed,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        cache[log_phi] = (fit.x, float(fit.fun))
        return float(fit.fun)

    outer = minimize_scalar(inner, bounds=log_phi_bounds, method="bounded", options={"xatol": 1e-3})
    beta, nll = cache[outer.x]
    phi = float(np.clip(np.exp(outer.x), _MIN_PHI, _MAX_PHI))

    # A dispersion pinned to a bound is not an estimate, it is the optimiser
    # running out of room. Say so rather than reporting it as a fit.
    at_bound = abs(outer.x - log_phi_bounds[0]) < 1e-2 or abs(outer.x - log_phi_bounds[1]) < 1e-2
    message = "dispersion hit the search bound" if at_bound else "ok"
    return beta, phi, nll, bool(outer.success) and not at_bound, message


class NegativeBinomialGLM:
    """Log-link Negative Binomial regression (NB2: var = mu + mu^2/phi).

    `phi` is the dispersion in the "size" parameterisation: large phi tends to
    Poisson, small phi means heavy overdispersion.
    """

    def __init__(self, feature_names: tuple[str, ...] = ()):
        self.feature_names = feature_names
        self.result: FitResult | None = None
        self.standardizer: Standardizer | None = None

    # ------------------------------------------------------------ likelihood

    @staticmethod
    def _nll(beta: np.ndarray, phi: float, design: np.ndarray, y: np.ndarray) -> float:
        mu = np.exp(np.clip(design @ beta, -_MAX_LOG_MU, _MAX_LOG_MU))
        mu = np.maximum(mu, _MIN_MU)
        ll = (
            gammaln(y + phi)
            - gammaln(phi)
            - gammaln(y + 1.0)
            + phi * np.log(phi / (phi + mu))
            + y * np.log(mu / (phi + mu))
        )
        total = float(np.sum(ll))
        return np.inf if not np.isfinite(total) else -total

    @staticmethod
    def _negative_log_likelihood(params: np.ndarray, design: np.ndarray, y: np.ndarray) -> float:
        """Joint form, kept for diagnostics and likelihood profiling in tests."""
        beta, log_phi = params[:-1], params[-1]
        phi = float(np.clip(np.exp(log_phi), _MIN_PHI, _MAX_PHI))
        mu = np.exp(np.clip(design @ beta, -_MAX_LOG_MU, _MAX_LOG_MU))
        mu = np.maximum(mu, _MIN_MU)

        ll = (
            gammaln(y + phi)
            - gammaln(phi)
            - gammaln(y + 1.0)
            + phi * np.log(phi / (phi + mu))
            + y * np.log(mu / (phi + mu))
        )
        total = float(np.sum(ll))
        return np.inf if not np.isfinite(total) else -total

    # ------------------------------------------------------------------ fit

    def fit(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        self.standardizer = Standardizer.fit(x)
        design = _design(self.standardizer.apply(x))
        y = np.asarray(y, dtype=float)

        # Start from the intercept-only solution: the mean is always a valid
        # answer, and starting at zero makes the exponential link overflow.
        start = np.zeros(design.shape[1])
        start[0] = np.log(max(y.mean(), _MIN_MU))

        beta, phi, nll, converged, message = _profile_fit(
            lambda b, p: self._nll(b, p, design, y),
            design,
            start,
            log_phi_bounds=(np.log(0.05), np.log(5e4)),
        )
        self.result = FitResult(
            coefficients=beta,
            dispersion=phi,
            log_likelihood=-nll,
            n_observations=len(y),
            n_parameters=len(beta) + 1,
            converged=converged,
            message=message,
            feature_names=("intercept", *self.feature_names),
        )
        return self.result

    # -------------------------------------------------------------- predict

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        self._require_fit()
        design = _design(self.standardizer.apply(x))
        return np.exp(np.clip(design @ self.result.coefficients, -_MAX_LOG_MU, _MAX_LOG_MU))

    def predict_variance(self, x: np.ndarray) -> np.ndarray:
        mu = self.predict_mean(x)
        return mu + mu**2 / self.result.dispersion

    def sample(self, x: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draws of shape (len(x), draws), via the gamma-Poisson mixture."""
        mu = self.predict_mean(x)[:, None]
        phi = self.result.dispersion
        rate = rng.gamma(shape=phi, scale=mu / phi, size=(len(mu), draws))
        return rng.poisson(rate)

    def cdf_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """P(Y <= y) under the fitted model, for calibration checks."""
        from scipy.stats import nbinom

        mu = self.predict_mean(x)
        phi = self.result.dispersion
        return nbinom.cdf(y, phi, phi / (phi + mu))

    def _require_fit(self) -> None:
        if self.result is None:
            raise RuntimeError("model is not fitted; call fit() first")


class BetaBinomialGLM:
    """Logit-link Beta-Binomial regression on (successes, trials).

    `phi` is the concentration: large phi tends to Binomial, small phi means
    the per-game rate itself varies a lot around its predicted mean.
    """

    def __init__(self, feature_names: tuple[str, ...] = ()):
        self.feature_names = feature_names
        self.result: FitResult | None = None
        self.standardizer: Standardizer | None = None

    @staticmethod
    def _nll(
        beta: np.ndarray, phi: float, design: np.ndarray, successes: np.ndarray, trials: np.ndarray
    ) -> float:
        p = np.clip(expit(design @ beta), 1e-9, 1 - 1e-9)
        a, b = p * phi, (1.0 - p) * phi
        ll = (
            betaln(successes + a, trials - successes + b)
            - betaln(a, b)
            + gammaln(trials + 1.0)
            - gammaln(successes + 1.0)
            - gammaln(trials - successes + 1.0)
        )
        total = float(np.sum(ll))
        return np.inf if not np.isfinite(total) else -total

    def fit(self, x: np.ndarray, successes: np.ndarray, trials: np.ndarray) -> FitResult:
        successes = np.asarray(successes, dtype=float)
        trials = np.asarray(trials, dtype=float)
        if np.any(successes > trials):
            raise ValueError("successes cannot exceed trials")

        self.standardizer = Standardizer.fit(x)
        design = _design(self.standardizer.apply(x))

        pooled = float(np.clip(successes.sum() / max(trials.sum(), 1.0), 1e-6, 1 - 1e-6))
        start = np.zeros(design.shape[1])
        start[0] = np.log(pooled / (1 - pooled))

        beta, phi, nll, converged, message = _profile_fit(
            lambda b, p: self._nll(b, p, design, successes, trials),
            design,
            start,
            log_phi_bounds=(np.log(0.5), np.log(5e4)),
        )
        self.result = FitResult(
            coefficients=beta,
            dispersion=phi,
            log_likelihood=-nll,
            n_observations=len(successes),
            n_parameters=len(beta) + 1,
            converged=converged,
            message=message,
            feature_names=("intercept", *self.feature_names),
        )
        return self.result

    def predict_rate(self, x: np.ndarray) -> np.ndarray:
        self._require_fit()
        return expit(_design(self.standardizer.apply(x)) @ self.result.coefficients)

    def sample_rate(self, x: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draws of the RATE, shape (len(x), draws).

        The rate is what Layer 2 hands to Layer 3 -- sampling counts here and
        counts again downstream would apply binomial noise twice.
        """
        p = self.predict_rate(x)[:, None]
        phi = self.result.dispersion
        return rng.beta(p * phi, (1.0 - p) * phi, size=(len(p), draws))

    def _require_fit(self) -> None:
        if self.result is None:
            raise RuntimeError("model is not fitted; call fit() first")
