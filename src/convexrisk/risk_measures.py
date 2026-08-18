"""
Core convex risk measure functionals: Value-at-Risk, Average Value-at-Risk,
and the Gaussian closed form, matching the conventions of Chapter 2
(``Efficient Portfolios under Convex Risk Measures'').

Sign convention (fixed throughout the whole package): a financial
*position* ``pnl`` is a profit-and-loss random variable (higher is
better); a risk measure ``rho(pnl)`` returns the capital requirement
(higher = riskier). In particular ``rho`` of a constant ``a`` equals
``-a``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def empirical_var(pnl: np.ndarray, level: float) -> float:
    """Empirical Value-at-Risk at confidence level ``level`` in (0, 1),
    following Definition 2.2 (eq. 2.9): VaR_level(X) = inf{m : P(X+m<0) <=
    level}. Delegates to :func:`discrete_var` on the empirical
    distribution (n equally-weighted atoms), which is the exact discrete
    quantile rather than a plain order-statistic shortcut: the two
    disagree whenever n * level is not an integer, and an earlier version
    of this function used the (slightly inconsistent) order-statistic
    shortcut directly -- see the regression test
    ``test_discrete_avar_matches_empirical_avar_under_uniform_weights``
    in tests/test_discrete_avar_exact.py, which caught the mismatch on a
    sample of size 997 (deliberately not a "nice" multiple of any level
    tested).
    """
    pnl = np.asarray(pnl, dtype=float)
    n = pnl.size
    probs = np.full(n, 1.0 / n)
    return discrete_var(pnl, probs, level)


def empirical_avar(pnl: np.ndarray, level: float) -> float:
    """Empirical Average Value-at-Risk at level ``level``, following
    Definition 2.7 (eq. 2.13). Delegates to :func:`discrete_avar` on the
    empirical distribution (n equally-weighted atoms), the exact
    discretisation of the defining integral rather than the coarser
    "average of the ceil(n*level) worst observations" shortcut, which
    disagrees with it whenever n * level is not an integer.
    """
    pnl = np.asarray(pnl, dtype=float)
    n = pnl.size
    probs = np.full(n, 1.0 / n)
    return discrete_avar(pnl, probs, level)


def avar_variational(pnl: np.ndarray, level: float) -> tuple[float, float]:
    """Compute AVaR via the variational formula (Theorem 2.2, eq. 2.15):
        AVaR_level(X) = min_c [ c + (1/level) * E[(-X - c)^+] ].
    Returns (avar_value, argmin_c). This is an *independent* numerical
    route to the same quantity as :func:`empirical_avar`, used as a
    cross-check (the two share no code path beyond the raw sample).
    """
    pnl = np.asarray(pnl, dtype=float)

    def objective(c: float) -> float:
        return c + (1.0 / level) * np.mean(np.maximum(-pnl - c, 0.0))

    # F(., c) is convex and piecewise linear in c with kinks at -pnl_i;
    # bracket generously around the empirical VaR to bound the search.
    lo, hi = -np.max(pnl) - 1.0, -np.min(pnl) + 1.0
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10})
    return float(result.fun), float(result.x)


def kappa(level: float) -> float:
    """The Gaussian AVaR constant kappa(level) = phi(Phi^{-1}(level)) /
    level appearing in Lemma 2.1 / eq. 3.10. Strictly positive for level
    in (0, 1), decreasing, with kappa(level) -> 0 as level -> 1.
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")
    z = norm.ppf(level)
    return float(norm.pdf(z) / level)


def gaussian_avar(mean: float, std: float, level: float) -> float:
    """Closed-form AVaR of a Gaussian position X ~ N(mean, std^2),
    following Lemma 2.1 (eq. 2.14 / 3.10):
        AVaR_level(X) = -mean + std * kappa(level).
    """
    if std < 0:
        raise ValueError("std must be non-negative")
    return -mean + std * kappa(level)


def gaussian_var(mean: float, std: float, level: float) -> float:
    """Closed-form VaR of a Gaussian position X ~ N(mean, std^2):
        VaR_level(X) = -mean - std * Phi^{-1}(level).
    """
    if std < 0:
        raise ValueError("std must be non-negative")
    return -mean - std * norm.ppf(level)


def discrete_var(values: np.ndarray, probs: np.ndarray, level: float) -> float:
    """Exact VaR_level of a discrete position with atoms ``values`` and
    (not necessarily uniform) probabilities ``probs``, following
    Definition 2.2 directly rather than an equal-weight sample estimator.
    """
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")
    order = np.argsort(values)
    values_sorted = values[order]
    probs_sorted = probs[order]
    cum = np.cumsum(probs_sorted)
    k = int(np.searchsorted(cum, level, side="left"))
    k = min(k, len(values_sorted) - 1)
    return -float(values_sorted[k])


def discrete_avar(values: np.ndarray, probs: np.ndarray, level: float) -> float:
    """Exact AVaR_level of a discrete position with atoms ``values`` and
    (not necessarily uniform) probabilities ``probs``:

        AVaR_level(X) = -(1/level) * [ sum of full atoms strictly below
                          the level-quantile, weighted by probability,
                          plus a partial weight on the boundary atom ].

    This is the direct discretisation of AVaR_level(X) = (1/level)
    int_0^level VaR_s(X) ds and coincides with :func:`empirical_avar`
    when ``probs`` is uniform (tested in test_risk_measures.py).
    """
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")
    order = np.argsort(values)
    values_sorted = values[order]
    probs_sorted = probs[order]
    cum = np.cumsum(probs_sorted)
    k = int(np.searchsorted(cum, level, side="left"))
    k = min(k, len(values_sorted) - 1)
    mass_before = cum[k - 1] if k > 0 else 0.0
    partial_weight = level - mass_before
    total = probs_sorted[:k] @ values_sorted[:k] + partial_weight * values_sorted[k]
    return -float(total / level)


def sup_of_coherent_measures(rhos: np.ndarray) -> float:
    """Given an array of values (rho_i(X))_i of several coherent risk
    measures evaluated at the *same* position X, return their supremum.
    This is the elementary building block behind
    Proposition 6.1 (sup of coherent risk measures is coherent),
    used by :mod:`convexrisk.robust`.
    """
    return float(np.max(rhos))
