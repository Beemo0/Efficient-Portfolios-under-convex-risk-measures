"""
Classical Markowitz mean-variance efficient portfolio (closed form),
used throughout as the comparison point for Theorem 3.1 (exact
coincidence under Gaussian returns) and its numerical validation, and
for the divergence experiment under skewed/fat-tailed returns.
"""
from __future__ import annotations

import numpy as np


def markowitz_portfolio(mu: np.ndarray, sigma: np.ndarray, target_return: float) -> np.ndarray:
    """Closed-form minimiser of pi' Sigma pi subject to pi' mu = target_return,
    over pi in R^d (no budget/sum constraint, exactly the setting of
    Chapter 3: pi is the vector of excess-return exposures, the residual
    wealth sitting in the riskless asset).

    pi* = (target_return / (mu' Sigma^{-1} mu)) * Sigma^{-1} mu
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_inv_mu = np.linalg.solve(sigma, mu)
    denom = mu @ sigma_inv_mu
    if denom <= 0:
        raise ValueError("mu' Sigma^{-1} mu must be strictly positive (Sigma must be SPD)")
    return (target_return / denom) * sigma_inv_mu


def markowitz_frontier_std(mu: np.ndarray, sigma: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Standard deviation of the excess position x0 * pi'.(R - r_f 1) on
    the Markowitz frontier, for each target in ``targets`` (x0=1)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    stds = np.empty_like(np.asarray(targets, dtype=float))
    for i, target in enumerate(targets):
        pi = markowitz_portfolio(mu, sigma, target)
        stds[i] = np.sqrt(pi @ sigma @ pi)
    return stds
