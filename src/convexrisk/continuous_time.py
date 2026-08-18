"""
Continuous-time constant-mix strategies in the Black-Scholes model
(Chapter 5): the closed-form AVaR-optimal proportion (Theorem 5.1) and a
Monte Carlo simulator used to validate it independently.
"""
from __future__ import annotations

import numpy as np

from .risk_measures import gaussian_avar, kappa


def log_return_moments(pi: float, mu: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    """(m_pi, sigma_pi) of R_T^pi ~ N(m_pi, sigma_pi^2), eq. 5.6."""
    m_pi = (r + pi * (mu - r) - 0.5 * pi ** 2 * sigma ** 2) * T
    sigma_pi = abs(pi) * sigma * np.sqrt(T)
    return m_pi, sigma_pi


def closed_form_avar(pi: float, mu: float, r: float, sigma: float, T: float, level: float) -> float:
    """AVaR_level(R_T^pi) via Proposition 5.1 (eq. 5.7)."""
    m_pi, sigma_pi = log_return_moments(pi, mu, r, sigma, T)
    return gaussian_avar(m_pi, sigma_pi, level)


def optimal_constant_mix(mu: float, r: float, sigma: float, T: float, level: float) -> float:
    """pi*(level, T) of Theorem 5.1 (eq. 5.8), clipped at 0 if the
    no-risky-exposure corner solution is optimal (Sharpe ratio too small
    relative to the risk-aversion correction)."""
    myopic = (mu - r) / sigma ** 2
    correction = kappa(level) / (sigma * np.sqrt(T))
    return max(myopic - correction, 0.0)


def simulate_terminal_log_return(
    pi: float, mu: float, r: float, sigma: float, T: float,
    n_paths: int, rng: np.random.Generator,
) -> np.ndarray:
    """Simulate R_T^pi = ln(X_T^pi / x0) directly from the exact
    closed-form solution of the constant-mix wealth SDE (eq. 5.5),
    i.e. exact simulation (no discretisation error): each path is
    m_pi + pi*sigma*W_T with W_T ~ N(0, T), pathwise identical to the
    formula the closed form is derived from, so any discrepancy between
    the simulated AVaR and the closed form is attributable purely to
    Monte Carlo sampling noise, not to a discretisation scheme.
    """
    m_pi, _ = log_return_moments(pi, mu, r, sigma, T)
    W_T = rng.normal(0.0, np.sqrt(T), size=n_paths)
    return m_pi + pi * sigma * W_T
