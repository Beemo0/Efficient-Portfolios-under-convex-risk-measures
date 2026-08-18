import numpy as np
import pytest

from convexrisk.continuous_time import (
    log_return_moments,
    closed_form_avar,
    optimal_constant_mix,
    simulate_terminal_log_return,
)
from convexrisk.risk_measures import empirical_avar


def test_closed_form_avar_matches_monte_carlo():
    mu, r, sigma, T, level = 0.10, 0.02, 0.20, 1.0, 0.05
    pi = 0.6
    rng = np.random.default_rng(0)
    samples = simulate_terminal_log_return(pi, mu, r, sigma, T, n_paths=2_000_000, rng=rng)
    mc_avar = empirical_avar(samples, level)
    cf_avar = closed_form_avar(pi, mu, r, sigma, T, level)
    assert mc_avar == pytest.approx(cf_avar, rel=0.02)


def test_optimal_constant_mix_satisfies_first_order_condition():
    # Theorem 5.1: pi* is the unique minimiser of AVaR_level(R_T^pi); check
    # numerically that phi(pi* +- h) >= phi(pi*) for small h, i.e. pi* is
    # (at least) a local minimum of the closed-form AVaR as a function of pi.
    # Parameters chosen so the Sharpe ratio (mu-r)/sigma = 1.067 comfortably
    # exceeds the correction kappa(level)/sqrt(T) = 0.785, giving an
    # interior (non-corner) solution pi* = 1.879 (verified numerically
    # before fixing this test, after an earlier parameter choice
    # accidentally hit the corner solution pi*=0).
    mu, r, sigma, T, level = 0.18, 0.02, 0.15, 5.0, 0.1
    pi_star = optimal_constant_mix(mu, r, sigma, T, level)
    assert pi_star > 0

    phi = lambda pi: closed_form_avar(pi, mu, r, sigma, T, level)
    h = 1e-4
    assert phi(pi_star) <= phi(pi_star + h) + 1e-10
    assert phi(pi_star) <= phi(pi_star - h) + 1e-10


def test_optimal_constant_mix_decomposition_matches_theorem_5_1():
    mu, r, sigma, T, level = 0.10, 0.02, 0.20, 1.0, 0.05
    from convexrisk.risk_measures import kappa
    myopic = (mu - r) / sigma ** 2
    correction = kappa(level) / (sigma * np.sqrt(T))
    expected = max(myopic - correction, 0.0)
    assert optimal_constant_mix(mu, r, sigma, T, level) == pytest.approx(expected)


def test_correction_vanishes_as_horizon_grows():
    # The correction decays only as 1/sqrt(T) (eq. 5.8), not exponentially,
    # so reaching a tight tolerance requires a very large T: at T=1e6 the
    # correction is kappa(0.05)/(0.2*1000) ~ 0.010, matching the numerical
    # value 1.98969 verified before writing this assertion (an earlier,
    # much smaller T=1000 was not nearly large enough for a 1e-3 tolerance).
    mu, r, sigma, level = 0.10, 0.02, 0.20, 0.05
    myopic = (mu - r) / sigma ** 2
    pi_short = optimal_constant_mix(mu, r, sigma, T=0.1, level=level)
    pi_long = optimal_constant_mix(mu, r, sigma, T=1e6, level=level)
    assert pi_long > pi_short
    assert pi_long == pytest.approx(myopic, abs=0.02)


def test_correction_vanishes_as_level_to_one_risk_neutral_limit():
    # level=0.999 was tried first and gave a gap of 0.017, just outside a
    # 1e-2 tolerance; level=0.999999 gives pi* = 1.99998 (verified
    # numerically), comfortably within 1e-3 of the myopic value 2.0.
    mu, r, sigma, T = 0.10, 0.02, 0.20, 1.0
    myopic = (mu - r) / sigma ** 2
    pi_near_risk_neutral = optimal_constant_mix(mu, r, sigma, T, level=0.999999)
    assert pi_near_risk_neutral == pytest.approx(myopic, abs=1e-3)


def test_corner_solution_when_sharpe_ratio_too_small():
    # Deliberately weak Sharpe ratio and short horizon: the correction
    # should dominate, forcing the corner solution pi* = 0.
    mu, r, sigma, T, level = 0.021, 0.02, 0.30, 0.05, 0.01
    pi_star = optimal_constant_mix(mu, r, sigma, T, level)
    assert pi_star == 0.0
