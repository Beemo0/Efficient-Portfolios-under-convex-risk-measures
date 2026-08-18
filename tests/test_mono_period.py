import numpy as np
import pytest

from convexrisk.mono_period import avar_efficient_portfolio, avar_efficient_frontier
from convexrisk.markowitz import markowitz_portfolio
from convexrisk.risk_measures import gaussian_avar


def _simulate_scenarios(mu, sigma, n_scenarios, rng):
    returns = rng.multivariate_normal(mu, sigma, size=n_scenarios)
    probs = np.full(n_scenarios, 1.0 / n_scenarios)
    return returns, probs


def test_lp_recovers_target_return_exactly():
    rng = np.random.default_rng(0)
    mu = np.array([0.08, 0.05])
    sigma = np.array([[0.04, 0.01], [0.01, 0.02]])
    returns, probs = _simulate_scenarios(mu, sigma, 4000, rng)
    target = 0.03
    result = avar_efficient_portfolio(returns, probs, level=0.05, target_return=target)
    assert result["success"]
    assert result["expected_return"] == pytest.approx(target, abs=1e-6)


def test_gaussian_case_avar_lp_coincides_with_markowitz():
    # Theorem 3.1: under Gaussian returns, the AVaR-efficient portfolio
    # coincides exactly with the Markowitz portfolio, for every level.
    # Here we solve the LP on a large simulated Gaussian scenario set and
    # check it recovers the closed-form Markowitz weights up to Monte
    # Carlo / LP-discretisation error.
    rng = np.random.default_rng(42)
    mu = np.array([0.10, 0.04, 0.06])
    sigma = np.array([
        [0.06, 0.01, 0.02],
        [0.01, 0.03, 0.00],
        [0.02, 0.00, 0.05],
    ])
    target = 0.05
    returns, probs = _simulate_scenarios(mu, sigma, 20000, rng)

    for level in (0.01, 0.05, 0.10):
        result = avar_efficient_portfolio(returns, probs, level=level, target_return=target)
        assert result["success"]
        pi_markowitz = markowitz_portfolio(mu, sigma, target)
        # Direction and magnitude should coincide up to Monte Carlo noise.
        relative_error = np.linalg.norm(result["pi"] - pi_markowitz) / np.linalg.norm(pi_markowitz)
        assert relative_error < 0.20, f"level={level}: relative error {relative_error:.3f}"


def test_gaussian_case_lp_avar_matches_closed_form_value():
    rng = np.random.default_rng(7)
    mu = np.array([0.09, 0.05])
    sigma = np.array([[0.05, 0.015], [0.015, 0.03]])
    target = 0.04
    returns, probs = _simulate_scenarios(mu, sigma, 30000, rng)
    level = 0.05

    result = avar_efficient_portfolio(returns, probs, level=level, target_return=target)
    assert result["success"]

    pi = result["pi"]
    portfolio_mean = mu @ pi
    portfolio_std = np.sqrt(pi @ sigma @ pi)
    closed_form_avar = gaussian_avar(portfolio_mean, portfolio_std, level)

    assert result["avar"] == pytest.approx(closed_form_avar, rel=0.10)


def test_frontier_is_non_decreasing_in_risk_as_target_increases():
    rng = np.random.default_rng(3)
    mu = np.array([0.07, 0.03])
    sigma = np.array([[0.05, 0.0], [0.0, 0.02]])
    returns, probs = _simulate_scenarios(mu, sigma, 5000, rng)
    targets = np.linspace(0.0, 0.06, 6)
    results = avar_efficient_frontier(returns, probs, level=0.05, targets=targets)
    avars = np.array([r["avar"] for r in results if r["success"]])
    # A higher target return should require weakly more risk on the
    # efficient frontier (the frontier's fundamental risk-return trade-off).
    assert np.all(np.diff(avars) >= -1e-6)


def test_box_constraint_limits_exposure():
    rng = np.random.default_rng(5)
    mu = np.array([0.5, 0.5])  # deliberately large excess returns
    sigma = np.array([[0.10, 0.0], [0.0, 0.10]])
    returns, probs = _simulate_scenarios(mu, sigma, 3000, rng)
    result = avar_efficient_portfolio(
        returns, probs, level=0.05, target_return=0.05, pi_bounds=(0.0, 1.0)
    )
    assert result["success"]
    assert np.all(result["pi"] >= -1e-8)
    assert np.all(result["pi"] <= 1.0 + 1e-8)
