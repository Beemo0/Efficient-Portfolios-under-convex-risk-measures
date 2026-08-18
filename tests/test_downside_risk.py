import numpy as np
import pytest

from convexrisk.downside_risk import lower_partial_moment, semi_variance


def test_lpm_of_a_position_always_above_target_is_zero():
    pnl = np.full(1000, 5.0)
    assert lower_partial_moment(pnl, target=1.0, order=2.0) == pytest.approx(0.0)


def test_lpm_rejects_order_below_one():
    with pytest.raises(ValueError):
        lower_partial_moment(np.array([1.0, 2.0]), target=0.0, order=0.5)


def test_semi_variance_defaults_target_to_the_mean():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.05, 0.2, size=100000)
    sv_default = semi_variance(pnl)
    sv_explicit = semi_variance(pnl, target=float(np.mean(pnl)))
    assert sv_default == pytest.approx(sv_explicit)


def test_semi_variance_of_symmetric_distribution_is_half_variance():
    # For a symmetric distribution around its mean, E[((m-X)^+)^2] is
    # exactly half of Var(X) (each side of the symmetric distribution
    # contributes equally, and only one side is penalised).
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.0, 1.0, size=2_000_000)
    sv = semi_variance(pnl, target=0.0)
    assert sv == pytest.approx(0.5, rel=0.01)


def test_harlow_portfolio_recovers_target_return():
    from convexrisk.downside_risk import harlow_efficient_portfolio
    rng = np.random.default_rng(0)
    mu = np.array([0.08, 0.04])
    sigma = np.array([[0.05, 0.01], [0.01, 0.03]])
    returns = rng.multivariate_normal(mu, sigma, size=5000)
    probs = np.full(5000, 1.0 / 5000)
    target = 0.05
    result = harlow_efficient_portfolio(returns, probs, target_return=target, order=2.0,
                                         lpm_target=0.0)
    assert result["success"]
    assert result["expected_return"] == pytest.approx(target, abs=1e-4)


def test_harlow_portfolio_direction_matches_markowitz_for_symmetric_returns():
    # For a symmetric (Gaussian) distribution, minimising LPM_2 relative
    # to a fixed target T for a portfolio whose expected return is
    # constrained to equal that same T is, at the binding optimum,
    # exactly the semivariance of the portfolio around its own mean --
    # i.e. half its variance -- so it should recover the same portfolio
    # direction as minimising variance directly (Markowitz), by the same
    # symmetry mechanism underlying Theorem 3.1's elliptical-case
    # coincidence. (An earlier version of this test used mu=[0,0], which
    # made markowitz_portfolio ill-defined since mu'Sigma^{-1}mu=0.)
    from convexrisk.downside_risk import harlow_efficient_portfolio
    from convexrisk.markowitz import markowitz_portfolio

    rng = np.random.default_rng(3)
    mu = np.array([0.08, 0.04])
    sigma = np.array([[0.05, 0.01], [0.01, 0.03]])
    returns = rng.multivariate_normal(mu, sigma, size=20000)
    probs = np.full(20000, 1.0 / 20000)
    target = 0.03

    result = harlow_efficient_portfolio(returns, probs, target_return=target, order=2.0,
                                         lpm_target=target)
    assert result["success"]
    pi_markowitz = markowitz_portfolio(mu, sigma, target)
    relative_error = np.linalg.norm(result["pi"] - pi_markowitz) / np.linalg.norm(pi_markowitz)
    assert relative_error < 0.15
