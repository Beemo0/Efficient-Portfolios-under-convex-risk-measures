import numpy as np
import pytest

from convexrisk.risk_measures import (
    empirical_var,
    empirical_avar,
    avar_variational,
    kappa,
    gaussian_avar,
    gaussian_var,
    sup_of_coherent_measures,
)


def test_avar_geq_var_on_a_discrete_sample():
    # Theory: AVaR_level >= VaR_level always (Section 2.5, since VaR_s is
    # non-increasing in s and AVaR averages VaR_s over s <= level).
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.05, 0.2, size=5000)
    for level in (0.01, 0.05, 0.1, 0.25):
        v = empirical_var(pnl, level)
        a = empirical_avar(pnl, level)
        assert a >= v - 1e-9


def test_avar_of_a_constant_is_minus_the_constant():
    pnl = np.full(1000, 3.7)
    assert empirical_avar(pnl, 0.05) == pytest.approx(-3.7)
    assert empirical_var(pnl, 0.05) == pytest.approx(-3.7)


def test_gaussian_closed_form_matches_empirical_avar_at_large_n():
    rng = np.random.default_rng(1)
    mean, std, level = 0.08, 0.22, 0.05
    pnl = rng.normal(mean, std, size=2_000_000)
    closed_form = gaussian_avar(mean, std, level)
    empirical = empirical_avar(pnl, level)
    # Large-sample Monte Carlo error at this tail level: allow 2% relative.
    assert empirical == pytest.approx(closed_form, rel=0.02)


def test_gaussian_closed_form_matches_gaussian_var_ordering():
    mean, std = 0.1, 0.3
    for level in (0.01, 0.05, 0.5):
        v = gaussian_var(mean, std, level)
        a = gaussian_avar(mean, std, level)
        assert a >= v


def test_variational_formula_matches_empirical_avar():
    # Independent numerical route (Proposition 3.2 / Theorem 2.2):
    # min_c [c + (1/level) E[(-X-c)^+]] must equal the direct
    # order-statistic estimator, and the argmin must equal the empirical VaR.
    rng = np.random.default_rng(2)
    pnl = rng.standard_t(df=4, size=20000) * 0.15 + 0.05
    level = 0.05
    direct = empirical_avar(pnl, level)
    via_variational, argmin_c = avar_variational(pnl, level)
    assert via_variational == pytest.approx(direct, rel=1e-6, abs=1e-8)
    var_direct = empirical_var(pnl, level)
    assert argmin_c == pytest.approx(var_direct, rel=1e-3, abs=1e-3)


def test_kappa_decreases_to_zero_as_level_to_one():
    levels = np.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.99])
    values = np.array([kappa(l) for l in levels])
    assert np.all(np.diff(values) < 0)  # strictly decreasing
    assert kappa(0.999) < 0.05


def test_kappa_rejects_level_of_one_would_be_risk_neutral_limit():
    # AVaR_1(X) = -E[X] in the limit; kappa(level) -> 0, so
    # gaussian_avar(mean, std, level) -> -mean as level -> 1.
    mean, std = 0.1, 0.4
    val = gaussian_avar(mean, std, 0.9999)
    assert val == pytest.approx(-mean, abs=1e-2)


def test_invalid_level_raises():
    pnl = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        empirical_var(pnl, 0.0)
    with pytest.raises(ValueError):
        empirical_avar(pnl, 1.0)
    with pytest.raises(ValueError):
        kappa(1.5)


def test_sup_of_coherent_measures_is_the_max():
    rhos = np.array([1.0, 3.5, 2.2])
    assert sup_of_coherent_measures(rhos) == 3.5
