import numpy as np
import pytest

from convexrisk.risk_measures import discrete_avar, discrete_var, empirical_avar


def test_discrete_avar_matches_hand_derived_example_worst_dominant():
    # Hand-derived in preparing Chapter 4: {X=10 (p=0.5), X=-100 (p=0.5)},
    # AVaR_{0.5} = 100 exactly (the single worst atom dominates at the
    # matching threshold).
    values = np.array([10.0, -100.0])
    probs = np.array([0.5, 0.5])
    assert discrete_avar(values, probs, 0.5) == pytest.approx(100.0)


def test_discrete_avar_matches_hand_derived_example_both_negative():
    # Hand-derived: W = -100 (p=0.5) or -1 (p=0.5), AVaR_{0.5} = 100.
    values = np.array([-100.0, -1.0])
    probs = np.array([0.5, 0.5])
    assert discrete_avar(values, probs, 0.5) == pytest.approx(100.0)


def test_discrete_avar_matches_hand_derived_four_point_example():
    # Hand-derived unconditional distribution: -100 (0.25), 100 (0.25),
    # -1 (0.25), 1 (0.25); AVaR_{0.25} = 100 (single worst atom, prob
    # exactly matches the threshold).
    values = np.array([-100.0, 100.0, -1.0, 1.0])
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    assert discrete_avar(values, probs, 0.25) == pytest.approx(100.0)


def test_discrete_avar_of_constant_is_minus_constant():
    values = np.array([5.0, 5.0, 5.0])
    probs = np.array([0.3, 0.3, 0.4])
    assert discrete_avar(values, probs, 0.1) == pytest.approx(-5.0)
    assert discrete_var(values, probs, 0.1) == pytest.approx(-5.0)


def test_discrete_avar_matches_empirical_avar_under_uniform_weights():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=997)  # awkward size, not a multiple of anything
    probs = np.full(values.size, 1.0 / values.size)
    for level in (0.01, 0.05, 0.2):
        a1 = discrete_avar(values, probs, level)
        a2 = empirical_avar(values, level)
        assert a1 == pytest.approx(a2, rel=1e-9, abs=1e-9)


def test_discrete_avar_geq_discrete_var():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, size=50)
    probs = rng.dirichlet(np.ones(50))
    for level in (0.05, 0.3, 0.7):
        v = discrete_var(values, probs, level)
        a = discrete_avar(values, probs, level)
        assert a >= v - 1e-9
