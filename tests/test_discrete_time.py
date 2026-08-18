import numpy as np
import pytest

from convexrisk.discrete_time import (
    BinaryTree,
    terminal_distribution,
    naive_static_avar,
    recursive_composed_rho0,
    search_for_divergence,
)


def _two_period_example_from_chapter4_writeup():
    # The exact tree hand-worked while preparing Chapter 4: Up (prob 1/2)
    # leads to {10, -100} each conditional prob 1/2; Down (prob 1/2) is
    # deterministic 0. Both the naive static AVaR_{1/4} and the composed
    # rho_0 (one-step level 1/2 at both steps) were hand-computed to be
    # exactly 100.
    depth = 2
    cond_prob_up = [np.array([0.5]), np.array([0.5, 0.5])]
    incr_up = [np.array([0.0]), np.array([10.0, 0.0])]
    incr_down = [np.array([0.0]), np.array([-100.0, 0.0])]
    return BinaryTree(depth, cond_prob_up, incr_up, incr_down)


def test_terminal_distribution_matches_hand_worked_example():
    tree = _two_period_example_from_chapter4_writeup()
    payoffs, probs = terminal_distribution(tree)
    # Leaves in order: UU, UD, DU, DD.
    assert payoffs == pytest.approx([10.0, -100.0, 0.0, 0.0])
    assert probs == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_naive_static_avar_matches_hand_computation():
    tree = _two_period_example_from_chapter4_writeup()
    assert naive_static_avar(tree, level=0.25) == pytest.approx(100.0)


def test_recursive_composed_matches_hand_computation():
    tree = _two_period_example_from_chapter4_writeup()
    assert recursive_composed_rho0(tree, one_step_level=0.5) == pytest.approx(100.0)


def test_recursive_composition_is_deterministic_and_finite():
    rng = np.random.default_rng(0)
    from convexrisk.discrete_time import generate_random_tree
    tree = generate_random_tree(depth=4, rng=rng)
    value1 = recursive_composed_rho0(tree, one_step_level=0.3)
    value2 = recursive_composed_rho0(tree, one_step_level=0.3)
    assert np.isfinite(value1)
    assert value1 == value2  # purely deterministic given the tree


def test_search_for_divergence_runs_and_returns_a_valid_tree():
    rng = np.random.default_rng(123)
    best = search_for_divergence(depth=4, n_trials=200, rng=rng,
                                  one_step_level=0.3, static_level=0.1)
    assert best["relative_gap"] >= 0.0
    assert np.isfinite(best["naive"])
    assert np.isfinite(best["composed"])
    # A meaningful search over 200 random trees at depth 4 should be able
    # to find at least *some* non-trivial disagreement between the two
    # conventions -- this is the honest, code-verified counterpart to the
    # hand-constructions in Chapter 4 that kept coinciding.
    assert best["relative_gap"] > 0.01
