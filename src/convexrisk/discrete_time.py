"""
Discrete-time multi-period dynamic risk measures on a non-recombining
binary tree, following Chapter 4: the recursive composition
(Proposition 4.1, automatically time-consistent) versus the naive
application of AVaR directly to the terminal (unconditional) law
(Remark 4.1, not automatically time-consistent). Section
:func:`search_for_divergence` implements the numerical search flagged
explicitly in Chapter 4 as future computational work, since several
hand-constructed trees in preparing that chapter collapsed to exact
coincidence.

Tree representation
--------------------
A tree of depth ``T`` has, at depth ``t`` (t = 0, ..., T-1), ``2**t``
non-leaf nodes, each with:
  - ``cond_prob_up[t][i]``  : P(up-move | node i at depth t)
  - ``incr_up[t][i]``, ``incr_down[t][i]`` : one-step P&L increment.
Leaves (depth T) are indexed 0, ..., 2**T - 1 in the natural binary
order (node i at depth t has children 2*i (up) and 2*i+1 (down)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk_measures import discrete_avar


@dataclass
class BinaryTree:
    depth: int
    cond_prob_up: list[np.ndarray]
    incr_up: list[np.ndarray]
    incr_down: list[np.ndarray]


def generate_random_tree(depth: int, rng: np.random.Generator,
                          prob_range: tuple[float, float] = (0.1, 0.9),
                          increment_scale: float = 1.0) -> BinaryTree:
    """Generate a tree with independent random conditional probabilities
    and one-step increments at every node (state-dependent, not i.i.d.
    across nodes at the same depth)."""
    cond_prob_up = [rng.uniform(*prob_range, size=2 ** t) for t in range(depth)]
    incr_up = [rng.normal(0.0, increment_scale, size=2 ** t) for t in range(depth)]
    incr_down = [rng.normal(0.0, increment_scale, size=2 ** t) for t in range(depth)]
    return BinaryTree(depth, cond_prob_up, incr_up, incr_down)


def terminal_distribution(tree: BinaryTree) -> tuple[np.ndarray, np.ndarray]:
    """Return (payoffs, probs), each of length 2**depth: the terminal
    cumulative P&L and unconditional probability of every leaf."""
    payoffs = np.array([0.0])
    probs = np.array([1.0])
    for t in range(tree.depth):
        n = len(payoffs)
        new_payoffs = np.empty(2 * n)
        new_probs = np.empty(2 * n)
        p_up = tree.cond_prob_up[t]
        new_payoffs[0::2] = payoffs + tree.incr_up[t]
        new_probs[0::2] = probs * p_up
        new_payoffs[1::2] = payoffs + tree.incr_down[t]
        new_probs[1::2] = probs * (1.0 - p_up)
        payoffs, probs = new_payoffs, new_probs
    return payoffs, probs


def naive_static_avar(tree: BinaryTree, level: float) -> float:
    """rho_0^naive(X) = AVaR_level applied directly to the unconditional
    terminal distribution of X (Remark 4.1)."""
    payoffs, probs = terminal_distribution(tree)
    return discrete_avar(payoffs, probs, level)


def recursive_composed_rho0(tree: BinaryTree, one_step_level: float) -> float:
    """rho_0(X) built by the recursive composition of Proposition 4.1,
    using the *same* one-step confidence level at every node. Returns
    the scalar rho_0(X); intermediate values rho_t(X) are internal.
    """
    payoffs, _ = terminal_distribution(tree)
    values = -payoffs.copy()  # rho_T(X) = -X at the leaves (eq. 4.4)
    for t in range(tree.depth - 1, -1, -1):
        n = 2 ** t
        p_up = tree.cond_prob_up[t]
        new_values = np.empty(n)
        for i in range(n):
            z_up = -values[2 * i]
            z_down = -values[2 * i + 1]
            new_values[i] = discrete_avar(
                np.array([z_up, z_down]),
                np.array([p_up[i], 1.0 - p_up[i]]),
                one_step_level,
            )
        values = new_values
    return float(values[0])


def relative_divergence(tree: BinaryTree, one_step_level: float, static_level: float) -> float:
    """Relative gap |naive - composed| / max(|composed|, 1e-8) between the
    two dynamic-risk-measure conventions of Remark 4.1."""
    naive = naive_static_avar(tree, static_level)
    composed = recursive_composed_rho0(tree, one_step_level)
    return abs(naive - composed) / max(abs(composed), 1e-8), naive, composed


def search_for_divergence(
    depth: int,
    n_trials: int,
    rng: np.random.Generator,
    one_step_level: float = 0.3,
    static_level: float = 0.1,
    increment_scale: float = 1.0,
) -> dict:
    """Random search over ``n_trials`` independently generated trees for
    the configuration exhibiting the largest relative divergence between
    the naive and recursively-composed dynamic risk measures. Returns the
    best tree found together with the two values and the relative gap.
    """
    best = {"relative_gap": -1.0}
    for _ in range(n_trials):
        tree = generate_random_tree(depth, rng, increment_scale=increment_scale)
        gap, naive, composed = relative_divergence(tree, one_step_level, static_level)
        if gap > best["relative_gap"]:
            best = {
                "relative_gap": gap,
                "naive": naive,
                "composed": composed,
                "tree": tree,
            }
    return best
