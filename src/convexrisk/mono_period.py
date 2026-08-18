"""
Mono-period AVaR-efficient portfolio problem, following Chapter 3.

Given a finite set of S scenarios (excess returns r_s in R^d, probability
p_s), problem (P2) is reformulated as the linear program of Section 3.5
(eq. 3.20-3.23):

    min_{pi, c, u} c + (1/level) sum_s p_s u_s
    s.t.           u_s >= -x0 pi.r_s - c   for every s
                   u_s >= 0
                   x0 sum_s p_s pi.r_s >= mu_bar
                   pi in Theta (box constraints, optional)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix


def avar_efficient_portfolio(
    scenario_returns: np.ndarray,
    scenario_probs: np.ndarray,
    level: float,
    target_return: float,
    x0: float = 1.0,
    pi_bounds: tuple[float, float] | None = None,
) -> dict:
    """Solve the finite-scenario AVaR-efficient portfolio LP (eq. 3.20-3.23).

    Parameters
    ----------
    scenario_returns : (S, d) array of excess returns r_s per scenario.
    scenario_probs   : (S,) array of scenario probabilities, summing to 1.
    level            : AVaR confidence level in (0, 1).
    target_return    : minimum required expected excess return (eq. 3.23).
    x0               : initial wealth.
    pi_bounds        : optional (lo, hi) box applied to every pi^i
                        (Proposition 3.2's compact-Theta existence case).

    Returns
    -------
    dict with keys: pi (d,), c, avar, expected_return, success.
    """
    scenario_returns = np.asarray(scenario_returns, dtype=float)
    scenario_probs = np.asarray(scenario_probs, dtype=float)
    S, d = scenario_returns.shape
    if scenario_probs.shape != (S,):
        raise ValueError("scenario_probs must have shape (S,)")
    if not np.isclose(scenario_probs.sum(), 1.0, atol=1e-8):
        raise ValueError("scenario_probs must sum to 1")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")

    # Decision vector z = (pi_1,...,pi_d, c, u_1,...,u_S).
    n_pi, n_c, n_u = d, 1, S
    n = n_pi + n_c + n_u

    def idx_pi(j):
        return j

    def idx_c():
        return n_pi

    def idx_u(s):
        return n_pi + n_c + s

    # Objective: c + (1/level) sum_s p_s u_s.
    cost = np.zeros(n)
    cost[idx_c()] = 1.0
    for s in range(S):
        cost[idx_u(s)] = scenario_probs[s] / level

    # Constraint (3.21): u_s + x0 pi.r_s + c >= 0  <=>  -u_s - x0 r_s.pi - c <= 0
    # Built sparsely: each of the S rows only touches d+2 of the S+d+1
    # columns, so a dense (S, S+d+1) array would waste O(S^2) memory for
    # large scenario sets.
    A_ub = lil_matrix((S + 1, n))
    b_ub = np.zeros(S + 1)
    for s in range(S):
        A_ub[s, idx_u(s)] = -1.0
        A_ub[s, idx_c()] = -1.0
        for j in range(d):
            A_ub[s, idx_pi(j)] = -x0 * scenario_returns[s, j]
        b_ub[s] = 0.0

    # Constraint (3.23): x0 sum_s p_s pi.r_s >= target_return
    #   <=>  -x0 sum_s p_s r_s . pi <= -target_return
    expected_excess_return_coeffs = x0 * (scenario_probs[:, None] * scenario_returns).sum(axis=0)
    for j in range(d):
        A_ub[S, idx_pi(j)] = -expected_excess_return_coeffs[j]
    b_ub[S] = -target_return

    A_ub = csr_matrix(A_ub)

    # Bounds: pi in pi_bounds (or unbounded), c unbounded, u_s >= 0 (3.22).
    bounds = []
    for _ in range(d):
        bounds.append((pi_bounds[0], pi_bounds[1]) if pi_bounds is not None else (None, None))
    bounds.append((None, None))  # c
    for _ in range(S):
        bounds.append((0.0, None))  # u_s >= 0

    result = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return {"success": False, "message": result.message}

    z = result.x
    pi = z[:d]
    c = z[idx_c()]
    avar_value = result.fun
    expected_return = x0 * float(scenario_probs @ (scenario_returns @ pi))

    return {
        "success": True,
        "pi": pi,
        "c": c,
        "avar": avar_value,
        "expected_return": expected_return,
    }


def avar_efficient_frontier(
    scenario_returns: np.ndarray,
    scenario_probs: np.ndarray,
    level: float,
    targets: np.ndarray,
    x0: float = 1.0,
    pi_bounds: tuple[float, float] | None = None,
) -> list[dict]:
    """Trace the AVaR-efficient frontier by solving the LP for each target
    return in ``targets``. Returns the list of solve results, in the same
    order as ``targets`` (infeasible targets are marked success=False)."""
    return [
        avar_efficient_portfolio(
            scenario_returns, scenario_probs, level, mu, x0=x0, pi_bounds=pi_bounds
        )
        for mu in targets
    ]
