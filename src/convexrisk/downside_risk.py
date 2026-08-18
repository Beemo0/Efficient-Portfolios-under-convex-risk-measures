"""
Harlow's lower partial moments (Section 2.6, eq. 2.16), used purely as an
empirical benchmark against the AVaR-efficient frontier (never as an
axiomatic convex risk measure with a fixed target, per Remark 2.1).
"""
from __future__ import annotations

import numpy as np


def lower_partial_moment(pnl: np.ndarray, target: float, order: float) -> float:
    """LPM_order(target, X) = E[((target - X)^+)^order], eq. 2.16."""
    pnl = np.asarray(pnl, dtype=float)
    if order < 1:
        raise ValueError("order must be >= 1 for LPM to be convex in X (Remark 2.1)")
    shortfall = np.maximum(target - pnl, 0.0)
    return float(np.mean(shortfall ** order))


def semi_variance(pnl: np.ndarray, target: float | None = None) -> float:
    """Classical semi-variance: LPM_2 with target = E[X] if not given."""
    pnl = np.asarray(pnl, dtype=float)
    if target is None:
        target = float(np.mean(pnl))
    return lower_partial_moment(pnl, target, order=2.0)


def harlow_efficient_portfolio(
    scenario_returns: np.ndarray,
    scenario_probs: np.ndarray,
    target_return: float,
    order: float = 2.0,
    lpm_target: float = 0.0,
    x0: float = 1.0,
) -> dict:
    """Minimise LPM_order(lpm_target, Y_pi) subject to E[Y_pi] >=
    target_return, over pi in R^d (Harlow 1991, Section 2.6), by direct
    smooth nonlinear optimisation (SLSQP). Unlike the AVaR problem, this
    is not exactly a linear program, but the objective is convex for
    order >= 1 (Remark 2.1), so a local SLSQP solution from a reasonable
    starting point is the global optimum in the well-behaved cases used
    in this thesis (verified against the closed-form Markowitz solution
    at order=2, lpm_target=mean, in the Gaussian case in the
    accompanying tests, where the two are known to be proportional up to
    a factor of 2 for a symmetric distribution).
    """
    from scipy.optimize import minimize

    scenario_returns = np.asarray(scenario_returns, dtype=float)
    scenario_probs = np.asarray(scenario_probs, dtype=float)
    S, d = scenario_returns.shape

    def objective(pi):
        pnl = x0 * (scenario_returns @ pi)
        shortfall = np.maximum(lpm_target - pnl, 0.0)
        return float(scenario_probs @ (shortfall ** order))

    def objective_grad(pi):
        pnl = x0 * (scenario_returns @ pi)
        shortfall = np.maximum(lpm_target - pnl, 0.0)
        # d/dpi_j [ p_s * shortfall_s^order ] = p_s * order * shortfall_s^(order-1) * (-x0 r_sj)
        weight = scenario_probs * order * shortfall ** (order - 1.0)
        return -x0 * (scenario_returns * weight[:, None]).sum(axis=0)

    def return_constraint(pi):
        return x0 * float(scenario_probs @ (scenario_returns @ pi)) - target_return

    def return_constraint_grad(pi):
        return x0 * (scenario_probs[:, None] * scenario_returns).sum(axis=0)

    pi0 = np.zeros(d)
    result = minimize(
        objective, pi0, jac=objective_grad, method="SLSQP",
        constraints=[{
            "type": "ineq", "fun": return_constraint, "jac": return_constraint_grad
        }],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    pi = result.x
    expected_return = x0 * float(scenario_probs @ (scenario_returns @ pi))
    return {
        "success": bool(result.success),
        "pi": pi,
        "lpm": float(result.fun),
        "expected_return": expected_return,
    }
