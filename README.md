# Efficient Portfolios under Convex Risk Measures

**ING3 CY Tech 2025–2026 | M222 - Dauphine 2026-2027**

**Corentin Stephan**

## Abstract

Classical portfolio optimisation relies on the Markowitz mean-variance framework, which identifies variance with financial risk. This identification is problematic because variance penalises upside and downside deviations symmetrically and is not monotone: a position that dominates another in every state can nevertheless have a larger variance.

This project develops an alternative portfolio-selection framework based on **convex and coherent risk measures**, following the axiomatic approach of Föllmer and Schied. The thesis studies the resulting efficient-portfolio problem in three settings: mono-period, discrete-time multi-period, and continuous-time portfolio optimisation.

The theoretical framework begins with monetary risk measures, acceptance sets and the robust representation of convex risk measures. Value-at-Risk is shown explicitly to fail subadditivity, motivating the use of **Average Value-at-Risk (AVaR)** as the main risk measure. AVaR is coherent, admits both a robust representation and a tractable variational formulation, and therefore provides a natural replacement for variance in portfolio optimisation.

The numerical implementation validates the theoretical results through Monte Carlo simulations, finite-scenario linear programming, discrete-time scenario-tree experiments and continuous-time closed-form calculations. An additional contribution studies the statistical estimation risk arising when a portfolio is optimised directly on a finite sample, introducing bootstrap-based robustifications of AVaR and analysing the resulting **optimizer’s curse**.

---

## Overview

This repository implements the theoretical and numerical pipeline of efficient portfolio optimisation under convex risk measures.

The project is divided into three main modelling settings and one additional statistical contribution:

* **Part 1 — Mono-period portfolio optimisation:** axiomatic risk measures, AVaR-efficient frontiers, finite-scenario linear programming, and comparison with the classical Markowitz frontier.
* **Part 2 — Discrete-time dynamic risk:** conditional risk measures, recursive composition, time-consistency and comparison with a naive terminal-law risk assessment.
* **Part 3 — Continuous-time portfolio optimisation:** Black-Scholes market, continuously rebalanced constant-mix strategies, closed-form AVaR and the optimal risky proportion.
* **Part 4 — Distributionally-robust AVaR:** bootstrap-based uncertainty sets, the optimizer’s curse, and comparison between bootstrap-sup, bootstrap-quantile and nested-bootstrap corrections.

The central conclusion is that replacing variance by AVaR changes little under Gaussian/elliptical returns, but can materially change portfolio allocations once returns become skewed. In the numerical experiments, the mean relative portfolio-weight error versus Markowitz rises from approximately **4.44% in the Gaussian case to 34.5% under skewness**.

---

## Repository Structure

```text
├── src/
│   └── convexrisk/
│       ├── __init__.py
│       ├── risk_measures.py       # VaR, AVaR, Gaussian closed forms,
│       │                           # discrete AVaR and variational formula
│       ├── mono_period.py         # AVaR-efficient portfolio as scenario LP
│       ├── markowitz.py            # Classical Markowitz closed-form solution
│       ├── discrete_time.py       # Dynamic risk measures and time-consistency
│       ├── continuous_time.py     # Continuous-time Merton/AVaR solution
│       ├── downside_risk.py       # Harlow lower partial moments
│       └── robust.py              # Bootstrap-robust AVaR and optimizer's curse
│
├── tests/
│   ├── test_risk_measures.py
│   ├── test_discrete_time.py
│   ├── test_continuous_time.py
│   ├── test_robust.py
│   ├── test_discrete_avar_exact.py
│   ├── test_downside_risk.py
│   └── test_mono_period.py
│
├── notebooks/
│   ├── 01_mono_period_avar_frontier.ipynb
│   ├── 02_discrete_time_consistency.ipynb
│   ├── 03_continuous_time_constant_mix.ipynb
│   ├── 04_downside_risk_comparison.ipynb
│   └── 05_distributionally_robust_avar.ipynb
│
├── pyproject.toml
└── README.md
```

The repository also contains generated figures and notebook-building scripts. The five notebooks correspond directly to the five experimental components described in Chapter 6 of the thesis.

---

## Core Modules (`src/convexrisk/`)

### `risk_measures.py`

Core implementation of the risk-measure framework.

Provides:

* empirical Value-at-Risk;
* empirical Average Value-at-Risk;
* exact discrete VaR and AVaR for arbitrary scenario probabilities;
* the Gaussian closed-form AVaR;
* the Gaussian closed-form VaR;
* the AVaR variational representation;
* the Gaussian constant

[
\kappa(\lambda)
===============

\frac{\phi(\Phi^{-1}(\lambda))}{\lambda};
]

* the supremum of several coherent risk measures.

The main convention throughout the package is that a financial position is a **P&L**, so larger values are better and the risk measure returns the associated capital requirement.

For a Gaussian position

[
X\sim\mathcal N(m,\sigma^2),
]

the implementation uses

[
\operatorname{AVaR}_\lambda(X)
==============================

-m+\sigma\kappa(\lambda).
]

This closed form is the key bridge between AVaR and the Markowitz mean-variance framework.

---

### `mono_period.py`

Implements the mono-period AVaR-efficient portfolio problem.

For portfolio weights (\pi), initial wealth (x_0), risk-free rate (r_f), and risky returns (R),

[
Y_\pi
=====

x_0\pi^\top(R-r_f\mathbf 1)
]

is the excess terminal position.

The portfolio optimisation problem is formulated as

[
\min_{\pi}
\operatorname{AVaR}*\lambda(Y*\pi)
]

subject to a target expected excess return.

Using the variational representation of AVaR,

[
\operatorname{AVaR}*\lambda(Y*\pi)
==================================

\min_c
\left[
c+
\frac{1}{\lambda}
\mathbb E[(-Y_\pi-c)^+]
\right],
]

the problem becomes a jointly convex optimisation problem in ((\pi,c)).

For finite scenarios, this is implemented as a **linear program** using `scipy.optimize.linprog`. The implementation uses a sparse constraint matrix so that large scenario sets remain computationally tractable.

---

### `markowitz.py`

Provides the classical Markowitz benchmark.

The module computes the minimum-variance portfolio for a prescribed expected return and is used throughout the experiments to determine whether AVaR produces the same or different efficient portfolios.

Under Gaussian/elliptical returns, the thesis proves that AVaR and Markowitz generate the **same efficient portfolios**, with only the numerical representation of the risk axis changing.

---

### `discrete_time.py`

Implements dynamic risk assessment in a discrete-time market.

The module considers conditional one-step risk measures

[
\rho_t^{(1)}
]

and their recursive composition to construct a dynamic risk measure.

The key theoretical point is that recursive composition provides a **time-consistent** risk measure, whereas simply applying a static risk measure to the terminal distribution can lead to a different value.

The implementation also contains a random search over binary scenario trees used to find an explicit example where the two approaches genuinely diverge.

In the numerical experiment, **3,000 random four-period trees** are generated. The largest absolute gap found is approximately **2.80**, and **37% of the trees exhibit a relative gap greater than 10%**.

---

### `continuous_time.py`

Implements the continuous-time Black-Scholes setting for continuously rebalanced constant-mix strategies.

The risky asset follows the standard Black-Scholes dynamics, and the portfolio invests a constant fraction (\pi) of wealth in the risky asset.

The terminal log-return is Gaussian, allowing an explicit expression for its AVaR.

The resulting optimal risky proportion can be interpreted as

[
\pi^*
=====

## \pi_{\text{Merton}}

\text{AVaR risk-aversion correction}.
]

Thus the classical Merton myopic proportion is recovered when the AVaR correction disappears.

The numerical implementation independently validates the analytical formula using exact Monte Carlo simulation.

---

### `downside_risk.py`

Implements Harlow's lower partial moment criterion

[
LPM_n(\tau,X)
=============

\mathbb E[(\tau-X)_+^n].
]

The module is used as an empirical benchmark against both AVaR and Markowitz.

Unlike AVaR, the fixed-target LPM formulation is **not cash-invariant**, so it is not treated as a monetary/coherent risk measure in the theoretical framework. It is instead used as a downside-risk portfolio criterion for comparison.

---

### `robust.py`

Contains the original computational contribution of the project: **distributionally-robust AVaR and the optimizer's curse**.

The module implements three bootstrap-based approaches:

* **Bootstrap-sup:** maximum AVaR over bootstrap resamples;
* **Bootstrap-quantile:** an upper quantile of bootstrap AVaR estimates;
* **Nested bootstrap:** re-optimisation on each bootstrap sample to estimate the selection bias created by portfolio optimisation.

The theoretical starting point is that the supremum of coherent risk measures is itself coherent. This is verified directly in the code against all four relevant axioms.

However, the experiments show an important tension:

> the estimator with the cleanest coherence guarantee is not necessarily the estimator with the best statistical calibration.

The exact bootstrap-sup overcorrects the optimizer's curse because the maximum of many bootstrap replicates introduces an additional order-statistic effect.

---

## Notebooks

### Mono-period Portfolio Optimisation

| Notebook                             | Description                                                                | Key outputs                                                                |
| ------------------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `01_mono_period_avar_frontier.ipynb` | AVaR-efficient frontier versus Markowitz under Gaussian and skewed returns | Gaussian coincidence, skewness-induced divergence, portfolio-weight errors |

The Gaussian experiment uses three risky assets, 40,000 simulated scenarios and 12 target returns. The AVaR problem is solved through the finite-scenario LP and compared with the closed-form Markowitz solution.

Results:

* Gaussian returns: **4.44% mean relative weight error**;
* left-skewed asset: **34.5% mean relative weight error**.

The skewed asset preserves the same mean and variance as the Gaussian benchmark, meaning Markowitz is unchanged while AVaR reacts to the additional tail asymmetry.

---

### Discrete-Time Dynamic Risk

| Notebook                             | Description                                                                                   | Key outputs                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `02_discrete_time_consistency.ipynb` | Tests recursive dynamic risk and searches for divergence from the naive terminal-law approach | Scenario-tree examples, gap distribution, largest divergence |

The experiment generates 3,000 random binary trees and compares:

[
\rho_0^{\text{naive}}(X_T)
]

with

[
\rho_0(X_T)
]

obtained by recursive composition of conditional risk measures.

Key result:

* maximum absolute gap: **≈ 2.80**;
* **37%** of generated trees have a relative gap above 10%.

This demonstrates that the agreement observed in several hand-built examples is not generic.

---

### Continuous-Time Constant-Mix

| Notebook                                | Description                                                               | Key outputs                                                               |
| --------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `03_continuous_time_constant_mix.ipynb` | Validates the closed-form AVaR of continuous-time constant-mix strategies | Analytical vs. Monte Carlo curves, optimal proportion, limiting behaviour |

For

[
\mu=0.10,\quad r=0.02,\quad \sigma=0.20,\quad
T=1,\quad \lambda=0.05,
]

the closed-form AVaR is compared against **(3\times10^6)** exact Monte Carlo paths for 25 portfolio proportions.

Maximum relative error:

[
\boxed{0.24%}
]

No time-discretisation error is introduced because the terminal distribution is sampled directly from the closed-form solution.

A second experiment uses

[
\mu=0.18,\quad r=0.02,\quad \sigma=0.15,\quad
T=5,\quad \lambda=0.1.
]

A grid search gives

[
\pi^*_{\text{grid}}=1.8785,
]

while the analytical formula gives

[
\pi^*_{\text{closed form}}=1.8788.
]

The difference is attributable to the grid resolution.

---

### Downside Risk Comparison

| Notebook                            | Description                                  | Key outputs                                     |
| ----------------------------------- | -------------------------------------------- | ----------------------------------------------- |
| `04_downside_risk_comparison.ipynb` | Comparison of Harlow LPM, AVaR and Markowitz | Gaussian agreement and skewed-return divergence |

Under Gaussian returns, the three criteria produce portfolios that are numerically close.

Under skewed returns, they diverge significantly:

* AVaR vs. Markowitz: **46.6% mean relative weight error**;
* Harlow vs. AVaR: **11.3% mean relative weight error**.

This highlights the fact that downside-sensitive criteria react differently once the return distribution is no longer elliptical.

---

### Distributionally-Robust AVaR

| Notebook                                | Description                                 | Key outputs                                              |
| --------------------------------------- | ------------------------------------------- | -------------------------------------------------------- |
| `05_distributionally_robust_avar.ipynb` | Bootstrap uncertainty and optimizer's curse | Bootstrap distribution, bias comparison, coherence tests |

The experiment uses:

* (n=50) observations;
* (B=2000) bootstrap resamples;
* AVaR level (\lambda=0.1);
* 400 Monte Carlo trials;
* a known Gaussian population, allowing the true population AVaR to be computed exactly.

The comparison is:

| Estimator | Coherent? | Mean signed bias | Mean (|\text{bias}|) |
|---|---:|---:|---:|
| Naive | Yes | +0.0141 | 0.0200 |
| Bootstrap-sup | Yes | −0.0258 | 0.0331 |
| Bootstrap 80%-quantile | No | +0.0013 | 0.0189 |
| Nested bootstrap | Convex only | ≈ 0 | 0.0169 |

The naive estimator understates the true risk because the portfolio has been selected by minimising the same empirical risk measure. This is the **optimizer's curse**.

The bootstrap-sup estimator is theoretically coherent but overcorrects: its mean absolute bias is about **65% larger** than the naive estimator. The bootstrap-quantile and nested-bootstrap approaches are better calibrated in this experiment, but do not retain the same coherence guarantee.

---

## Main Theoretical Results

### Convex Risk Measures

A monetary risk measure

[
\rho:\mathcal X\rightarrow\mathbb R
]

satisfies:

* **Monotonicity**
* **Cash invariance**

A convex risk measure additionally satisfies

[
\rho(\lambda X+(1-\lambda)Y)
\leq
\lambda\rho(X)+(1-\lambda)\rho(Y).
]

A coherent risk measure adds positive homogeneity, which implies subadditivity.

The thesis develops the robust representation

[
\rho(X)
=======

\sup_{Q}
\left(
\mathbb E_Q[-X]-\alpha_{\min}(Q)
\right),
]

where (\alpha_{\min}) is the minimal penalty function.

In the coherent case, the penalty reduces to a (0/\infty) indicator, giving

[
\rho(X)=\sup_{Q\in\mathcal Q}\mathbb E_Q[-X].
]

The representation is proved explicitly for finite probability spaces in the thesis.

---

### Value-at-Risk

VaR is defined as

[
\operatorname{VaR}_\lambda(X)
=============================

\inf
\left{
m:
P(X+m<0)\leq\lambda
\right}.
]

It satisfies monotonicity, cash invariance and positive homogeneity, but is **not generally convex or subadditive**.

An explicit two-default counterexample demonstrates that diversification can increase VaR, violating the fundamental subadditivity property expected from a coherent risk measure.

---

### Average Value-at-Risk

AVaR is defined by averaging VaR over the lower tail:

[
\operatorname{AVaR}_\lambda(X)
==============================

\frac{1}{\lambda}
\int_0^\lambda
\operatorname{VaR}_s(X),ds.
]

It admits the robust representation

[
\operatorname{AVaR}_\lambda(X)
==============================

\sup_{Q\in\mathcal Q_\lambda}
\mathbb E_Q[-X],
]

with

[
\mathcal Q_\lambda
==================

\left{
Q\ll P:
\frac{dQ}{dP}\leq\frac1\lambda
\right}.
]

It also has the computationally important representation

[
\operatorname{AVaR}_\lambda(X)
==============================

\min_c
\left[
c+
\frac1\lambda
\mathbb E(-X-c)^+
\right].
]

This formulation is what makes the portfolio problem numerically tractable.

---

## Efficient Portfolio Problem

The mono-period problem is formulated in two equivalent ways.

### Risk budget

[
\sup_{\pi\in\Theta}
\mathbb E[Y_\pi]
]

subject to

[
\operatorname{AVaR}*\lambda(Y*\pi)\leq\bar c.
]

### Target return

[
\inf_{\pi\in\Theta}
\operatorname{AVaR}*\lambda(Y*\pi)
]

subject to

[
\mathbb E[Y_\pi]\geq\bar\mu.
]

Because AVaR is convex and the portfolio payoff is linear in (\pi), these are convex optimisation problems whenever the portfolio constraint set (\Theta) is convex.

---

## Gaussian / Markowitz Equivalence

For

[
R\sim\mathcal N(r_f\mathbf 1+\mu,\Sigma),
]

the excess portfolio payoff satisfies

[
Y_\pi
\sim
\mathcal N
\left(
x_0\pi^\top\mu,
x_0^2\pi^\top\Sigma\pi
\right).
]

Therefore

[
\operatorname{AVaR}*\lambda(Y*\pi)
==================================

-x_0\pi^\top\mu
+
\kappa(\lambda)x_0
\sqrt{\pi^\top\Sigma\pi}.
]

Consequently, for a fixed target return, the AVaR-efficient portfolio is exactly the Markowitz minimum-variance portfolio.

Thus:

> **Under elliptical returns, replacing variance by AVaR does not change the efficient portfolios.**

The difference appears once the return distribution contains features such as skewness that are invisible to the first two moments.

---

## Continuous-Time Result

In the Black-Scholes market, continuously rebalanced constant-mix strategies lead to a Gaussian terminal log-return.

The AVaR-optimal portfolio proportion has the structure

[
\pi^*
=====

## \pi_{\mathrm{Merton}}

\text{risk-aversion correction}.
]

The correction:

* decreases risky exposure;
* disappears as (T\to\infty);
* disappears as (\lambda\to1);
* can become sufficiently large that the optimal solution is the corner (\pi^*=0).

The thesis explicitly restricts this closed-form result to a **single risky asset and deterministic constant-mix strategies**; genuinely adapted strategies and multi-asset continuous-time markets are not solved in closed form.

---

## Key Results

| Experiment                           | Result                                                      |
| ------------------------------------ | ----------------------------------------------------------- |
| Mono-period — Gaussian               | AVaR vs. Markowitz mean weight error: **4.44%**             |
| Mono-period — skewed                 | AVaR vs. Markowitz mean weight error: **34.5%**             |
| Continuous-time AVaR                 | Maximum relative Monte Carlo error: **0.24%**               |
| Continuous-time optimum              | Closed form (1.8788) vs. grid (1.8785)                      |
| Discrete-time consistency            | Maximum absolute gap: **2.80**                              |
| Discrete-time consistency            | **37%** of trees have >10% relative gap                     |
| Robust AVaR — naive                  | Mean signed bias **+0.0141**, mean absolute bias **0.0200** |
| Robust AVaR — bootstrap-sup          | Mean signed bias **−0.0258**, mean absolute bias **0.0331** |
| Robust AVaR — bootstrap 80% quantile | Mean signed bias **+0.0013**, mean absolute bias **0.0189** |
| Robust AVaR — nested bootstrap       | Mean signed bias ≈ **0**, mean absolute bias **0.0169**     |
| Harlow — Gaussian                    | Criteria essentially agree within Monte Carlo noise         |
| Harlow — skewed                      | AVaR vs. Markowitz **46.6%**, Harlow vs. AVaR **11.3%**     |

The full experimental programme and its numerical values are summarised directly in the thesis.

---

## Requirements

```text
Python >= 3.10
numpy >= 1.24
scipy >= 1.10
pytest >= 7.0
matplotlib >= 3.7
jupyter
```

The project is packaged with `pyproject.toml` and uses a standard editable installation.

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest tests/ -v
```

Execute a notebook:

```bash
jupyter nbconvert --to notebook --execute \
    notebooks/01_mono_period_avar_frontier.ipynb \
    --output notebooks/01_mono_period_avar_frontier.ipynb
```

The same procedure applies to notebooks `02` through `05`.

---

## Tests

The repository contains a dedicated regression test suite covering the principal theoretical and numerical components:

```text
tests/
├── test_risk_measures.py
├── test_discrete_time.py
├── test_continuous_time.py
├── test_robust.py
├── test_discrete_avar_exact.py
├── test_downside_risk.py
└── test_mono_period.py
```

The thesis reports **45 tests** for the version used in the final manuscript, while the repository README currently states **42 tests**. The README also states that the core theory is fully implemented and tested.

The test suite covers, among other things:

* VaR/AVaR properties;
* exact discrete AVaR;
* existence of mono-period efficient portfolios;
* Gaussian coincidence with Markowitz;
* discrete-time recursive risk;
* continuous-time AVaR closed forms;
* Harlow downside risk;
* robustness/coherence properties.

The thesis explicitly maps theoretical claims to corresponding tests and notebooks.

---

## Reproducibility

All numerical results reported in the thesis are generated from the accompanying `convexrisk` repository.

The computational workflow is deliberately traceable:

```text
Theory
   │
   ├── Risk-measure axioms
   ├── Robust representation
   ├── AVaR formulation
   └── Portfolio optimisation
           │
           ▼
Implementation
   │
   ├── Core Python modules
   ├── Regression tests
   └── Numerical notebooks
           │
           ▼
Experiments
   │
   ├── Monte Carlo
   ├── Linear programming
   ├── Scenario-tree search
   └── Bootstrap experiments
```

The thesis also documents two bugs discovered during development:

1. a dense LP constraint matrix exhausted memory at around 20,000 scenarios and was replaced by a sparse implementation;
2. an initially incorrect claim concerning coherence of the nested-bootstrap estimator was detected and corrected through direct numerical testing.

These corrections are retained as part of the regression-testing philosophy of the project.

---

## Limitations

The project deliberately focuses on **AVaR**, rather than solving the efficient-portfolio problem for an arbitrary convex risk measure.

The mono-period existence result is also established under a compact portfolio constraint set; general coercivity on the unconstrained space (\mathbb R^d) is not claimed.

The numerical experiments use only two or three risky assets, so the magnitude of the reported portfolio-weight differences has not been systematically tested as the number of assets increases.

The discrete-time time-consistency result is supported by a random search rather than a complete characterisation of all scenario trees.

Finally, the bootstrap contribution is validated on a controlled Gaussian experiment with a known population distribution. No out-of-sample historical backtest is performed, so the reported bias corrections should be interpreted as controlled numerical evidence rather than as empirical estimates for real financial markets.

---

## References

* Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. — *Thinking Coherently*, coherent risk measures and their axiomatic foundations.
* Föllmer, H. & Schied, A. — *Convex Measures of Risk*.
* Föllmer, H. & Schied, A. — *Stochastic Finance: An Introduction in Discrete Time*.
* Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
* Pliska, S.R. (1998). *Introduction to Mathematical Finance: Discrete Time Models*.
* Harlow, W.V. (1991). *Asset Allocation in a Downside-Risk Framework*.
* Taflin, E. (2000). Insurance application of dynamic risk measures.
* Bassi, F., Embrechts, P. & Kafetzaki, M. (1997). Reliable estimation of extreme quantiles.
* Merton, R.C. — continuous-time portfolio optimisation and the classical myopic proportion.

---

## Project Summary

The project can be viewed as the following progression:

[
\boxed{
\text{Variance}
\rightarrow
\text{Convex Risk Measures}
\rightarrow
\text{AVaR}
\rightarrow
\text{Efficient Portfolios}
}
]

with three successive levels of complexity:

[
\boxed{
\text{Mono-period}
\rightarrow
\text{Discrete-time}
\rightarrow
\text{Continuous-time}
}
]

and a final statistical extension:

[
\boxed{
\text{AVaR optimisation}
\rightarrow
\text{Finite-sample estimation}
\rightarrow
\text{Optimizer's Curse}
\rightarrow
\text{Bootstrap Robustification}.
}
]

The main empirical message is deliberately nuanced: **AVaR reproduces Markowitz under Gaussian/elliptical returns, but becomes materially different when higher-order distributional features such as skewness matter.** The project then extends this distinction to dynamic risk assessment and continuous-time portfolio choice, while the bootstrap contribution highlights the separate problem of estimating a risk measure after optimising against the same finite sample.
