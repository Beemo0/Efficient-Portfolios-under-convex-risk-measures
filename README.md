# Efficient Portfolios under Convex Risk Measures

**ING3 CY Tech 2025–2026 | M222 - Dauphine 2026–2027**

**Corentin Stephan**

## Abstract

Portfolio optimisation is traditionally formulated through expected return
and variance, leading to the classical Markowitz framework. This approach,
however, only captures the first two moments of the return distribution and
can fail to reflect asymmetric or tail risk. This project studies portfolio
optimisation under convex and coherent risk measures, with a particular
focus on Average Value-at-Risk (AVaR).

The project develops the mathematical framework from the axiomatic
definition of monetary risk measures to their robust representation, and
then implements the corresponding portfolio optimisation problems in
mono-period, discrete-time, and continuous-time settings. AVaR is studied
through its variational representation, its Gaussian closed form, and its
interpretation as a worst-case expectation over a family of probability
measures. The resulting portfolios are compared with classical Markowitz
and downside-risk portfolios, first under Gaussian returns and then under
non-elliptical distributions.

The numerical part also investigates two extensions. The first is the
construction of time-consistent dynamic risk measures through recursive
conditional AVaR in discrete time. The second addresses estimation risk
and the optimizer's curse through bootstrap-based distributionally-robust
AVaR estimators.

## Overview

This repository implements the full theoretical and numerical pipeline
of portfolio optimisation under convex risk measures, in mono-period,
discrete-time, and continuous-time settings.

The project is divided into five parts, corresponding to the main
numerical developments:

* **Part 1 Mono-period portfolio optimisation:** AVaR-efficient portfolios,
  the finite-scenario linear-programming formulation, and comparison with
  the classical Markowitz frontier.
* **Part 2 Dynamic risk measures:** recursive conditional AVaR, time
  consistency, and numerical search for divergence between recursive and
  naive terminal risk assessments.
* **Part 3 Continuous-time optimisation:** constant-mix strategies,
  closed-form AVaR, exact Monte Carlo validation, and the AVaR correction
  to the classical Merton myopic proportion.
* **Part 4 Downside risk:** comparison between AVaR, Markowitz variance,
  and Harlow's lower partial moments under Gaussian and skewed returns.
* **Part 5 Distributionally-robust AVaR:** bootstrap-based risk estimation,
  coherence tests, and the optimizer's-curse experiment.

## Repository Structure

```text
├── Part 1/
│   └── 01_mono_period_avar_frontier.ipynb    # AVaR-efficient frontier vs. Markowitz
│
├── Part 2/
│   └── 02_discrete_time_consistency.ipynb    # recursive AVaR and time consistency
│
├── Part 3/
│   └── 03_continuous_time_constant_mix.ipynb # closed-form AVaR + Monte Carlo
│
├── Part 4/
│   └── 04_downside_risk_comparison.ipynb     # AVaR vs. Markowitz vs. Harlow LPM
│
├── Part 5/
│   └── 05_distributionally_robust_avar.ipynb # bootstrap and optimizer's curse
│
├── src/
│   └── convexrisk/
│       ├── __init__.py
│       ├── risk_measures.py          # VaR, AVaR, Gaussian formulas and
│       │                             # variational representations
│       ├── mono_period.py            # AVaR-efficient portfolio optimisation
│       ├── markowitz.py              # classical Markowitz closed forms
│       ├── discrete_time.py          # recursive conditional risk measures
│       ├── continuous_time.py        # constant-mix AVaR closed form
│       ├── downside_risk.py          # Harlow lower partial moments
│       └── robust.py                 # distributionally-robust AVaR
│
├── tests/
│   ├── test_risk_measures.py
│   ├── test_discrete_avar_exact.py
│   ├── test_mono_period.py
│   ├── test_discrete_time.py
│   ├── test_continuous_time.py
│   ├── test_downside_risk.py
│   └── test_robust.py
│
├── pyproject.toml
└── README.md
```

## Core Modules (`src/convexrisk/`)

### `risk_measures.py`

Implementation of the main risk-measure primitives used throughout the
project.

The module provides:

* Value-at-Risk (VaR);
* Average Value-at-Risk (AVaR);
* the variational representation of AVaR;
* exact AVaR computation for discrete distributions;
* closed-form formulas for Gaussian positions;
* the supremum representation used to construct distributionally-robust
  risk measures.

For a Gaussian position

$[X \sim \mathcal{N}(m,\sigma^2),]$

the AVaR admits the closed-form expression

$[\operatorname{AVaR}_{\lambda}(X) = -m+\kappa(\lambda)\sigma,]$

where

$[\kappa(\lambda) = \frac{\phi\left(\Phi^{-1}(\lambda)\right)}{\lambda}.]$

This formula is used throughout the project to derive analytical
benchmarks and validate numerical optimisation procedures.

### `mono_period.py`

Finite-scenario AVaR portfolio optimisation.

The module implements the variational reformulation of the AVaR minimisation
problem as a convex optimisation problem and computes the corresponding
efficient frontier.

For a portfolio (\pi) and excess-return vector

$[R-r_f\mathbf{1},]$

the terminal excess position is

$[Y_{\pi} = x_0\pi^\top(R-r_f\mathbf{1}).]$

Under a finite scenario distribution, the AVaR problem can therefore be
formulated as a linear program. The resulting AVaR-efficient portfolios
are used to compare tail-risk optimisation with the classical Markowitz
solution.

### `markowitz.py`

Classical mean-variance portfolio optimisation used as the benchmark
throughout the mono-period experiments.

The module provides:

* the closed-form Markowitz portfolio for a target expected return;
* the corresponding efficient frontier;
* analytical portfolio weights under the standard fully invested constraint.

Under Gaussian returns, AVaR depends only on the portfolio mean and
standard deviation. Consequently, the AVaR and Markowitz efficient
portfolios coincide in the population model, providing a natural
benchmark for the numerical experiments.

### `discrete_time.py`

Dynamic risk-measure framework on finite binary scenario trees.

The module implements:

* binary scenario trees;
* terminal probability distributions;
* one-step conditional AVaR;
* recursively composed conditional AVaR;
* naive terminal AVaR;
* divergence measures between recursive and static risk assessments.

The recursive construction is used to study **time consistency** and to
search numerically for scenario trees on which recursive and naive
evaluations differ substantially.

### `continuous_time.py`

Continuous-time constant-mix portfolio optimisation.

Consider a risky asset with drift (\mu), volatility (\sigma), risk-free
rate (r), investment horizon (T), and constant risky proportion (\pi).
The terminal log-return is Gaussian, which allows its AVaR to be computed
in closed form.

The module implements:

* terminal log-return moments;
* closed-form AVaR;
* the analytical optimal constant-mix proportion;
* exact Monte Carlo simulation from the closed-form terminal distribution;
* numerical validation of the analytical optimum.

When the optimal risky proportion is positive,

$[\pi^*(\lambda,T) = \frac{\mu-r}{\sigma^2}\frac{\kappa(\lambda)}{\sigma\sqrt{T}}.]$

The first term corresponds to the classical Merton myopic proportion,
while the second term is the explicit AVaR risk-aversion correction.

This expression makes it possible to study directly how tail-risk aversion
depends on the confidence level and investment horizon.

### `downside_risk.py`

Implementation of Harlow's lower partial moments.

The module is used to construct downside-risk portfolio criteria and
compare them with AVaR and Markowitz optimisation.

The experiments highlight the differences between:

* variance-based risk measures;
* tail-based AVaR;
* target-based downside-risk measures.

These differences become particularly visible when the return distribution
is asymmetric.

### `robust.py`

Distributionally-robust AVaR and bootstrap-based estimation.

The module implements:

* bootstrap AVaR;
* bootstrap-sup AVaR;
* bootstrap-quantile AVaR;
* nested-bootstrap correction;
* coherence tests;
* optimizer's-curse experiments.

The objective is to study the gap between the empirical AVaR used during
portfolio selection and the population risk of the portfolio ultimately
selected by the optimisation procedure.

In particular, the experiments investigate whether bootstrap-based
corrections can reduce estimation bias without sacrificing the structural
properties expected from coherent risk measures.



## Notebooks

### Part 1 Mono-Period Portfolio Optimisation

| Notebook                             | Description                                                                         | Key outputs                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `01_mono_period_avar_frontier.ipynb` | Finite-scenario AVaR optimisation compared with the closed-form Markowitz portfolio | AVaR-efficient frontier, Markowitz frontier, portfolio-weight errors under Gaussian and skewed returns |

### Part 2 Dynamic Risk Measures

| Notebook                             | Description                                                                                               | Key outputs                                                                       |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `02_discrete_time_consistency.ipynb` | Comparison between naive terminal AVaR and recursively composed conditional AVaR on binary scenario trees | Time-consistency checks, divergence distributions, largest random-tree divergence |

### Part 3 Continuous-Time Optimisation

| Notebook                                | Description                                                                                            | Key outputs                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `03_continuous_time_constant_mix.ipynb` | Validation of the closed-form constant-mix AVaR formula and analytical optimum using exact Monte Carlo | Closed-form vs. Monte Carlo AVaR, residuals, optimal risky proportion, horizon and confidence-level effects |

### Part 4 Downside Risk

| Notebook                            | Description                                                              | Key outputs                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `04_downside_risk_comparison.ipynb` | Comparison of AVaR, Markowitz variance, and Harlow lower partial moments | Efficient frontiers, portfolio weights, Gaussian vs. skewed-return comparison |

### Part 5 Distributionally-Robust AVaR

| Notebook                                | Description                                                                  | Key outputs                                                                              |
| --------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `05_distributionally_robust_avar.ipynb` | Study of bootstrap corrections for estimation risk and the optimizer's curse | Bootstrap risk estimates, coherence checks, bias comparison, robust optimisation results |



## Key Results

| Experiment                            | Result                                                                                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **E1 Mono-period Gaussian**         | AVaR and Markowitz frontiers essentially overlap, with a mean relative portfolio-weight error of **4.44%** across the tested targets.                           |
| **E2 Mono-period skewed**           | Replacing one Gaussian marginal with a left-skewed distribution increases the mean relative weight error to **34.5%**, while preserving the Markowitz frontier. |
| **E3 Continuous-time AVaR**         | Closed-form AVaR and exact Monte Carlo estimates agree with a maximum relative error of **0.24%** across the tested constant-mix proportions.                   |
| **E3 Continuous-time optimum**      | The analytical optimum is (\pi^*=1.8788), compared with a grid-search optimum of (\pi^*=1.8785).                                                                |
| **E3 Long-horizon behaviour**       | The AVaR correction decreases as (T) increases, and the optimal risky proportion converges towards the Merton myopic proportion.                                |
| **E3 Confidence-level behaviour**   | The AVaR correction vanishes as (\lambda\to1), recovering the classical expected-return criterion.                                                              |
| **E4 Discrete-time consistency**    | Recursive and naive AVaR can differ substantially on general binary scenario trees; the largest absolute divergence found numerically is **2.80**.              |
| **E4 Random-tree search**           | **37%** of the tested random trees exhibit a relative divergence greater than **10%**.                                                                          |
| **E5 Distributionally-robust AVaR** | Bootstrap-sup produces a coherent but conservative risk estimator and can substantially overcorrect the optimizer's curse.                                      |
| **E5 Nested bootstrap**             | Nested bootstrap improves empirical bias calibration in the controlled experiment but does not preserve full coherence.                                         |
| **E5 Coherence tests**              | Bootstrap-sup satisfies the tested coherence properties, whereas the better-calibrated bootstrap variants do not retain full positive homogeneity.              |


## Requirements

* Python `>= 3.10`
* `numpy`
* `scipy`
* `matplotlib`
* `pytest`
* `jupyter`

## Installation

Create and activate a virtual environment, then install the project in
editable mode with the development and notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

Run the complete test suite with:

```bash
python -m pytest tests/ -v
```


## References

- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). *Coherent Measures
  of Risk*. Mathematical Finance, 9(3), 203–228.
- Föllmer, H. & Schied, A. (2002). *Convex Measures of Risk and Trading
  Constraints*. Finance and Stochastics, 6, 429–447.
- Föllmer, H. & Schied, A. *Stochastic Finance: An Introduction in Discrete Time*.
- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
- Harlow, W. V. (1991). *Asset Allocation in a Downside-Risk
  Framework*. Financial Analysts Journal, 47(5), 28–40.
- Pliska, S. R. (1998). *Introduction to Mathematical
  Finance: Discrete Time Models*.
- Kusuoka, S. (2001). *On Law Invariant Coherent Risk
  Measures*. Advances in Mathematical Economics, 3.
- Rockafellar, R. T. & Uryasev, S. (2000). *Optimization of Conditional
  Value-at-Risk*. Journal of Risk, 2, 21–41.
