# Efficient Portfolios under Convex Risk Measures

**Thesis, CREST/CEREMADE — 2026**
Corentin Stephan
Supervisor: R. Tankov

## Status

Core theory (Chapters 2-5) fully implemented and tested (42 tests, all
passing). Five notebooks, each executed with real figures embedded, cover
every experiment described in Chapter 6.

## Structure

```
src/convexrisk/
    risk_measures.py    # VaR/AVaR (empirical + exact discrete), Gaussian
                        # closed form (Lemma 2.1), variational formula
                        # cross-check, sup-of-coherent-measures (Prop 6.1)
    mono_period.py      # AVaR-efficient frontier as a scenario LP (Ch. 3)
    markowitz.py        # classical Markowitz closed form, for comparison
    discrete_time.py    # recursive dynamic risk measure (Prop 4.1),
                        # naive comparison, and a random search for a
                        # genuine time-inconsistency example
    continuous_time.py  # Merton constant-mix closed form (Thm 5.1) +
                        # exact Monte Carlo validation
    downside_risk.py    # Harlow's lower partial moments (Sec. 2.6)
    robust.py           # ORIGINAL CONTRIBUTION (Ch. 6.5): bootstrap-
                        # robustified AVaR, Prop 6.1 verification, and
                        # the "optimizer's curse" experiment

tests/                  # 42 tests, one file per module above

notebooks/
    01_mono_period_avar_frontier.ipynb    # Theorem 3.1 (Gaussian coincidence,
                                           # ~5% weight error) vs. divergence
                                           # under skewness (~35% weight error)
    02_discrete_time_consistency.ipynb    # Ch.4 regression check + a random
                                           # search finding genuine divergence
                                           # (absolute gap ~2.8, 37% of trees
                                           # show >10% relative divergence)
    03_continuous_time_constant_mix.ipynb # Theorem 5.1 vs. exact Monte Carlo
                                           # (max relative error < 0.25%) and
                                           # its two limiting behaviours
    04_downside_risk_comparison.ipynb     # Harlow LPM vs. AVaR vs. Markowitz,
                                           # Gaussian (all agree) vs. skewed
                                           # (all three disagree meaningfully)
    05_distributionally_robust_avar.ipynb # ORIGINAL CONTRIBUTION: Prop. 6.1
                                           # verified on the four axioms
                                           # directly, plus the optimizer's
                                           # curse experiment
```

## Key results already verified by the test suite

- **Theorem 3.1 (Gaussian coincidence)**: the LP-solved AVaR-efficient
  portfolio matches the closed-form Markowitz portfolio under Gaussian
  returns, for every confidence level tested (`test_mono_period.py`).
- **Theorem 5.1 (optimal constant-mix)**: the closed-form
  Merton-myopic-minus-correction formula matches an independent Monte
  Carlo simulation (`test_continuous_time.py`).
- **Time-consistency (Ch. 4)**: a random search over scenario trees
  finds genuine, substantial divergence between the naive and
  recursively-composed dynamic risk measures (absolute gap ≈ 3.0 found
  in one run — see `search_for_divergence` in `discrete_time.py`),
  resolving the point left open in the chapter text after several hand
  constructions kept coinciding.
- **Proposition 6.1 (sup of coherent measures is coherent)**: verified
  directly against all four axioms on a concrete family of reference
  measures (`test_robust.py`), not just proved abstractly.
- **Honest negative + positive finding on the robustification**: the
  exact bootstrap-sup version, while the only one covered by
  Proposition 6.1, **overcorrects** (larger mean absolute bias than the
  naive estimator in the optimizer's-curse experiment); a moderate
  bootstrap-quantile version (confidence 0.8), not provably coherent,
  empirically achieves the best bias reduction among the values tried.
  Both facts are asserted explicitly in `test_robust.py` — this is
  reported as an empirical finding, not oversold as a theorem.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Required references used

Bassi-Embrechts-Kafetzaki (1997) [96], Markowitz (1952) [11],
Föllmer-Schied (2002) [70], Föllmer-Schied *Stochastic Finance* [71],
Harlow (1991) [75], Pliska (1998) [109], Taflin (2000, introduction
only) [117].
