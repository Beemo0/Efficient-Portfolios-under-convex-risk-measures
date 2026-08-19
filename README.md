# Efficient Portfolios under Convex Risk Measures

**ING3 CY Tech 2025–2026 | M222 - Dauphine 2026-2027**

**Corentin Stephan**

---

## Abstract

The classical mean-variance approach to portfolio choice treats variance as a synonym for risk, an identification that fails on two counts: variance penalizes upside and downside deviations identically, and it is not monotone. This project develops the axiomatic theory of convex and coherent risk measures to address these limitations. It formulates and solves the efficient portfolio problem under convex risk measures across mono-period, discrete-time multi-period, and continuous-time settings. Numerical simulations validate the theoretical results, including a nested-bootstrap correction for estimation risk in finite samples.

**Keywords:** Convex risk measures, coherent risk measures, Average Value-at-Risk (AVaR), Value-at-Risk (VaR), efficient portfolios, Markowitz, time-consistency, dynamic risk measures, distributionally robust optimization.

---

## Overview

This repository implements the theoretical and numerical pipeline for portfolio optimization under convex risk measures. The project is divided into two parts:

1. **Theoretical Foundations:**
   - Axiomatic theory of convex and coherent risk measures.
   - Robust representation theorem for convex risk measures.
   - Proof of Value-at-Risk’s failure of subadditivity.
   - Adoption of Average Value-at-Risk (AVaR) as a coherent and numerically tractable risk measure.

2. **Numerical Implementation:**
   - Mono-period AVaR-efficient portfolios.
   - Discrete-time multi-period portfolios with time-consistency.
   - Continuous-time portfolios with closed-form solutions for constant-mix strategies.
   - Nested-bootstrap correction for estimation risk.

---

## Repository Structure
