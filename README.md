# Online Learning for Correlated Data Streams under Concept Drift

The repository compares the classical Affine Projection Algorithm (APA) with an adaptive Jackson-regularized variant (aq-JAPA) on controlled, highly correlated and spatially heteroscedastic AR(1) streams.

The framework specifically models a "favorable regime" for q-calculus adaptive filters: tracking non-stationary systems subject to abrupt shrinkage concept drifts, where the input data streams are highly correlated and possess heteroscedastic spatial energies.

## Overview

Standard adaptive filtering algorithms suffer from performance degradation when the input signal's autocorrelation matrix is ill-conditioned (e.g., highly correlated AR(1) processes). When an abrupt **shrinkage drift** occurs on high-energy coordinates, classical algorithms struggle to forget the old state, resulting in a slow recovery time ($\tau_{rec}$).

This project introduces a modified **aq-JAPA** that detects post-drift error spikes and triggers a temporal $q_t$ parameter. This dynamic deformation temporarily penalizes the adaptive weights, accelerating the recovery time and improving the post-drift Mean Square Deviation (MSD) without sacrificing steady-state performance.

## The Algorithm: aq-JAPA with Error Trigger

Unlike traditional q-calculus algorithms that rely solely on the input condition number, this implementation of aq-JAPA activates the q-deformation factor using a post-drift error spike decay mechanism.

1. **Error Tracking:** Maintains an exponentially weighted moving average of the prediction error energy $$\bar{E}_t = \beta \bar{E}_{t-1} + (1-\beta) |e_t|^2$$.
2. **Spike Detection:** Computes a positive activation factor $\psi_t$ upon detecting a sudden increase above the pre-drift baseline, followed by a temporal exponential decay ($L=80$).
3. **Adaptive Parameter:** Updates the deformation parameter as $q_t = 1 + \alpha \psi_t$.
4. **Weight Update Rule:** Applies a transient shrinkage penalty proportional to the input energy matrix $\mathbf{D}_t$:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \mu \mathbf{X}_t (\mathbf{X}_t^T \mathbf{X}_t + \epsilon \mathbf{I})^{-1} \mathbf{e}_t - \gamma_t \mathbf{D}_t \mathbf{w}_t$$

Where $\gamma_t = \frac{\mu}{2}(q_t - 1)$ and $\mathbf{D}_t = \text{diag}(\mathbf{X}_t \mathbf{X}_t^T)$.

## Experimental Setup (The Favorable Regime)

The simulation script builds a heavily ill-conditioned environment to test the limits of the algorithms:
* **Input Data:** $M=10$ independent AR(1) processes with extreme temporal correlation ($\rho = 0.99$).
* **Heteroscedasticity:** Spatial variances are distributed exponentially ($\sigma^2_{min} = 0.1$, ratio $= 10^3$).
* **Abrupt Shrinkage Drift:** At iteration $T/2$, the true weights $\mathbf{w}^*$ collapse on the highest-energy coordinate by a shrinkage factor $c$ (where $0 < c < 1$).
* **Input Normalization:** Features are globally normalized to bounded energy limits.

## How to Run


## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For tests:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```


### Dependencies
Ensure you have the following Python packages installed:
* `numpy`
* `matplotlib`

### Execution
Run the main script to execute the Monte Carlo simulations and generate the comparative figures:

```bash
python aq_japa_simulation.py
