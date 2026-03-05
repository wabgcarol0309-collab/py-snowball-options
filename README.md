# py-snowball-options
A pure Python Monte Carlo pricing engine for Snowball options, featuring Greeks calculation and Delta hedging backtest.
# ❄️ Quant Snowball Options Pricing Engine

A high-performance, object-oriented Monte Carlo pricing engine for Snowball options (Autocallables) written in pure Python. 

This project demonstrates quantitative modeling capabilities, focusing on vectorized computations, extensible architecture (Strategy Pattern), and advanced stochastic dynamics.

## ✨ Core Features (Current Progress)

* **Fully Vectorized Monte Carlo**: Eliminates slow Python `for` loops along the time axis by leveraging NumPy's matrix operations.
* **Model-Agnostic Engine**: Designed with OOP principles (Abstract Base Classes) to easily decouple the pricing engine from underlying market dynamics.
* **Multiple Stochastic Models**:
    * Standard Geometric Brownian Motion (GBM).
    * **Heston Stochastic Volatility Model**: Implemented using the Euler-Maruyama Full Truncation scheme to handle negative variance instabilities and capture volatility skew.
* **Path-Dependent Payoffs**: Matrix-level logic gating for complex Knock-in (KI) and Knock-out (KO) discrete barrier observations.

## 🧮 Mathematical Framework

### 1. The Heston Model Dynamics
Under the risk-neutral measure $\mathbb{Q}$, the underlying asset $S_t$ and its variance $v_t$ are simulated as:

$$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S$$
$$dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_t^v$$

where $dW_t^S dW_t^v = \rho dt$. The negative correlation $\rho$ is crucial for capturing the asymmetric volatility smile typical in equity derivatives.

## 🏗️ Project Architecture

```text

py-snowball-options/
├── core/
│   ├── models.py      # Abstract BaseModel, GBM, Heston, and LocalVol implementations
│   ├── sde.py         # Stochastic Differential Equation solvers (Euler-Maruyama)
│   ├── payoff.py      # Logic-gated vectorized payoff for KI/KO events
│   └── engine.py      # Model-agnostic Monte Carlo Pricing Engine
├── utils/
│   ├── data_loader.py # AKShare wrappers for live Indices & Risk-free rates
│   └── greeks.py      # Finite Difference Greek calculators
├── images/            # Visualizations (Gamma surface, MC paths)
├── main_pricing.py    # Multi-model benchmarking script (Main Entry)
└── plot_risk_surface.py # 3D Risk Profiling script
```
## 🚀 快速开始
**克隆项目并进入目录**:
   ```bash
   git clone [https://github.com/你的用户名/py-snowball-options.git](https://github.com/你的用户名/py-snowball-options.git)
   cd py-snowball-options


