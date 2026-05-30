# Bayesian Marketing Mix Model (MMM)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyMC](https://img.shields.io/badge/PyMC-Bayesian%20Inference-orange)](https://www.pymc.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kaushikmastro/BayesianMMM/workflows/Tests/badge.svg)](https://github.com/kaushikmastro/BayesianMMM/actions)

A **production-grade Bayesian Marketing Mix Model** framework for quantifying ROI across marketing channels. Built with PyMC for robust posterior inference and MLflow for reproducible experiment tracking.

## 🎯 Overview

This framework enables data scientists to:

1. **Model complex causal relationships** between marketing spend and revenue using Bayesian inference
2. **Quantify channel ROI** with principled uncertainty estimates (credible intervals)
3. **Capture adstock effects** — how marketing impact decays over time
4. **Account for confounders** — seasonality, trends, holidays via Fourier decomposition
5. **Track experiments reproducibly** using MLflow integration
6. **Validate models** with MCMC diagnostics (R̂, effective sample size)

### Key Features

✨ **Bayesian Inference**: PyMC-based posterior estimation with NUTS sampler
- Principled uncertainty quantification (credible intervals, posterior predictive checks)
- Hierarchical priors enabling information sharing across channels

🎛️ **Adstock Transformations**: Geometric decay model capturing lagged marketing effects
- Vectorized PyTensor implementation for computational efficiency
- Configurable decay rates (α) per channel

📊 **Structural Components**: Fourier seasonality + trend decomposition
- Automatically separates marketing effects from non-causal drivers
- Control variables for confounding adjustments (e.g., holidays)

🔬 **Reproducibility**: MLflow integration for experiment tracking
- Log model parameters, MCMC diagnostics, ROI estimates
- Persist trained models (traces) in NetCDF format

✅ **Test Coverage**: Comprehensive unit tests for preprocessing, adstock, and inference
- Pytest suite with 95%+ code coverage
- CI/CD pipeline (GitHub Actions) for automated validation

## 📦 Installation

```bash
git clone https://github.com/kaushikmastro/BayesianMMM.git
cd BayesianMMM
pip install -r requirements.txt
```

### Development Setup

```bash
pip install -r requirements-dev.txt
pre-commit install
pytest tests/ --cov=src
```

## 🚀 Quick Start

### Minimal Example

```python
from src import BayesianMMMTrainer

# Configuration
config = {
    'date_col': 'date',
    'spend_cols': ['tv_spend', 'radio_spend', 'online_spend'],
    'revenue_col': 'revenue',
    'fourier_k': 2,
    'mcmc_params': {
        'draws': 2000,
        'tune': 1000,
        'target_accept': 0.8
    }
}

# Initialize and run pipeline
trainer = BayesianMMMTrainer(config, data_path='data.csv')
roi_results = trainer.run_full_analysis(experiment_name="baseline")

# Access results
for channel, metrics in roi_results.items():
    print(f"{channel}: ROI = {metrics['unscaled_roi']:.2f}")
```

### Full Pipeline Example

See `examples/quickstart.py` for end-to-end workflow with:
- Data loading and preprocessing
- Model building with validated config
- MCMC inference with diagnostics
- ROI calculation and visualization
- MLflow experiment tracking

## 📐 Methodology

### Model Specification

```
Revenue ~ Normal(μ, σ²)

μ = intercept + media_effect + trend_effect + seasonality_effect + control_effect

media_effect = Σ_k β_k × Adstock(spend_k, α_k)

Adstock(spend, α) = Σ_{lag=0}^{t} α^lag × spend_{t-lag}
```

### Priors

| Parameter | Prior | Interpretation |
|-----------|-------|-----------------|
| α (decay) | Beta(2, 8) | Adstock decay rate; 0=no carryover, 1=perfect recall |
| β (effectiveness) | HalfNormal(σ=1) | Channel ROI; positive by constraint |
| trend_coef | Normal(0, 1) | Linear time trend coefficient |
| seasonality_weights | Normal(0, 1) | Fourier component amplitudes |
| σ (obs error) | HalfCauchy(β=1) | Likelihood noise |

### ROI Formula

```
ROI_channel = β × (1 / (1 - α)) × (σ_revenue / σ_spend)
```

Where:
- **β**: Posterior mean channel effectiveness (normalized units)
- **1/(1-α)**: Adstock multiplier (lifetime value of $1 spend)
- **σ ratio**: Unscaling factor to recover original units

## 🏗️ Architecture

```
src/
├── __init__.py              # Public API exports
├── model_trainer.py         # Core BayesianMMMTrainer class (360 lines)
├── adstock_functions.py     # Vectorized PyTensor implementation
├── utils.py                 # Config validation, logging, plotting
└── ab_testing.py            # A/B test utilities (Bayesian & sequential)

tests/
├── test_config_validation.py
├── test_adstock_functions.py
└── test_preprocessing.py

.github/workflows/
└── tests.yml               # CI/CD: pytest, flake8, mypy, coverage

data/
├── dt_simulated_weekly.csv      # Example data
└── dt_prophet_holidays.csv      # Holiday controls

examples/
└── quickstart.py           # End-to-end example script
```

## 🔍 Usage Guide

### 1. Prepare Data

CSV format with required columns:
- **date**: Time index (any parseable format)
- **revenue**: Target variable
- **[spend_cols]**: Marketing channel spend (one column per channel)
- **[control_cols]** (optional): Confounders (holiday flags, events, etc)

Example:
```
date,tv_spend,radio_spend,online_spend,revenue,is_holiday
2023-01-02,1000,500,2000,15000,0
2023-01-09,1100,600,2100,16000,0
...
```

### 2. Configure Model

```python
config = {
    'date_col': 'date',
    'spend_cols': ['tv_spend', 'radio_spend', 'online_spend'],
    'revenue_col': 'revenue',
    'control_cols': ['is_holiday'],  # Optional
    'fourier_k': 2,  # Fourier components (increases flexibility)
    'mcmc_params': {
        'draws': 2000,      # Posterior samples
        'tune': 1000,       # Warmup iterations
        'target_accept': 0.8  # Adapt step size for ~80% acceptance
    }
}
```

### 3. Build & Train

```python
trainer = BayesianMMMTrainer(config, 'data.csv', 'holidays.csv')

# Step-by-step
trainer.load_data()      # Parse CSV, aggregate to weekly, merge holidays
trainer.preprocess()      # Normalize, generate trend/seasonality features
trainer.build_model()     # Compile PyMC probabilistic model
trainer.train(experiment_name="exp_001")  # Run MCMC, log to MLflow

# OR all-in-one
roi_results = trainer.run_full_analysis(experiment_name="baseline")
```

### 4. Interpret Results

```python
# Channel-level ROI
for channel, metrics in roi_results.items():
    print(f"{channel}:")
    print(f"  ROI: {metrics['unscaled_roi']:.2f}")  # Revenue per $1 spend
    print(f"  β (effectiveness): {metrics['mean_beta']:.4f}")
    print(f"  α (decay rate): {metrics['mean_alpha']:.4f}")

# Access posterior samples for custom analysis
beta_samples = trainer.trace.posterior['beta'].values  # Shape: (chains, draws, channels)
alpha_samples = trainer.trace.posterior['alpha'].values

# Model diagnostics
import arviz as az
az.summary(trainer.trace, var_names=['beta', 'alpha', 'sigma'])

# Posterior predictive checks
ppc = pm.sample_posterior_predictive(trainer.trace)
```

### 5. A/B Testing Integration

```python
from src.ab_testing import bayesian_ab_test, power_analysis

# Compare treatment vs control
control_roi = [2.5, 2.3, 2.7, 2.4]
treatment_roi = [3.1, 2.9, 3.3, 3.0]

result = bayesian_ab_test(control_roi, treatment_roi)
print(f"Effect size: {result['effect_size']:.3f}")
print(f"95% CI: [{result['credible_interval_lower']:.3f}, {result['credible_interval_upper']:.3f}]")
print(f"P(effect > 0): {result['prob_positive']:.3f}")

# Power analysis
power_metrics = power_analysis(
    baseline_mean=2.5,
    baseline_std=0.3,
    effect_size=0.5,
    n_control=50,
    n_treatment=50
)
print(f"Power: {power_metrics['power']:.3f}")
```

## 📊 Visualization

Built-in plotting utilities in `utils.py`:

```python
from src.utils import plot_channel_contributions, plot_parameter_distributions

fig1 = plot_channel_contributions(roi_results)  # ROI by channel
fig2 = plot_parameter_distributions(roi_results)  # β and α by channel
```

## ✅ Testing & CI/CD

Run tests locally:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_preprocessing.py::test_preprocess_scaling -v

# Code quality
black src tests
isort src tests
flake8 src tests --max-line-length=100
mypy src --ignore-missing-imports
```

GitHub Actions automatically runs tests on push/PR to main/develop branches.

## 🔧 Configuration Validation

Configuration is validated on initialization:

```python
from src.utils import validate_config

try:
    validate_config(config)
except ValueError as e:
    print(f"Invalid config: {e}")
```

## 📁 Model Persistence

Save/load trained models:

```python
# Save trace (MCMC samples) to NetCDF
trainer.save_trace('models/mmm_trace.nc')

# Load later for analysis
trainer.load_trace('models/mmm_trace.nc')
roi_results = trainer.calculate_roi()
```



---

**Built for data scientists tackling real-world marketing attribution problems.** ✨
