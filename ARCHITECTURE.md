# Architecture & Design Document

## System Overview

This document outlines the technical architecture of the Bayesian Marketing Mix Model framework.

## Design Principles

1. **Separation of Concerns**: Data pipeline, modeling, inference, and analysis are decoupled
2. **Configuration-Driven**: All hyperparameters centralized in a validated config dict
3. **Reproducibility**: MLflow integration ensures experiments are tracked and recoverable
4. **Testability**: Pure functions with clear I/O for unit testing
5. **Production-Ready**: Logging, error handling, and type hints throughout

## Core Components

### 1. `BayesianMMMTrainer` (src/model_trainer.py)

**Responsibility**: Orchestrates the complete MMM pipeline

**Methods**:
- `load_data()`: CSV parsing, date aggregation to weekly, holiday merging
- `preprocess()`: Feature generation (trend, Fourier), normalization (StandardScaler)
- `build_model()`: PyMC model compilation with validated config
- `train()`: MCMC sampling with MLflow logging
- `calculate_roi()`: Posterior-based ROI quantification
- `save_trace()` / `load_trace()`: Model persistence
- `run_full_analysis()`: End-to-end pipeline execution

**Key Design Decisions**:

| Decision | Rationale |
|----------|-----------|
| Weekly aggregation | Reduces noise, aligns with typical marketing cadence |
| StandardScaler normalization | Stabilizes MCMC, enables prior comparability across channels |
| Fourier seasonality | Non-parametric, flexible, handles multiple cycles |
| Shared StandardScaler for spend | Preserves relative scales across channels |
| Config validation | Catches errors early, improves debuggability |

### 2. `vectorized_geometric_adstock()` (src/adstock_functions.py)

**Responsibility**: Apply adstock decay transformation to spend arrays

**Implementation**: Vectorized PyTensor (no loops, hardware-accelerated)

**Formula**:
```
Adstock[t, k] = Σ_{lag=0}^{t} α_k^lag × Spend[t-lag, k]
```

**Complexity**: O(n² × c) where n=weeks, c=channels. Efficient for typical datasets (52-156 weeks).

**Design Choices**:
- Matrix operations instead of recursion → avoids computational overhead
- Masking for causality → future spend can't affect past periods
- Broadcasting for multi-channel → single operation instead of loop

### 3. `utils.py` - Supporting Utilities

**Functions**:
- `setup_logger()`: Structured logging with timestamp formatting
- `validate_config()`: Schema validation (required keys, type checks, bounds)
- `save_config()` / `load_config()`: JSON persistence for reproducibility
- `plot_channel_contributions()`: Matplotlib visualization of ROI
- `plot_parameter_distributions()`: β and α visualization

**Logging Strategy**:
- Replace print() with logger.info/warning/error
- Enables production monitoring, log aggregation
- Can be redirected to files, external services

### 4. `ab_testing.py` - A/B Testing Utilities

**Functions**:
- `bayesian_ab_test()`: Hierarchical Bayesian test with posterior effect estimation
- `power_analysis()`: Monte Carlo power simulation
- `sequential_testing()`: Early stopping via continuous monitoring

**Design**:
- Uses PyMC for effect estimation (prior uncertainty → posterior credible intervals)
- Not frequentist (no p-values); uses P(effect > 0) as decision statistic
- Flexible for continuous outcomes (ROI, spend, revenue lift)

## Data Pipeline

```
CSV Input
   ↓
[load_data]
   ├─ Parse date column
   ├─ Aggregate to weekly (W-MON)
   ├─ Merge holiday controls
   └─ Fill missing values
   ↓
[preprocess]
   ├─ Generate trend: t = 1, 2, ..., n_weeks
   ├─ Generate Fourier seasonality: sin/cos(2πk × day_of_year / 365.25)
   ├─ Normalize revenue → y_norm ~ N(0, 1)
   ├─ Normalize spend → x_norm ~ N(0, 1) [shared scaler]
   ├─ Scale control variables → x_ctrl_norm
   └─ Store scalers for post-hoc unscaling
   ↓
Normalized Arrays Ready for Modeling
```

## Bayesian Model Specification

```
┌─────────────────────────────────────────────────────────────┐
│                   Probabilistic Model                        │
└─────────────────────────────────────────────────────────────┘

Parameters (Random Variables):
  α_k ~ Beta(2, 8)                    [Decay rates, k=channel]
  β_k ~ HalfNormal(σ=1)               [Effectiveness, k=channel]
  intercept ~ Normal(0, 10)            [Baseline revenue]
  trend_coef ~ Normal(0, 1)            [Time trend]
  seasonality_weights_j ~ N(0, 1)     [Fourier components]
  control_coefs_i ~ Normal(0, 1)      [Control effects]
  σ ~ HalfCauchy(β=1)                 [Observation noise]

Deterministic:
  x_adstocked[t,k] = Σ_lag α_k^lag × x_spend[t-lag, k]
  media_effect[t] = Σ_k β_k × x_adstocked[t, k]
  trend_effect[t] = trend_coef × t
  seasonal_effect[t] = Σ_j seasonality_weights_j × fourier_j[t]
  control_effect[t] = Σ_i control_coefs_i × x_control_i[t]
  μ[t] = intercept + media_effect[t] + trend_effect[t] + seasonal_effect[t] + control_effect[t]

Likelihood:
  y_obs[t] ~ Normal(μ[t], σ²)         [Observed normalized revenue]
```

**Prior Justification**:

| Prior | Reasoning |
|-------|-----------|
| Alpha ~ Beta(2, 8) | Mode at 0.2, concentration toward low decay (typical for marketing) |
| Beta ~ HalfNormal | Positive by constraint; σ=1 allows 95% of mass in [0, 2.8] (flexible) |
| Intercept ~ N(0, 10) | Weakly informative; normalized revenue is ~N(0,1), so 10σ is broad |
| σ ~ HalfCauchy(β=1) | Heavy-tailed for robustness; typical Bayesian regression choice |

## Inference & Diagnostics

**Sampler**: NUTS (No-U-Turn Sampler, default PyMC)
- Hamiltonian Monte Carlo variant
- Efficient for high-dimensional posteriors
- Automatic step size tuning via dual averaging

**Convergence Diagnostics**:
- **R̂ (Potential Scale Reduction Factor)**: Target < 1.05
  - Compares within-chain vs between-chain variance
  - Values >> 1 indicate non-convergence
- **ESS (Effective Sample Size)**: Goal >> min(draws)
  - Accounts for autocorrelation in samples
  - Lower values indicate higher correlation
- **Trace plots**: Visual inspection for convergence patterns

**Posterior Predictive Checks**:
```python
ppc = pm.sample_posterior_predictive(trace)
y_pred = ppc.posterior_predictive['y_obs']  # Shape: (chains, draws, obs)
```

Validates model generates data similar to observed.

## ROI Calculation

Given posterior samples for channel k:

```
ROI_k = β_k_mean × (1 / (1 - α_k_mean)) × (σ_revenue / σ_spend_k)
      └─────┬─────┘   └──────────┬──────────┘   └──────┬──────┘
         Sensitivity    Lifetime Multiplier    Unscaling Factor
         (normalized)    (accounts for decay)   (back to original units)
```

**Credible Intervals** computed from posterior samples:
```
ROI_k_samples = β_samples_k × (1 / (1 - α_samples_k)) × unscaling
95% HDI = [percentile(ROI_k_samples, 2.5), percentile(ROI_k_samples, 97.5)]
```

## MLflow Integration

**Experiment Tracking**:
```python
mlflow.set_experiment("mmm_baseline")
with mlflow.start_run():
    mlflow.log_param("n_draws", 2000)
    mlflow.log_metric("mean_r_hat", 1.02)
    mlflow.log_artifact("roi_results.json")
```

**Artifacts Saved**:
- Model trace (NetCDF)
- Config (JSON)
- ROI results (JSON)
- Plots (PNG)

**Dashboard**:
- View experiments side-by-side
- Compare metrics across runs
- Track hyperparameter sensitivity

## Testing Strategy

**Test Categories**:

1. **Unit Tests** (test_*.py)
   - Config validation
   - Adstock formula correctness
   - Preprocessing (scaling, Fourier)
   - ROI calculation

2. **Integration Tests**
   - End-to-end pipeline with synthetic data
   - Data → preprocessing → model → inference → ROI

3. **Regression Tests**
   - Store expected outputs for synthetic scenarios
   - Catch unintended behavior changes

**Coverage Target**: 90%+

## Performance Considerations

| Operation | Time (Typical) | Notes |
|-----------|---------|-------|
| Data loading | <1s | CSV parsing, 52-156 weeks |
| Preprocessing | <1s | StandardScaler, Fourier |
| Model building | <1s | PyMC compilation |
| MCMC (2000 draws, 1000 tune) | 2-10 min | GPU can accelerate via AESMC |
| ROI calculation | <1s | Posterior aggregation |

**Scalability**:
- **Weeks**: Linear up to ~500 (typical marketing timeframe)
- **Channels**: O(n²) adstock → feasible for 10-20 channels
- **Control variables**: Scales linearly with GLM regression

## Error Handling

**Strategy**: Fail fast, provide actionable messages

```python
# Config validation
if not isinstance(config['spend_cols'], list):
    raise ValueError("'spend_cols' must be a list")

# Data integrity
if self.data_df.empty:
    raise ValueError("DataFrame empty after preprocessing")

# MCMC convergence
if (summary['r_hat'] > 1.05).any():
    logger.warning("Convergence issues detected")
```

## Future Extensions

1. **Saturation Curves**: Diminishing returns via Hill equation
   - `β_effective = β / (1 + (K/spend)^power)`
   - Captures non-linear channel response

2. **Hierarchical Modeling**: Share information across regions/brands
   - Multilevel priors → better estimates with limited data

3. **Instrumental Variables**: Causal identification via external shocks
   - E.g., TV outages, platform changes

4. **Sequential Inference**: Online learning as new data arrives
   - Update posterior incrementally

5. **Interactive Dashboard**: Streamlit or Dash for stakeholder exploration
   - Parameter sensitivity analysis
   - Scenario planning (what-if ROI)

## Dependencies & Versions

| Package | Version | Purpose |
|---------|---------|---------|
| pymc | 5.15.1 | Bayesian inference |
| arviz | 0.17.1 | Diagnostics & visualization |
| pytensor | 2.20.1 | Tensor computation backend |
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.3+ | StandardScaler |
| mlflow | 2.0+ | Experiment tracking |

## References

- [PyMC Documentation](https://docs.pymc.io/)
- [Bayesian Inference Tutorial](https://www.probabilisticprogramming.org/)
- [Adstock Decay Models in Marketing](https://en.wikipedia.org/wiki/Advertising_elasticity_of_demand)
- [MCMC Diagnostics](https://arxiv.org/pdf/1903.08008.pdf)
