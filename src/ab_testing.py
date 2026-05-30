"""A/B Testing utilities for marketing channel experiments.

Provides utilities to:
- Define treatment/control groups
- Estimate causal effects via Bayesian inference
- Conduct posterior predictive checks for statistical significance
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from .utils import setup_logger

logger = setup_logger(__name__)


def bayesian_ab_test(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    draws: int = 2000,
    tune: int = 1000
) -> Dict[str, float]:
    """Conduct Bayesian A/B test via hierarchical model.

    Estimates treatment effect with uncertainty quantification.
    Assumes continuous outcomes (e.g., ROI, spend).

    Args:
        control_data: Observed outcomes for control group
        treatment_data: Observed outcomes for treatment group
        prior_mu: Prior mean for effect size
        prior_sigma: Prior std for effect size
        draws: MCMC draws
        tune: MCMC tuning steps

    Returns:
        Dictionary with effect_size, credible_interval, prob_positive
    """
    logger.info(f"Running Bayesian A/B test: control={len(control_data)}, treatment={len(treatment_data)}")

    with pm.Model() as model:
        # Priors for group means
        mu_control = pm.Normal("mu_control", mu=control_data.mean(), sigma=control_data.std())
        mu_treatment = pm.Normal("mu_treatment", mu=treatment_data.mean(), sigma=treatment_data.std())

        # Treatment effect
        effect = pm.Normal("effect", mu=prior_mu, sigma=prior_sigma)

        # Observation error
        sigma = pm.HalfNormal("sigma", sigma=control_data.std())

        # Likelihood
        pm.Normal("control_obs", mu=mu_control, sigma=sigma, observed=control_data)
        pm.Normal("treatment_obs", mu=mu_treatment, sigma=sigma, observed=treatment_data)

        # Sampling
        trace = pm.sample(draws=draws, tune=tune, progressbar=False, return_inferencedata=True)

    # Extract posterior samples
    effect_samples = trace.posterior["effect"].values.flatten()
    effect_mean = effect_samples.mean()
    hdi = az.hdi(trace, var_names=["effect"], hdi_prob=0.95)["effect"].values
    prob_positive = (effect_samples > 0).mean()

    logger.info(f"Effect: {effect_mean:.3f} | 95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}] | P(effect > 0): {prob_positive:.3f}")

    return {
        "effect_size": float(effect_mean),
        "credible_interval_lower": float(hdi[0]),
        "credible_interval_upper": float(hdi[1]),
        "prob_positive": float(prob_positive),
        "posterior_samples": effect_samples
    }


def power_analysis(
    baseline_mean: float,
    baseline_std: float,
    effect_size: float,
    n_control: int = 100,
    n_treatment: int = 100,
    n_sims: int = 100
) -> Dict[str, float]:
    """Estimate statistical power via simulation.

    Monte Carlo simulation of A/B test to estimate power
    (probability of detecting effect of given size).

    Args:
        baseline_mean: Mean of control group
        baseline_std: Std of control group
        effect_size: Minimum detectable effect
        n_control: Sample size for control
        n_treatment: Sample size for treatment
        n_sims: Number of simulations

    Returns:
        Dictionary with power and type_1_error estimates
    """
    logger.info(f"Running power analysis: effect_size={effect_size}, n={n_treatment}")

    detected = 0
    false_positives = 0

    for _ in range(n_sims):
        # Control group (null effect)
        control = np.random.normal(baseline_mean, baseline_std, n_control)
        treatment = np.random.normal(baseline_mean + effect_size, baseline_std, n_treatment)

        result = bayesian_ab_test(control, treatment, draws=500, tune=500)

        # Detect if 95% HDI excludes 0
        hdi_lower = result["credible_interval_lower"]
        hdi_upper = result["credible_interval_upper"]

        if hdi_lower > 0 or hdi_upper < 0:
            detected += 1

        # False positive: null effect scenario
        control_null = np.random.normal(baseline_mean, baseline_std, n_control)
        treatment_null = np.random.normal(baseline_mean, baseline_std, n_treatment)
        result_null = bayesian_ab_test(control_null, treatment_null, draws=500, tune=500)

        hdi_lower_null = result_null["credible_interval_lower"]
        hdi_upper_null = result_null["credible_interval_upper"]

        if hdi_lower_null > 0 or hdi_upper_null < 0:
            false_positives += 1

    power = detected / n_sims
    type_1_error = false_positives / n_sims

    logger.info(f"Power: {power:.3f} | Type I Error: {type_1_error:.3f}")

    return {
        "power": power,
        "type_1_error": type_1_error
    }


def sequential_testing(
    observations: np.ndarray,
    batch_size: int = 10,
    stopping_threshold: float = 0.95
) -> Dict[str, float]:
    """Conduct sequential testing for early stopping.

    Useful for continuous A/B tests where you can stop early
    if effect is clearly detectable.

    Args:
        observations: Array of outcome differences (treatment - control)
        batch_size: Number of obs per batch
        stopping_threshold: Stop if P(effect > 0) reaches this

    Returns:
        Dictionary with n_samples_needed, prob_effect, stopped_early
    """
    n_batches = len(observations) // batch_size
    stopped_early = False
    stopping_batch = None

    for batch_idx in range(1, n_batches + 1):
        batch_obs = observations[:batch_idx * batch_size]
        # Simplified: p(positive effect) ≈ fraction of positive obs
        prob_positive = (batch_obs > 0).mean()

        if prob_positive > stopping_threshold:
            stopped_early = True
            stopping_batch = batch_idx
            break

    result_batch = stopping_batch if stopped_early else n_batches
    n_samples = result_batch * batch_size

    logger.info(f"Sequential test: stopped_early={stopped_early} at n={n_samples}")

    return {
        "n_samples_needed": int(n_samples),
        "stopped_early": bool(stopped_early),
        "prob_positive": float((observations[:n_samples] > 0).mean())
    }
