"""Bayesian Marketing Mix Model (MMM) framework for ROI quantification.

A production-grade implementation of Bayesian MMM using PyMC for posterior inference
and MLflow for experiment tracking. Enables causal estimation of marketing channel
effectiveness through adstock transformations and Bayesian hierarchical modeling.

Main Components:
    - BayesianMMMTrainer: Orchestrates complete MMM pipeline
    - adstock_functions: Geometric adstock transformation (vectorized)
    - utils: Configuration validation, logging, visualization

Example:
    >>> config = {
    ...     'date_col': 'date',
    ...     'spend_cols': ['tv', 'radio', 'online'],
    ...     'revenue_col': 'revenue',
    ...     'fourier_k': 2,
    ...     'mcmc_params': {'draws': 2000, 'tune': 1000, 'target_accept': 0.8}
    ... }
    >>> trainer = BayesianMMMTrainer(config, 'data.csv')
    >>> roi_results = trainer.run_full_analysis()
"""

from .model_trainer import BayesianMMMTrainer
from .adstock_functions import vectorized_geometric_adstock
from .utils import setup_logger, validate_config, save_config, load_config

__version__ = "1.0.0"
__author__ = "Kaushik Mukherjee"

__all__ = [
    "BayesianMMMTrainer",
    "vectorized_geometric_adstock",
    "setup_logger",
    "validate_config",
    "save_config",
    "load_config",
]
