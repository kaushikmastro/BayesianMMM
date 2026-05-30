"""Quick start example: End-to-end Bayesian MMM pipeline.

This script demonstrates the complete workflow:
1. Load and preprocess marketing data
2. Build Bayesian model with adstock effects
3. Run MCMC inference
4. Quantify channel ROI with uncertainty
5. Track experiments with MLflow
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import BayesianMMMTrainer, setup_logger

logger = setup_logger(__name__)


def main():
    """Run complete MMM pipeline."""

    # Configuration
    config = {
        'date_col': 'date',
        'spend_cols': ['tv_spend', 'radio_spend', 'online_spend'],
        'revenue_col': 'revenue',
        'control_cols': ['is_holiday'],
        'fourier_k': 2,
        'mcmc_params': {
            'draws': 2000,
            'tune': 1000,
            'target_accept': 0.8
        }
    }

    # Data paths (assuming you have synthetic data)
    data_path = Path(__file__).parent.parent / 'data' / 'dt_simulated_weekly.csv'
    holidays_path = Path(__file__).parent.parent / 'data' / 'dt_prophet_holidays.csv'

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please ensure data files exist in data/ directory")
        return

    # Initialize trainer
    logger.info("Initializing BayesianMMMTrainer...")
    trainer = BayesianMMMTrainer(config, str(data_path), str(holidays_path))

    # Run full pipeline with MLflow tracking
    logger.info("Running full MMM pipeline...")
    roi_results = trainer.run_full_analysis(experiment_name="mmm_baseline_run")

    # Display results
    logger.info("\n" + "="*60)
    logger.info("ROI RESULTS BY CHANNEL")
    logger.info("="*60)

    for channel, metrics in roi_results.items():
        logger.info(
            f"{channel:20s} | "
            f"ROI: {metrics['unscaled_roi']:8.2f} | "
            f"Beta: {metrics['mean_beta']:6.3f} | "
            f"Alpha: {metrics['mean_alpha']:6.3f}"
        )

    # Save model trace for future analysis
    trace_path = Path(__file__).parent.parent / 'models' / 'mmm_trace.nc'
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_trace(str(trace_path))
    logger.info(f"Model trace saved to {trace_path}")

    return roi_results


if __name__ == "__main__":
    roi_results = main()
    print("\n✓ Pipeline completed successfully!")
