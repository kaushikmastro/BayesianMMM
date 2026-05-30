import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def validate_config(config: Dict[str, Any]) -> None:
    """Validate MMM configuration dictionary."""
    required_keys = ['date_col', 'spend_cols', 'revenue_col', 'fourier_k', 'mcmc_params']
    missing_keys = [k for k in required_keys if k not in config]

    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")

    if not isinstance(config['spend_cols'], list) or not config['spend_cols']:
        raise ValueError("'spend_cols' must be a non-empty list")

    mcmc_params = config['mcmc_params']
    mcmc_required = ['draws', 'tune', 'target_accept']
    mcmc_missing = [k for k in mcmc_required if k not in mcmc_params]

    if mcmc_missing:
        raise ValueError(f"'mcmc_params' missing required keys: {mcmc_missing}")

    if mcmc_params['draws'] < 100 or mcmc_params['tune'] < 100:
        raise ValueError("MCMC 'draws' and 'tune' must be >= 100")

    if not (0 < mcmc_params['target_accept'] < 1):
        raise ValueError("'target_accept' must be between 0 and 1")


def save_config(config: Dict[str, Any], filepath: Path) -> None:
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_config(filepath: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_data(X: np.ndarray, scaler: Optional[StandardScaler] = None) -> tuple:
    """Normalize data using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
    else:
        X_norm = scaler.transform(X)

    return X_norm, scaler


def plot_channel_contributions(
    roi_results: Dict[str, Dict[str, float]],
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """Visualize ROI contributions by marketing channel."""
    channels = list(roi_results.keys())
    roi_values = [roi_results[ch]['unscaled_roi'] for ch in channels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(channels, roi_values, color='steelblue', alpha=0.8)
    ax.set_ylabel('ROI', fontsize=12)
    ax.set_title('Marketing Channel ROI Contributions', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    return fig


def plot_parameter_distributions(
    roi_results: Dict[str, Dict[str, float]],
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """Visualize beta (effectiveness) and alpha (decay) parameters by channel."""
    channels = list(roi_results.keys())
    betas = [roi_results[ch]['mean_beta'] for ch in channels]
    alphas = [roi_results[ch]['mean_alpha'] for ch in channels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(channels, betas, color='coral', alpha=0.8)
    ax1.set_ylabel('Beta (Channel Effectiveness)', fontsize=11)
    ax1.set_title('Marketing Channel Effectiveness', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(channels, alphas, color='teal', alpha=0.8)
    ax2.set_ylabel('Alpha (Adstock Decay Rate)', fontsize=11)
    ax2.set_title('Adstock Decay Rates by Channel', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig
