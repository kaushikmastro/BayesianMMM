"""Tests for configuration validation."""
import pytest
from src.utils import validate_config


def test_validate_config_valid():
    """Valid configuration should pass."""
    config = {
        'date_col': 'date',
        'spend_cols': ['tv', 'radio'],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
    }
    validate_config(config)


def test_validate_config_missing_keys():
    """Missing required keys should raise ValueError."""
    config = {'date_col': 'date', 'spend_cols': ['tv']}
    with pytest.raises(ValueError, match="missing required keys"):
        validate_config(config)


def test_validate_config_empty_spend_cols():
    """Empty spend_cols should raise ValueError."""
    config = {
        'date_col': 'date',
        'spend_cols': [],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
    }
    with pytest.raises(ValueError, match="non-empty list"):
        validate_config(config)


def test_validate_config_invalid_mcmc_params():
    """Invalid MCMC parameters should raise ValueError."""
    config = {
        'date_col': 'date',
        'spend_cols': ['tv'],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 50, 'tune': 100, 'target_accept': 0.8}  # draws too small
    }
    with pytest.raises(ValueError, match="draws.*tune.*>= 100"):
        validate_config(config)


def test_validate_config_invalid_target_accept():
    """Invalid target_accept should raise ValueError."""
    config = {
        'date_col': 'date',
        'spend_cols': ['tv'],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 1.5}
    }
    with pytest.raises(ValueError, match="target_accept"):
        validate_config(config)
