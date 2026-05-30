"""Tests for configuration validation."""
import pytest
from src.utils import validate_config


def test_validate_config_good():
    """Valid config should pass."""
    config = {
        'date_col': 'date',
        'spend_cols': ['a', 'b'],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
    }
    validate_config(config)  # Should not raise


def test_validate_config_missing_keys():
    """Missing required keys should raise."""
    config = {'date_col': 'date'}
    with pytest.raises(ValueError):
        validate_config(config)


def test_validate_config_empty_spend():
    """Empty spend_cols should raise."""
    config = {
        'date_col': 'date',
        'spend_cols': [],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
    }
    with pytest.raises(ValueError):
        validate_config(config)


def test_validate_config_bad_mcmc():
    """Invalid MCMC params should raise."""
    config = {
        'date_col': 'date',
        'spend_cols': ['a'],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 50, 'tune': 100, 'target_accept': 0.8}
    }
    with pytest.raises(ValueError):
        validate_config(config)
