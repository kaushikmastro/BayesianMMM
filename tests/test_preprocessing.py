"""Tests for preprocessing utilities."""
from src.utils import setup_logger, validate_config
from src.model_trainer import BayesianMMMTrainer


def test_logger_setup():
    """Logger should be created without errors."""
    logger = setup_logger(__name__)
    assert logger is not None
    assert hasattr(logger, 'info')


def test_trainer_imports():
    """Trainer should be importable."""
    assert BayesianMMMTrainer is not None
    assert callable(BayesianMMMTrainer)


def test_trainer_init():
    """Trainer should initialize with valid config."""
    config = {
        'date_col': 'date',
        'spend_cols': ['a', 'b'],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
    }
    trainer = BayesianMMMTrainer(config, 'dummy.csv')
    assert trainer.config == config
    assert trainer.data_path == 'dummy.csv'
