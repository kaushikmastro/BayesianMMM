"""Tests for data preprocessing pipeline."""
import numpy as np
import pandas as pd
from src.model_trainer import BayesianMMMTrainer


def create_test_config():
    """Create a minimal test configuration."""
    return {
        'date_col': 'date',
        'spend_cols': ['channel_0', 'channel_1', 'channel_2'],
        'revenue_col': 'revenue',
        'fourier_k': 2,
        'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
    }


def create_test_dataframe(n_weeks=52):
    """Create synthetic test data without file I/O."""
    dates = pd.date_range('2023-01-01', periods=n_weeks, freq='W-MON')
    data = {
        'date': dates,
        'revenue': np.random.uniform(1000, 5000, n_weeks),
        'channel_0': np.random.uniform(100, 500, n_weeks),
        'channel_1': np.random.uniform(100, 500, n_weeks),
        'channel_2': np.random.uniform(100, 500, n_weeks),
    }
    return pd.DataFrame(data)


def test_config_creation():
    """Test that config can be created."""
    config = create_test_config()
    assert config['fourier_k'] == 2
    assert len(config['spend_cols']) == 3


def test_dataframe_creation():
    """Test that synthetic data can be created."""
    df = create_test_dataframe(52)
    assert len(df) == 52
    assert 'revenue' in df.columns
    assert len(df.columns) == 5  # date + revenue + 3 channels


def test_trainer_initialization():
    """Test trainer can be initialized with valid config."""
    config = create_test_config()
    df = create_test_dataframe()

    # Save to temp location
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, 'test_data.csv')
        df.to_csv(data_file, index=False)

        # Create trainer
        trainer = BayesianMMMTrainer(config, data_file)
        assert trainer.data_path == data_file
        assert trainer.config == config


def test_preprocessing_shapes():
    """Test preprocessing produces correct shapes."""
    config = create_test_config()
    df = create_test_dataframe(52)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, 'test_data.csv')
        df.to_csv(data_file, index=False)

        trainer = BayesianMMMTrainer(config, data_file)
        trainer.load_data()
        x_spend, y_revenue, x_seasonality, x_trend, x_controls = trainer.preprocess()

        # Verify shapes
        assert x_spend.shape[0] == y_revenue.shape[0]  # Same number of rows
        assert x_spend.shape[1] == 3  # 3 channels
        assert x_trend.shape[0] == y_revenue.shape[0]  # Trend matches data length
        assert x_seasonality.shape[0] == y_revenue.shape[0]  # Seasonality matches data


def test_preprocessing_normalization():
    """Test that normalization produces ~N(0,1) data."""
    config = create_test_config()
    df = create_test_dataframe(52)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, 'test_data.csv')
        df.to_csv(data_file, index=False)

        trainer = BayesianMMMTrainer(config, data_file)
        trainer.load_data()
        x_spend, y_revenue, _, _, _ = trainer.preprocess()

        # Check normalization: mean ≈ 0, std ≈ 1
        assert np.abs(y_revenue.mean()) < 0.1, f"Revenue mean {y_revenue.mean()} not close to 0"
        assert np.abs(y_revenue.std() - 1.0) < 0.1, f"Revenue std {y_revenue.std()} not close to 1"
