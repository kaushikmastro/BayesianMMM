"""Tests for data preprocessing pipeline."""
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from src.model_trainer import BayesianMMMTrainer


def create_test_data(n_weeks=52, n_channels=3):
    """Create synthetic test data."""
    dates = pd.date_range('2023-01-01', periods=n_weeks, freq='W-MON')
    data = {
        'date': dates,
        'revenue': np.random.uniform(1000, 5000, n_weeks),
    }
    for i in range(n_channels):
        data[f'channel_{i}'] = np.random.uniform(100, 500, n_weeks)

    return pd.DataFrame(data)


def test_load_data():
    """Data loading should parse dates and aggregate to weekly."""
    df = create_test_data(n_weeks=52)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        config = {
            'date_col': 'date',
            'spend_cols': ['channel_0', 'channel_1', 'channel_2'],
            'revenue_col': 'revenue',
            'fourier_k': 2,
            'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
        }

        trainer = BayesianMMMTrainer(config, f.name)
        loaded_df = trainer.load_data()

        assert loaded_df is not None
        assert len(loaded_df) > 0
        assert 'revenue' in loaded_df.columns
        Path(f.name).unlink()


def test_preprocess_scaling():
    """Preprocessing should normalize spend and revenue."""
    df = create_test_data(n_weeks=52)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        config = {
            'date_col': 'date',
            'spend_cols': ['channel_0', 'channel_1', 'channel_2'],
            'revenue_col': 'revenue',
            'fourier_k': 2,
            'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
        }

        trainer = BayesianMMMTrainer(config, f.name)
        trainer.load_data()
        x_spend, y_revenue, _, _, _ = trainer.preprocess()

        # Check shapes
        assert x_spend.shape[0] == y_revenue.shape[0]
        assert x_spend.shape[1] == 3  # 3 channels

        # Check normalization: mean ≈ 0, std ≈ 1
        assert np.abs(y_revenue.mean()) < 0.1
        assert np.abs(y_revenue.std() - 1.0) < 0.1

        Path(f.name).unlink()


def test_preprocess_fourier_features():
    """Preprocessing should generate Fourier seasonality features."""
    df = create_test_data(n_weeks=52)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        config = {
            'date_col': 'date',
            'spend_cols': ['channel_0'],
            'revenue_col': 'revenue',
            'fourier_k': 2,
            'mcmc_params': {'draws': 100, 'tune': 100, 'target_accept': 0.8}
        }

        trainer = BayesianMMMTrainer(config, f.name)
        trainer.load_data()
        _, _, x_seasonality, x_trend, _ = trainer.preprocess()

        # fourier_k=2 should produce 2*2=4 features (sin/cos pairs)
        assert x_seasonality.shape[1] == 4
        # Should have same number of rows as data
        assert x_seasonality.shape[0] == len(trainer.data_df)

        # Trend should be monotonically increasing
        assert np.all(np.diff(x_trend.flatten()) > 0)

        Path(f.name).unlink()
