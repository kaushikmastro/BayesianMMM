"""Tests for adstock transformation."""
import numpy as np
import pytensor.tensor as pt
from src.adstock_functions import vectorized_geometric_adstock


def test_adstock_zero_decay():
    """With alpha=0, current spend fully maps to current period."""
    n_weeks = 10
    n_channels = 2
    x = pt.as_tensor(np.ones((n_weeks, n_channels)))
    alpha = pt.as_tensor([0.0, 0.0])

    result = vectorized_geometric_adstock(x, alpha)
    result_val = result.eval()

    assert result_val.shape == (n_weeks, n_channels)
    # With alpha=0, adstock[t] = spend[t]
    np.testing.assert_allclose(result_val[:, 0], 1.0, atol=1e-6)


def test_adstock_shape():
    """Output shape should match input shape."""
    n_weeks = 20
    n_channels = 3
    x = pt.as_tensor(np.random.uniform(0, 100, (n_weeks, n_channels)))
    alpha = pt.as_tensor([0.5, 0.6, 0.7])

    result = vectorized_geometric_adstock(x, alpha)
    result_val = result.eval()

    assert result_val.shape == (n_weeks, n_channels)


def test_adstock_causal():
    """Future spend should not affect past periods."""
    n_weeks = 10
    n_channels = 1
    x_np = np.zeros((n_weeks, n_channels))
    x_np[5, 0] = 100
    x = pt.as_tensor(x_np)
    alpha = pt.as_tensor([0.5])

    result = vectorized_geometric_adstock(x, alpha)
    result_val = result.eval()

    # Periods before impulse should be zero
    np.testing.assert_allclose(result_val[:5, 0], 0, atol=1e-6)


def test_adstock_accumulates():
    """Adstocked spend should be >= original spend due to carryover."""
    n_weeks = 10
    n_channels = 1
    x = pt.as_tensor(np.ones((n_weeks, n_channels)) * 10)
    alpha = pt.as_tensor([0.5])

    result = vectorized_geometric_adstock(x, alpha)
    result_val = result.eval()

    # Adstocked spend should be >= original (due to accumulation)
    assert np.all(result_val >= 10 - 1e-6)
