"""Tests for adstock transformation."""
import numpy as np
import pytensor.tensor as pt
from src.adstock_functions import vectorized_geometric_adstock


def test_adstock_imports():
    """Test adstock function can be imported."""
    assert callable(vectorized_geometric_adstock)


def test_adstock_returns_tensor():
    """Test adstock returns PyTensor tensor."""
    x = pt.as_tensor(np.ones((5, 2)))
    alpha = pt.as_tensor([0.5, 0.5])
    result = vectorized_geometric_adstock(x, alpha)
    assert result is not None
    assert hasattr(result, 'eval')


def test_adstock_output_shape():
    """Test output shape matches input shape."""
    x = pt.as_tensor(np.ones((10, 3)))
    alpha = pt.as_tensor([0.4, 0.5, 0.6])
    result = vectorized_geometric_adstock(x, alpha)
    result_val = result.eval()
    assert result_val.shape == (10, 3)
