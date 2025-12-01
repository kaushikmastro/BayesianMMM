import pytensor.tensor as pt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytensor.tensor import TensorVariable

def vectorized_geometric_adstock(x: 'TensorVariable', alpha: 'TensorVariable') -> 'TensorVariable':
    """
    Applies geometric adstock transformation using explicit matrix multiplication (convolutional form).
    This is the vectorized, non-recursive equivalent to the geometric adstock formula.
    
    Formula: A_t = Sum_{l=0}^{t-1} alpha^l * S_{t-l}

    Args:
        x (pt.TensorVariable): The normalized media spend matrix (N_weeks x N_channels).
        alpha (pt.TensorVariable): The decay rate for each channel (N_channels, where 0 < alpha < 1).

    Returns:
        pt.TensorVariable: The Adstock-transformed media effect matrix (N_weeks x N_channels).
    """
    n_weeks = x.shape[0]

    # Create the matrix of time differences (t - j)
    time_indices = pt.arange(n_weeks)
    TimeMatrix = time_indices.dimshuffle(0, 'x') - time_indices.dimshuffle('x', 0)

    # Create the Decay Matrix D: D[i, j, k] = alpha[k]^(i - j) if i >= j, else 0
    alpha_power_matrix = pt.power(alpha.dimshuffle('x', 'x', 0), TimeMatrix.dimshuffle(0, 1, 'x'))

    # Set elements where the lag is negative (i < j) to zero (no future spend affects past adstock)
    DecayMatrix = pt.where(TimeMatrix.dimshuffle(0, 1, 'x') >= 0, alpha_power_matrix, 0)
    
    #  Perform Batched Multiplication/Convolution
    Adstock_t,k = sum_j (DecayMatrix_t,j,k * X_j,k)
    
    # Reshape X for broadcasting across the middle axis of DecayMatrix
    X_reshaped = x.dimshuffle(0, 'x', 1) 
    
    # Element-wise product: DecayMatrix * X_reshaped
    product = DecayMatrix * X_reshaped 

    # Sum over the second axis (the lag index)
    adstock_output = pt.sum(product, axis=1) 
    
    return adstock_output