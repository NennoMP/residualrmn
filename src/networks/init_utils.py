"""Module containing utilities for weight matrices initialization and spectral rescaling.

Code is adapted from [1] and [2].

References:
[1] https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
[2] https://github.com/andreaceni/ResESN
"""
import numpy as np
import torch
import torch.nn as nn


def init_orthogonal(
    M: int,
    ortho_config: str, 
) -> nn.Parameter:
    """Generate a (M, M) orthogonal matrix to be used in the temporal residual connections.

    Supported orthogonal initializations include a random orthogonal matrix obtained via QR 
    decomposition ('ortho'), a cyclic orthogonal matrix ('cycle'), and the identity matrix 
    ('identity'). The cyclic orthogonal matrix is a zero matrix with ones on the lower subdiagonal 
    and a one in the top-right corner, i.e.:

    C = [[0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]]
    
    Args:
        M: Numbers of rows/columns.
        ortho_config: Type of orthogonal initialization to use. Options are 'ortho', 'cycle', 'identity'.

    Returns:
        A torch.nn.Parameter containing the initialized (M, M) orthogonal matrix.

    Raises:
        ValueError: If an invalid ortho_config is provided.
    """
    # Random orthogonal
    if ortho_config == 'ortho':
        O, _ = np.linalg.qr(2 * np.random.rand(M, M) - 1)
    # Cyclic orthogonal
    elif ortho_config == 'cycle':
        O = np.zeros((M, M))
        O[0, M - 1] = 1
        O[torch.arange(1, M), torch.arange(M - 1)] = 1
    # Identity 
    elif ortho_config == 'identity':
        O = np.eye(M, M)
    else:
        raise ValueError(
            f"Invalid skip option: {ortho_config}. Options are 'ortho', 'cycle', or 'identity'."
        )
    
    return nn.Parameter(torch.Tensor(O), requires_grad=False)

def sparse_tensor_init(M: int, N: int) -> torch.FloatTensor:
    """Generate a (M, N) fully-connected matrix to be used as input kernel.
    
    Args:
        M: Number of rows.
        N: Number of columns,

    Returns:
        A torch.FloatTensor containing the initialized (M, N) input kernel.
    """
    # Generate a random matrix with values in the range [-1, 1)
    return (2 * torch.rand(M, N) - 1).float()

def init_recurrent_kernel(
    M: int,
    rho: float,
    init: str = 'uniform',
) -> nn.Parameter:
    """Generate a (M, M) fully-connected matrix to be used as recurrent kernel.
    
    Args:
        N: Number of rows/columns.
        rho: Spectral radius for rescaling the matrix.
        init: Initialization strategy. Options are 'uniform'.

    Returns:
        A torch.nn.Parameter containing the initialized and rescaled (M, M) recurrent kernel.

    Raises:
        ValueError: If an invalid initialization option is provided.
    """
    # Generate a random matrix with values in the range [-1, 1)
    W = (2 * torch.rand(M, M) - 1).float()
    # Rescale the matrix to have the desired spectral radius
    if init == 'uniform':
        W = fast_spectral_rescaling(W, rho)
    else:
        raise ValueError(f"Invalid initialization {init}. Options are 'uniform'.")
    
    return nn.Parameter(W, requires_grad=False)

def fast_spectral_rescaling(W: torch.Tensor, rho_desired: float) -> torch.Tensor:
    """Rescale a given matrix, uniformly sampled in [-1, 1) and fully-connected, to have the 
    specified spectral radius.
    
    Note that uniform sampling and fully-connectedness are required for this method to work.

    Args:
        W: Matrix to be rescaled.
        rho_desired: Desired spectral radius.

    Returns:
        A torch.FloatTensor containing the rescaled matrix.
    """
    M = W.shape[0]
    rescaling  = (rho_desired / np.sqrt(M)) * (6 / np.sqrt(12))
    return W * rescaling

def init_bias(
    M: int,  
    bias_scaling: float, 
) -> nn.Parameter:
    """Generate a (M, 1) bias vector to be used in the recurrent kernel.

    Args:
        M: Vector size.
        bias_scaling: Scaling factor for the bias vector.
    
    Returns:
        A torch.nn.Parameter containing the initialized (M, 1) bias vector.
    """
    bias = torch.zeros(M)
    bias = nn.init.uniform_(bias, a=-bias_scaling, b=bias_scaling)
    return nn.Parameter(bias, requires_grad=False)
