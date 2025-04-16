"""Module containing PyTorch implementations of various reservoir modules employed in ResRMN.

Note that the concept of a linear memory reservoir has been introduced in [1], and the concept of residual echo state networks has been introduced in [2].

References:
[1] C. Gallicchio and A. Ceni, Reservoir Memory Networks, ESANN (2024).
[2] A. Ceni and C. Gallicchio, "Residual Echo State Networks: Residual recurrent neural networks 
with stable dynamics and fast learning." Neurocomputing (2024).
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import src.networks.init_utils as init_utils


class MemoryCell(nn.Module):
    """Class implementing the state update function of the memory reservoir.
    
    The memory reservoir applies a linear transformation and is driven solely by the external 
    input. The recurrent kernel is initialized as a cyclic (circular shift) orthogonal matrix.
    """
    def __init__(
        self, 
        in_size: int, 
        n_units: int, 
        in_scaling: float,
        bias_scaling: float,
    ) -> None:
        """
        Args:
            in_size: Size of the external input.
            n_units: Number of units.
            in_scaling: Scaling factor for the input kernel.
            bias_scaling: Scaling factor for the bias vector.
        """
        super().__init__()    
        self.n_units = n_units

        in_kernel = init_utils.sparse_tensor_init(
            M=in_size, 
            N=n_units,
        ) * in_scaling
        self.in_kernel = nn.Parameter(in_kernel, requires_grad=False)

        self.recurrent_memory_kernel = init_utils.init_orthogonal(
            M=n_units,
            ortho_config='cycle', # cyclic (circular shift) orthogonal matrix
        )
        self.bias = init_utils.init_bias(M=n_units, bias_scaling=bias_scaling)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        input_part = torch.mm(x, self.in_kernel)
        state_part = torch.mm(h_prev, self.recurrent_memory_kernel)
        return nn.Identity()(state_part + input_part + self.bias)
    

class ReservoirCell(nn.Module):
    """Class implementing the state update function of the non-linear reservoir.
    
    The non-linear (residual) reservoir applies a non-linear transformation and is driven by the 
    external input and the output of the memory reservoir. Its state update function consists of a linear branch (temporal residual connections) and a non-linear branch. The recurrent kernel is initialized as a random matrix and then rescaled to have a desired spectral radius.
    """
    def __init__(
        self, 
        in_size: int, 
        memory_size: int,
        n_units: int, 
        act: nn.Module,
        in_scaling: float,
        memory_scaling: float,
        bias_scaling: float,
        rho: float,
        alpha: float,
        beta: float,
    ) -> None:
        """
        Args:
            in_size: Size of the external input.
            memory_size: Size of the memory module output.
            n_units: Number of units.
            act: Activation function.
            in_scaling: Scaling factor for the input kernel.
            memory_scaling: Scaling factor for the memory kernel.
            bias_scaling: Scaling factor for the bias vector.
            rho: Spectral radius for rescaling the recurrent matrix.
            alpha: Coefficient for the temporal residual connections.
            beta: Coefficient for the non-linear branch.
        """
        super().__init__()    
        self.n_units = n_units
        self.act = act

        self.alpha = nn.Parameter(torch.tensor([alpha]), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor([beta]), requires_grad=False)
        
        in_kernel = init_utils.sparse_tensor_init(
            M=in_size, 
            N=n_units,
        ) * in_scaling
        self.in_kernel = nn.Parameter(in_kernel, requires_grad=False)

        memory_kernel = init_utils.sparse_tensor_init(
            M=memory_size, 
            N=n_units,
        ) * memory_scaling
        self.memory_kernel = nn.Parameter(memory_kernel, requires_grad=False)
        
        self.recurrent_kernel = init_utils.init_recurrent_kernel(
            M=n_units,
            rho=rho,
        )
        self.bias = init_utils.init_bias(M=n_units, bias_scaling=bias_scaling)
        
        # Temporal residual connections
        self.O = nn.Parameter(torch.zeros(n_units, n_units), requires_grad=False)
    
    def forward(self, x: torch.Tensor, m: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        input_part = torch.mm(x, self.in_kernel)
        memory_part = torch.mm(m, self.memory_kernel)
        state_part = torch.mm(h_prev, self.recurrent_kernel)
        residual_part = torch.mm(h_prev, self.O) # shuffle previous state

        out = (self.alpha * residual_part) + (self.beta * self.act(state_part + memory_part + input_part + self.bias))
        return out


class Reservoir(nn.Module):
    def __init__(self, cell: Union[MemoryCell, ReservoirCell]) -> None:
        """
        Args:
            cell: The reservoir cell to be used in the reservoir. Options are 'MemoryCell' or a 'ReservoirCell'.

        Raises:
            AssertionError: If the cell is not an instance of 'MemoryCell' or 'ReservoirCell'.
        """
        super().__init__()
        assert isinstance(cell, (MemoryCell, ReservoirCell)), \
            "reservoir cell must be an instance of 'MemoryCell' or 'ReservoirCell'."
        self.cell = cell

    def _init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize the hidden state of the reservoir."""
        return torch.zeros(batch_size, self.cell.n_units)

    def forward(
        self, 
        x: torch.Tensor, 
        m: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: External input tensor of shape (batch_size, T, in_size).
            m: Memory module input tensor of shape (batch_size, T, n_units_m).
            h_prev: Previous hidden state tensor of shape (batch_size, n_units).

        Returns:
            The hidden statest of the model, with shape (batch_size, T, n_units), and the hidden 
            state at the last time step, with shape (batch_size, n_units).
        """
        # batch size, time steps
        batch_size, T = x.shape[0], x.shape[1]
        # hidden states
        h = torch.zeros(batch_size, T, self.cell.n_units, device=x.device)

        if h_prev is None:
            h_prev = self._init_hidden(batch_size).to(x.device)

        for t in range(T):
            if isinstance(self.cell, MemoryCell):
                h_prev = self.cell(x[:, t], h_prev)
            elif isinstance(self.cell, ReservoirCell):
                h_prev = self.cell(x[:, t], m[:, t], h_prev)
            h[:, t, :] = h_prev

        return h, h[:, -1, :]