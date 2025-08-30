"""Implementation of Residual Reservoir Memory Network (ResRMN) in PyTorch.

Note that the concept of reservoir memory networks has been introduced in [1].

References:
[1] C. Gallicchio and A. Ceni, Reservoir Memory Networks, ESANN (2024)
"""

import torch
import torch.nn as nn

import src.networks.init_utils as init_utils
import src.networks.reservoir as reservoir


class ResidualRMN(nn.Module):
    """Residual Reservoir Memory Network (ResRMN).

    ResRMN is characterized by a modular and hierarchical structure, combining a linear
    memory reservoir and a non-linear reservoir based on temporal residual connections.
    """

    def __init__(
        self,
        in_size: int = 1,
        # Memory module parameters
        n_units_m: int = 100,
        in_scaling_m: float = 1.0,
        bias_scaling_m: float = 0.0,
        # Non-linear module parameters
        n_units: int = 100,
        act: nn.Module = nn.Tanh(),
        in_scaling: float = 1.0,
        memory_scaling: float = 1.0,
        bias_scaling: float = 0.0,
        rho: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            in_size: Size of the input.

            n_units_m: Number of units in the memory reservoir.
            in_scaling_m: Scaling for the input kernel of the memory reservoir.
            bias_scaling_m: Scaling for the bias vector of the memory reservoir.

            n_units: Number of units in the non-linear reservoir.
            act: Activation function for the non-linear reservoir.
            in_scaling: Scaling for the input kernel of the non-linear reservoir
            memory_scaling: Scaling for the memory kernel of the non-linear reservoir.
            bias_scaling: Scaling for the bias vector of the non-linear reservoir.
            rho: Spectral radius used to rescale the recurrent kernel of the non-linear
            reservoir.
            alpha: Scaling coefficient for the temporal residual connections.
            beta: Scaling coefficient for the non-linear branch.
        """
        super().__init__()
        self.__dict__.update(kwargs)

        self.in_size = in_size
        # Memory module parameters
        self.n_units_m = n_units_m
        self.in_scaling_m = in_scaling_m
        self.bias_scaling_m = bias_scaling_m
        # Non-linear module parameters
        self.n_units = n_units
        self.act = act
        self.in_scaling = in_scaling
        self.memory_scaling = memory_scaling
        self.bias_scaling = bias_scaling
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        # Initialize modules
        self._make_layers()

    def _make_layers(self) -> None:
        """Initialize the memory and non-linear modules."""
        # Memory module
        self.memory_reservoir = reservoir.Reservoir(
            cell=reservoir.MemoryCell(
                in_size=self.in_size,
                n_units=self.n_units_m,
                in_scaling=self.in_scaling_m,
                bias_scaling=self.bias_scaling_m,
            ),
        )
        # Non-linear module
        self.nonlinear_reservoir = reservoir.Reservoir(
            cell=reservoir.ReservoirCell(
                in_size=self.in_size,
                memory_size=self.n_units_m,
                n_units=self.n_units,
                act=self.act,
                in_scaling=self.in_scaling,
                memory_scaling=self.memory_scaling,
                bias_scaling=self.bias_scaling,
                rho=self.rho,
                alpha=self.alpha,
                beta=self.beta,
            )
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: External input tensor of shape (batch_size, T, in_size).

        Returns:
            h: Hidden states of shape (batch_size, T, n_units).
            last_h: Last hidden state of shape (batch_size, n_units).
        """
        m, _ = self.memory_reservoir(x)
        h, last_h = self.nonlinear_reservoir(x, m)
        return h, last_h


def residualrmn_(hparams: dict) -> ResidualRMN:
    if "skip_option_t" not in hparams:
        raise ValueError(
            "Required parameter 'skip_option_t' is missing. Options are 'ortho', "
            "'cycle', and 'identity'."
        )
    model = ResidualRMN(**hparams)
    model.nonlinear_reservoir.cell.O = init_utils.init_orthogonal(
        M=model.nonlinear_reservoir.cell.n_units, ortho_config=hparams["skip_option_t"]
    )
    return model
