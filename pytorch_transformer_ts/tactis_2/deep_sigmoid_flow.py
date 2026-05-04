import logging
import torch
from torch import nn

from .sigmoid_flow import SigmoidFlow

# Set up logging
logger = logging.getLogger(__name__)

class DeepSigmoidFlow(nn.Module):
    """
    A Deep Sigmoid Flow network for flexible distribution modeling
    """
    def __init__(self, n_layers: int, hidden_dim: int, a_max: float = 0.0):
        super().__init__()
        self.params_length = 3 * hidden_dim
        self.a_max = a_max

        # Pass a_max to each SigmoidFlow layer (Fix B). a_max=0 disables capping.
        layers = [SigmoidFlow(hidden_dim, a_max=a_max) for _ in range(n_layers - 1)]
        layers.append(SigmoidFlow(hidden_dim, no_logit=True, a_max=a_max))
        self.layers = nn.ModuleList(layers)

        # Fix A: accumulator for total per-(batch, vars) log-density across all
        # layers. Populated by forward(); read by lightning_module.training_step
        # to compute the log-density regularization term. None when not yet set.
        self._last_per_datapoint_log_density: torch.Tensor = None

    @property
    def total_params_length(self):
        return len(self.layers) * self.params_length

    def forward(self, params, x):
        """Transform with derivative computation"""
        # Initialize logdet with only the batch dimension, like the original TACTiS
        logdet = torch.zeros(x.shape[0], device=x.device)
        # Fix A: accumulate per-(batch, vars) log-density across layers. The total
        # log-density of the flow at each data point is the SUM of per-layer logj
        # (after log_sum_exp over the hidden dim). This is what runs to +inf in
        # collapsed flows, so we expose it for the regularizer to penalize.
        per_datapoint_log_density_sum = None

        for i, layer in enumerate(self.layers):
            # Extract parameters for this layer
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]

            # Process through the layer (Original logic)
            x, logdet = layer(
                layer_params,
                x,
                logdet,
            )
            # Accumulate per-(batch, vars) log-density from this layer.
            # SigmoidFlow.forward sets `_last_logj_per_datapoint` (gradient-attached).
            layer_logj = getattr(layer, "_last_logj_per_datapoint", None)
            if layer_logj is not None:
                if per_datapoint_log_density_sum is None:
                    per_datapoint_log_density_sum = layer_logj
                else:
                    per_datapoint_log_density_sum = per_datapoint_log_density_sum + layer_logj

        self._last_per_datapoint_log_density = per_datapoint_log_density_sum
        return x, logdet

    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        # Original logic: Iterate through layers calling forward_no_logdet
        for i, layer in enumerate(self.layers):
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]
            x = layer.forward_no_logdet(layer_params, x)
        return x