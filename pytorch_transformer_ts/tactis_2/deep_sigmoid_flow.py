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
        # Fix Sa: list of per-layer `a` tensors (post-cap, post-EPSILON).
        # Populated by forward() and forward_no_logdet(); read by lightning_module
        # to compute the a_floor regularization term. Empty list when not yet set.
        self._last_a_post_cap_per_layer: list = []
        # Fix Sw: list of per-layer per-(b,v) softmax-weight entropy.
        # Populated by forward() and forward_no_logdet(); read by lightning_module
        # to compute the w-entropy regularization term.
        self._last_w_entropy_per_layer: list = []

    @property
    def total_params_length(self):
        return len(self.layers) * self.params_length

    def forward(self, params, x):
        """Transform with derivative computation"""
        # Initialize logdet with only the batch dimension, like the original TACTiS
        logdet = torch.zeros(x.shape[0], device=x.device)
        # Fix A: accumulate per-(batch, vars) log-density across layers.
        per_datapoint_log_density_sum = None
        # Fix Sa: reset and rebuild per-layer post-cap `a` list each call.
        self._last_a_post_cap_per_layer = []
        # Fix Sw: reset and rebuild per-layer w-entropy list each call.
        self._last_w_entropy_per_layer = []

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
            layer_logj = getattr(layer, "_last_logj_per_datapoint", None)
            if layer_logj is not None:
                if per_datapoint_log_density_sum is None:
                    per_datapoint_log_density_sum = layer_logj
                else:
                    per_datapoint_log_density_sum = per_datapoint_log_density_sum + layer_logj
            # Fix Sa: collect per-layer post-cap `a` tensor.
            layer_a = getattr(layer, "_last_a_post_cap", None)
            if layer_a is not None:
                self._last_a_post_cap_per_layer.append(layer_a)
            # Fix Sw: collect per-layer w-entropy tensor.
            layer_we = getattr(layer, "_last_w_entropy", None)
            if layer_we is not None:
                self._last_w_entropy_per_layer.append(layer_we)

        self._last_per_datapoint_log_density = per_datapoint_log_density_sum
        return x, logdet

    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        # Fix Sa + Sw: reset and rebuild per-layer lists each call.
        self._last_a_post_cap_per_layer = []
        self._last_w_entropy_per_layer = []
        for i, layer in enumerate(self.layers):
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]
            x = layer.forward_no_logdet(layer_params, x)
            layer_a = getattr(layer, "_last_a_post_cap", None)
            if layer_a is not None:
                self._last_a_post_cap_per_layer.append(layer_a)
            layer_we = getattr(layer, "_last_w_entropy", None)
            if layer_we is not None:
                self._last_w_entropy_per_layer.append(layer_we)
        return x