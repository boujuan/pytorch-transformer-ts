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
    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        self.params_length = 3 * hidden_dim

        layers = [SigmoidFlow(hidden_dim) for _ in range(n_layers - 1)]
        layers.append(SigmoidFlow(hidden_dim, no_logit=True))
        self.layers = nn.ModuleList(layers)

    @property
    def total_params_length(self):
        return len(self.layers) * self.params_length

    def forward(self, params, x):
        """Transform with derivative computation"""
        # Initialize logdet with only the batch dimension, like the original TACTiS
        logdet = torch.zeros(x.shape[0], device=x.device)

        for i, layer in enumerate(self.layers):
            # Extract parameters for this layer
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]

            # Process through the layer (Original logic)
            x, logdet = layer(
                layer_params,
                x,
                logdet,
            )
        return x, logdet

    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        # Original logic: Iterate through layers calling forward_no_logdet
        for i, layer in enumerate(self.layers):
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]
            x = layer.forward_no_logdet(layer_params, x)
        return x