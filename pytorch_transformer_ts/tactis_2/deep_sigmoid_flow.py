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

        # Initialize logdet with the correct dimensions
        # We need to match x's full shape for proper accumulation of log determinant
        # This would be [batch_size, num_series, time_steps] for our model
        if x.dim() > 2:
            # For multi-dimensional tensors like [batch, series, timesteps]
            batch_shape = x.shape[:-1]  # All but the last dimension
        else:
            # For simpler cases
            batch_shape = x.shape
        logdet = torch.zeros(batch_shape, device=x.device) # Initialize logdet here

        # Track original x for debugging
        x_orig = x.clone()

        for i, layer in enumerate(self.layers):
            # Extract parameters for this layer
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]

            # Check for NaNs in inputs
            if torch.isnan(x).any():
                logger.warning(f"NaN in x before DeepSigmoidFlow layer {i}")
                # Replace NaNs with zeros to allow continued processing
                x = torch.nan_to_num(x)

            if torch.isnan(logdet).any():
                logger.warning(f"NaN in logdet before DeepSigmoidFlow layer {i}")
                logdet = torch.nan_to_num(logdet)

            # Process through the layer
            x, logdet = layer(
                layer_params,
                x,
                logdet,
            )

        # Final check for NaNs
        if torch.isnan(x).any() or torch.isnan(logdet).any():
            logger.warning("NaNs in final output of DeepSigmoidFlow, replacing with zeros")
            # Replace NaNs as a last resort
            x = torch.nan_to_num(x)
            logdet = torch.nan_to_num(logdet)

        return x, logdet

    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        for i, layer in enumerate(self.layers):
            x = layer.forward_no_logdet(
                params[..., i * self.params_length : (i + 1) * self.params_length], x
            )
        return x