import logging
from typing import Tuple
import torch
from torch import nn

from .deep_sigmoid_flow import DeepSigmoidFlow

# Set up logging
logger = logging.getLogger(__name__)

class DSFMarginal(nn.Module):
    """
    Deep Sigmoid Flow for marginal distribution modeling
    """
    def __init__(
        self,
        context_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        flow_layers: int,
        flow_hid_dim: int,
    ):
        super().__init__()
        self.marginal_flow = DeepSigmoidFlow(n_layers=flow_layers, hidden_dim=flow_hid_dim)

        # Dimension reduction logic removed as context shape is now expected to be correct
        # self.input_dim = None
        # self.dim_reducer = None

        # Create conditioner network - Expects context_dim input
        layers = [nn.Linear(context_dim, mlp_dim), nn.ReLU()]
        for _ in range(1, mlp_layers):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, self.marginal_flow.total_params_length)]
        self.marginal_conditioner = nn.Sequential(*layers)

        # Store the expected context_dim for later checks
        self.expected_context_dim = context_dim

    # Removed _ensure_dimensions_match method

    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform with derivative computation"""
        # Expect context shape [batch, N, dim], x shape [batch, N]
        # No internal reshaping needed for context
        batch_size = context.shape[0]
        N = context.shape[1] # N = series * time

        # Check context dim matches conditioner expectation
        if context.shape[-1] != self.expected_context_dim:
             # This should ideally not happen if called correctly, but log a warning
             logger.warning(f"Context dim {context.shape[-1]} != expected {self.expected_context_dim}. Check calling code.")
             # Attempt to proceed, conditioner might fail

        # Apply conditioner directly to context
        marginal_params = self.marginal_conditioner(context) # Input [batch, N, dim], Output [batch, N, param_len]

        # No reshaping needed for marginal_params, should be [batch, N, param_len]
        # No dimension matching needed, flow expects params [batch, N, ...] and x [batch, N]

        # Forward through the flow
        transformed_x, logdet = self.marginal_flow.forward(marginal_params, x) # Pass [batch, N, param_len] and [batch, N]

        return transformed_x, logdet


    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Transform without derivative computation"""
        # Expect context shape [batch, N, dim], x shape [batch, N]
        # No internal reshaping needed for context
        batch_size = context.shape[0]
        N = context.shape[1] # N = series * time

        # Check context dim matches conditioner expectation
        if context.shape[-1] != self.expected_context_dim:
             logger.warning(f"Context dim {context.shape[-1]} != expected {self.expected_context_dim}. Check calling code.")

        # Apply conditioner directly to context
        marginal_params = self.marginal_conditioner(context) # Input [batch, N, dim], Output [batch, N, param_len]

        # Match original TACTiS: unsqueeze marginal_params if x has a sample dimension
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]

        # Forward through the flow
        transformed_x = self.marginal_flow.forward_no_logdet(marginal_params, x) # Pass [batch, N, param_len] and [batch, N]

        return transformed_x

    def inverse(self, context: torch.Tensor, u: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
        """Compute inverse CDF using binary search"""
        # Expect context shape [batch, N, dim], u shape [batch, N] or [batch, N, samples]
        # No internal reshaping needed for context
        batch_size = context.shape[0]
        N = context.shape[1] # N = series * time

        # Check context dim matches conditioner expectation
        if context.shape[-1] != self.expected_context_dim:
             logger.warning(f"Context dim {context.shape[-1]} != expected {self.expected_context_dim}. Check calling code.")

        # Apply conditioner directly to context
        marginal_params = self.marginal_conditioner(context) # Output [batch, N, param_len]

        # unsqueeze marginal_params if u has a sample dimension
        if marginal_params.dim() == u.dim():
            marginal_params = marginal_params[:, :, None, :]
 
        # Adjust bounds for [-1, 1] normalized data scale
        left = -2.0 * torch.ones_like(u)
        right = 2.0 * torch.ones_like(u)

        for iter_num in range(max_iter): # Add iter_num for logging
            mid = (left + right) / 2
            cdf_mid = self.marginal_flow.forward_no_logdet(marginal_params, mid)

            # Update bounds
            left = torch.where(cdf_mid < u, mid, left)
            right = torch.where(cdf_mid >= u, mid, right)

        final_mid = (left + right) / 2

        return final_mid # Return the final mid value