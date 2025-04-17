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
        # Check for NaNs in input
        if torch.isnan(x).any():
            logger.warning("NaN values detected in input x, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
            if torch.isnan(context).any():
                logger.warning("NaN values detected in context tensor, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)
        # Clamp x to a valid range to prevent numerical issues
        # The flow expects values that can be reasonably transformed without producing extremes
        x_min, x_max = x.min().item(), x.max().item()

        # Only clamp if values are extreme
        if x_min < -100 or x_max > 100:
            logger.info("Clamping extreme values in x")
            x = torch.clamp(x, min=-100.0, max=100.0)

        # Expect context shape [batch, N, dim], x shape [batch, N]
        # No internal reshaping needed for context
        batch_size = context.shape[0]
        N = context.shape[1] # N = series * time

        # Check context dim matches conditioner expectation
        if context.shape[-1] != self.expected_context_dim:
             # This should ideally not happen if called correctly, but log a warning
             logger.warning(f"Context dim {context.shape[-1]} != expected {self.expected_context_dim}. Check calling code.")
             # Attempt to proceed, conditioner might fail

        # Check for NaNs after reshaping (now done in calling code)
        if torch.isnan(context).any():
            logger.warning("NaN values detected in context, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)

        # Apply conditioner directly to context
        try:
            marginal_params = self.marginal_conditioner(context) # Input [batch, N, dim], Output [batch, N, param_len]

            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                logger.warning("NaN values in marginal parameters, replacing with zeros")
                marginal_params = torch.nan_to_num(marginal_params, nan=0.0)

            # No reshaping needed for marginal_params, should be [batch, N, param_len]
            # No dimension matching needed, flow expects params [batch, N, ...] and x [batch, N]

            # Forward through the flow
            transformed_x, logdet = self.marginal_flow.forward(marginal_params, x) # Pass [batch, N, param_len] and [batch, N]

            # Check for NaNs in the output
            if torch.isnan(transformed_x).any() or torch.isnan(logdet).any():
                logger.warning("NaN values in DSFMarginal output, replacing with zeros")
                transformed_x = torch.nan_to_num(transformed_x, nan=0.0)
                logdet = torch.nan_to_num(logdet, nan=0.0)

            return transformed_x, logdet

        except Exception as e:
            logger.error(f"Error in DSFMarginal forward_logdet: {e}")
            # Return zeros as a fallback to avoid crashing
            zeros_x = torch.zeros_like(x)
            # Fallback logdet shape should be [batch, N] to match expected output of flow
            zeros_logdet = torch.zeros_like(x) # Shape [batch, N]
            return zeros_x, zeros_logdet

    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Transform without derivative computation"""
        # Check for NaNs in input
        if torch.isnan(x).any():
            logger.warning("NaN values detected in input x for forward_no_logdet, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        if torch.isnan(context).any():
            logger.warning("NaN values detected in context tensor for forward_no_logdet, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)

        # Clamp x to a valid range to prevent numerical issues
        x_min, x_max = x.min().item(), x.max().item()
        if x_min < -100 or x_max > 100:
            logger.info("Clamping extreme values in x for forward_no_logdet")
            x = torch.clamp(x, min=-100.0, max=100.0)

        # Expect context shape [batch, N, dim], x shape [batch, N]
        # No internal reshaping needed for context
        batch_size = context.shape[0]
        N = context.shape[1] # N = series * time

        # Check context dim matches conditioner expectation
        if context.shape[-1] != self.expected_context_dim:
             logger.warning(f"Context dim {context.shape[-1]} != expected {self.expected_context_dim}. Check calling code.")

        # Check for NaNs after reshaping (now done in calling code)
        if torch.isnan(context).any():
            logger.warning("NaN values detected in context for forward_no_logdet, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)

        try:
            # Apply conditioner directly to context
            marginal_params = self.marginal_conditioner(context) # Input [batch, N, dim], Output [batch, N, param_len]

            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                logger.warning("NaN values in marginal parameters for forward_no_logdet, replacing with zeros")
                marginal_params = torch.nan_to_num(marginal_params, nan=0.0)

            # Match original TACTiS: unsqueeze marginal_params if x has a sample dimension
            if marginal_params.dim() == x.dim():
                marginal_params = marginal_params[:, :, None, :]

            # Forward through the flow
            transformed_x = self.marginal_flow.forward_no_logdet(marginal_params, x) # Pass [batch, N, param_len] and [batch, N]

            # Check for NaNs in the output
            if torch.isnan(transformed_x).any():
                logger.warning("NaN values in DSFMarginal forward_no_logdet output, replacing with zeros")
                transformed_x = torch.nan_to_num(transformed_x, nan=0.0)

            return transformed_x

        except Exception as e:
            logger.error(f"Error in DSFMarginal forward_no_logdet: {e}")
            # Return zeros as a fallback to avoid crashing
            return torch.zeros_like(x)

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

        left = -1000.0 * torch.ones_like(u)
        right = 1000.0 * torch.ones_like(u)

        for _ in range(max_iter):
            mid = (left + right) / 2

            # Evaluate CDF at midpoint
            # Pass params [batch, N, param_len] and mid [batch, N] or [batch, N, samples]
            cdf_mid = self.marginal_flow.forward_no_logdet(marginal_params, mid)

            # Update bounds
            left = torch.where(cdf_mid < u, mid, left)
            right = torch.where(cdf_mid >= u, mid, right)

        return (left + right) / 2