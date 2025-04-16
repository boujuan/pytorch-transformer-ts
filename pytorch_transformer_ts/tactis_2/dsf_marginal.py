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

        # Add a dimension reduction layer if the expected context_dim doesn't match what we'll get from the encoder
        self.input_dim = None  # Will be set on first call if needed
        self.dim_reducer = None  # Will be initialized on first call if needed

        # Create conditioner network
        layers = [nn.Linear(context_dim, mlp_dim), nn.ReLU()]
        for _ in range(1, mlp_layers):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, self.marginal_flow.total_params_length)]
        self.marginal_conditioner = nn.Sequential(*layers)

        # Store the expected context_dim for later checks
        self.expected_context_dim = context_dim

    def _ensure_dimensions_match(self, input_tensor):
        """Ensure input dimensions match what's expected by adding a dimension reducer if needed"""
        # Get the dimension of the input tensor
        input_dim = input_tensor.size(-1)

        # If dimensions don't match and we haven't initialized a reducer yet
        if input_dim != self.expected_context_dim and self.dim_reducer is None:
            logger.debug(f"Creating dimension reducer from {input_dim} to {self.expected_context_dim}")
            self.input_dim = input_dim
            self.dim_reducer = nn.Linear(input_dim, self.expected_context_dim)
            # Move to the same device as the input tensor
            self.dim_reducer = self.dim_reducer.type_as(input_tensor) # CHANGE to(input_tensor.device)

        # Apply dimension reduction if needed
        if self.dim_reducer is not None:
            return self.dim_reducer(input_tensor)
        else:
            return input_tensor

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

        # Reshape context for linear layer
        # context comes in as [batch, series, time, dim] but the linear layer expects [batch, dim]
        # or [N, dim] where N = batch * series * time
        batch_size, num_series, num_timesteps, feature_dim = context.shape

        # Reshape to [batch*series*time, feature_dim]
        reshaped_context = context.reshape(-1, feature_dim)

        # Ensure dimensions match before applying conditioner
        reshaped_context = self._ensure_dimensions_match(reshaped_context)

        # Check for NaNs after reshaping
        if torch.isnan(reshaped_context).any():
            logger.warning("NaN values detected after reshaping context, replacing with zeros")
            reshaped_context = torch.nan_to_num(reshaped_context, nan=0.0)

        # Apply conditioner
        try:
            marginal_params = self.marginal_conditioner(reshaped_context)

            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                logger.warning("NaN values in marginal parameters, replacing with zeros")
                marginal_params = torch.nan_to_num(marginal_params, nan=0.0)

            # Reshape back to match x
            marginal_params = marginal_params.reshape(batch_size, num_series, num_timesteps, -1)

            # Make sure dimensions match
            if marginal_params.dim() == x.dim():
                # Check if reshaping is needed
                if marginal_params.shape != x.shape:
                    marginal_params = marginal_params[:, :, :, :x.size(-1)]

            # Forward through the flow
            transformed_x, logdet = self.marginal_flow.forward(marginal_params, x)

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
            zeros_logdet = torch.zeros(batch_size, num_series, num_timesteps, device=x.device)
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

        # Reshape context for linear layer
        batch_size, num_series, num_timesteps, feature_dim = context.shape

        # Reshape to [batch*series*time, feature_dim]
        reshaped_context = context.reshape(-1, feature_dim)

        # Ensure dimensions match before applying conditioner
        reshaped_context = self._ensure_dimensions_match(reshaped_context)

        # Check for NaNs after reshaping
        if torch.isnan(reshaped_context).any():
            logger.warning("NaN values after reshaping context in forward_no_logdet, replacing with zeros")
            reshaped_context = torch.nan_to_num(reshaped_context, nan=0.0)

        try:
            # Apply conditioner
            marginal_params = self.marginal_conditioner(reshaped_context)

            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                logger.warning("NaN values in marginal parameters for forward_no_logdet, replacing with zeros")
                marginal_params = torch.nan_to_num(marginal_params, nan=0.0)

            # Reshape back to match x
            marginal_params = marginal_params.reshape(batch_size, num_series, num_timesteps, -1)

            # Make sure dimensions match
            if marginal_params.dim() == x.dim():
                # Check if reshaping is needed
                if marginal_params.shape != x.shape:
                    marginal_params = marginal_params[:, :, :, :x.size(-1)]

            # Forward through the flow
            transformed_x = self.marginal_flow.forward_no_logdet(marginal_params, x)

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
        # Reshape context for linear layer
        if context.dim() == 4:  # [batch, series, time, dim]
            batch_size, num_series, num_timesteps, feature_dim = context.shape

            # Reshape to [batch*series*time, feature_dim]
            reshaped_context = context.reshape(-1, feature_dim)

            # Ensure dimensions match before applying conditioner
            reshaped_context = self._ensure_dimensions_match(reshaped_context)

            # Apply conditioner
            marginal_params = self.marginal_conditioner(reshaped_context)

            # Reshape back
            marginal_params = marginal_params.reshape(batch_size, num_series, num_timesteps, -1)

            # Make sure dimensions match
            if marginal_params.dim() == u.dim():
                # Check if reshaping is needed
                if marginal_params.shape != u.shape:
                    marginal_params = marginal_params[:, :, :, :u.size(-1)]
        else:
            # Fall back to original method for other cases
            reshaped_context = self._ensure_dimensions_match(context)
            marginal_params = self.marginal_conditioner(reshaped_context)

            # Make sure dimensions match
            if marginal_params.dim() == u.dim():
                if marginal_params.shape != u.shape:
                    # If the parameter is missing a dimension that u has
                    if marginal_params.dim() < u.dim():
                        for _ in range(u.dim() - marginal_params.dim()):
                            marginal_params = marginal_params.unsqueeze(-2)
                    # Make sure the last dimension matches
                    marginal_params = marginal_params[..., :u.size(-1)]

        left = -1000.0 * torch.ones_like(u)
        right = 1000.0 * torch.ones_like(u)

        for _ in range(max_iter):
            mid = (left + right) / 2

            # Evaluate CDF at midpoint
            cdf_mid = self.marginal_flow.forward_no_logdet(marginal_params, mid)

            # Update bounds
            left = torch.where(cdf_mid < u, mid, left)
            right = torch.where(cdf_mid >= u, mid, right)

        return (left + right) / 2