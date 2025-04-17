import math
import logging
import torch
from torch import nn

# Constants
EPSILON = 1e-6

# Set up logging
logger = logging.getLogger(__name__)

class SigmoidFlow(nn.Module):
    """
    A single layer of the Deep Sigmoid Flow network.
    """
    def __init__(self, hidden_dim: int, no_logit: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.no_logit = no_logit

    def forward(self, params, x, logdet):
        """Transform with derivative computation"""
        # Apply softplus with numerical stability
        a = torch.nn.functional.softplus(params[..., :self.hidden_dim]) + EPSILON
        b = params[..., self.hidden_dim:2*self.hidden_dim]
        w = torch.nn.functional.softmax(params[..., 2*self.hidden_dim:], dim=-1)

        # Compute pre-sigmoid with better numerical stability
        # Clamp x to avoid extreme values that could cause instability
        x_clamped = torch.clamp(x, min=-100.0, max=100.0)
        pre_sigm = a * x_clamped[..., None] + b # Revert: Broadcasting works directly here during training

        # Apply sigmoid with numerical stability
        # Use torch.sigmoid which is more numerically stable than manual computation
        sigm = torch.sigmoid(pre_sigm)

        # Add small epsilon to prevent zeros
        sigm = torch.clamp(sigm, min=EPSILON, max=1.0-EPSILON)

        # Compute weighted sum
        x_pre = (w * sigm).sum(dim=-1)

        # Compute log jacobian with numerical stability
        # (1-sigm) can be close to 0 for large pre_sigm values
        logj = (w * sigm * (1 - sigm) * a).sum(dim=-1)
        # Add epsilon to prevent log of zero
        logj = torch.clamp(logj, min=EPSILON)

        # Remove the dimension mismatch fix section (lines 50-55)

        # Add log jacobian with checks for NaN
        log_logj = torch.log(logj) # logj shape is likely [batch, series, time] or similar
        if torch.isnan(log_logj).any():
            logger.warning("NaN detected in log_logj, replacing with log(EPSILON)")
            # Replace NaNs with a large negative number (log of EPSILON)
            log_logj = torch.nan_to_num(log_logj, nan=math.log(EPSILON))

        # Sum log_logj over all non-batch dimensions before adding to logdet (which has shape [batch])
        logdet = logdet + log_logj.sum(dim=tuple(range(1, log_logj.dim())))

        # Check for NaNs in logdet
        if torch.isnan(logdet).any():
            logger.warning("NaN detected in logdet after addition")
            # Replace NaNs with zeros to allow training to continue
            logdet = torch.nan_to_num(logdet, nan=0.0)

        if self.no_logit:
            return x_pre, logdet

        # Apply logit transform with numerical stability
        x_pre_clipped = torch.clamp(x_pre, min=EPSILON, max=1.0-EPSILON)
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)

        # Update logdet to account for logit transform
        logdet_logit = torch.log(x_pre_clipped) + torch.log(1 - x_pre_clipped)
        logdet = logdet - logdet_logit.sum(dim=tuple(range(1, logdet_logit.dim())))

        # Final NaN check
        if torch.isnan(xnew).any() or torch.isnan(logdet).any():
            logger.warning(f"NaNs in final SigmoidFlow output, replacing with zeros")
            xnew = torch.nan_to_num(xnew)
            logdet = torch.nan_to_num(logdet)

        return xnew, logdet

    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        a = torch.nn.functional.softplus(params[..., :self.hidden_dim]) + EPSILON
        b = params[..., self.hidden_dim:2*self.hidden_dim]
        w = torch.nn.functional.softmax(params[..., 2*self.hidden_dim:], dim=-1)

        # Log shapes for debugging
        logger.error(f"[DEBUG] sigmoid_flow.forward_no_logdet shapes: a={a.shape}, b={b.shape}, x={x.shape}, params={params.shape}")

        pre_sigm = a * x[..., None] + b # Unsqueeze a and b for broadcasting
        sigm = torch.sigmoid(pre_sigm)
        x_pre = (w * sigm).sum(dim=-1)

        if self.no_logit:
            return x_pre

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)

        return xnew