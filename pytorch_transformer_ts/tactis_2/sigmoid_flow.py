import math
import logging
import torch
from torch import nn

# Constants
EPSILON = 1e-6

# Helper functions from original flow.py for numerical stability
def log_sum_exp(A, dim=-1, keepdim=False):
    """
    Compute a sum in logarithm space: log(exp(a) + exp(b))
    Properly handle values which exponential cannot be represented in float32.
    """
    max_A = A.max(axis=dim, keepdim=True).values
    norm_A = A - max_A
    result = torch.log(torch.exp(norm_A).sum(axis=dim, keepdim=True)) + max_A
    if keepdim:
        return result
    else:
        return torch.squeeze(result, dim=dim)

def log_sigmoid(x):
    """
    Logarithm of the sigmoid function.
    Substract the epsilon to avoid 0 as a possible value for large x.
    """
    return -nn.functional.softplus(-x) - EPSILON


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

        # Compute log jacobian using original stable method
        # logj shape: [b, v, h] or [b, n, h] etc.
        logj_per_hidden = (
            nn.functional.log_softmax(params[..., 2*self.hidden_dim:], dim=-1) # log(w)
            + log_sigmoid(pre_sigm)                                            # log(sigmoid(ax+b))
            + log_sigmoid(-pre_sigm)                                           # log(1-sigmoid(ax+b))
            + torch.log(a)                                                     # log(a)
        )

        # logj shape: [b, v] or [b, n] etc. (summed over hidden dim h)
        logj = log_sum_exp(logj_per_hidden, dim=-1, keepdim=False)

        # Sum logj over all non-batch dimensions before adding to logdet (which has shape [batch])
        logdet = logdet + logj.sum(dim=tuple(range(1, logj.dim())))

        # Check for NaNs in logdet
        if torch.isnan(logdet).any():
            logger.warning("NaN detected in logdet after addition")
            # Replace NaNs with zeros to allow training to continue
            logdet = torch.nan_to_num(logdet, nan=0.0)

        if self.no_logit:
            return x_pre, logdet

        # Apply logit transform with numerical stability
        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)

        # Update logdet to account for logit transform
        logdet_ = logj + math.log(1 - EPSILON) - (torch.log(x_pre_clipped) + torch.log(1 - x_pre_clipped))
        logdet = logdet_.sum(dim=tuple(range(1, logdet_.dim()))) + logdet

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

        pre_sigm = a * x[..., None] + b # Unsqueeze a and b for broadcasting
        sigm = torch.sigmoid(pre_sigm)
        x_pre = (w * sigm).sum(dim=-1)

        if self.no_logit:
            return x_pre

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)

        return xnew