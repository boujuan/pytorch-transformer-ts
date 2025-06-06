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
        """
        Transform the given value according to the given parameters,
        computing the derivative of the transformation at the same time.
        params third dimension must be equal to 3 times the number of hidden units.
        """
        # Indices:
        # b: batch
        # v: variables
        # h: hidden dimension

        # params: b, v, 3*h
        # x: b, v
        # logdet: b
        # output x: b, v
        # output logdet: b
        assert params.shape[-1] == 3 * self.hidden_dim

        a = torch.nn.functional.softplus(params[..., : self.hidden_dim]) + EPSILON  # b, v, h - Use softplus directly
        b = params[..., self.hidden_dim : 2 * self.hidden_dim]  # b, v, h
        pre_w = params[..., 2 * self.hidden_dim :]  # b, v, h
        w = torch.nn.functional.softmax(pre_w, dim=-1)  # b, v, h

        pre_sigm = a * x[..., None] + b  # b, v, h
        sigm = torch.sigmoid(pre_sigm)  # b, v, h
        x_pre = (w * sigm).sum(dim=-1)  # b, v

        logj = (
            nn.functional.log_softmax(pre_w, dim=-1) + log_sigmoid(pre_sigm) + log_sigmoid(-pre_sigm) + torch.log(a)
        )  # b, v, h

        logj = log_sum_exp(logj, dim=-1, keepdim=False)  # b, v

        if self.no_logit:
            # Only keep the batch dimension, summing all others in case this method is called with more dimensions
            # Adding the passed logdet here to accumulate
            logdet = logj.sum(dim=tuple(range(1, logj.dim()))) + logdet
            return x_pre, logdet

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5  # b, v
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)  # b, v

        logdet_ = logj + math.log(1 - EPSILON) - (torch.log(x_pre_clipped) + torch.log(-x_pre_clipped + 1))  # b, v

        # Only keep the batch dimension, summing all others in case this method is called with more dimensions
        logdet = logdet_.sum(dim=tuple(range(1, logdet_.dim()))) + logdet

        return xnew, logdet

    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        a = torch.nn.functional.softplus(params[..., :self.hidden_dim]) + EPSILON
        b = params[..., self.hidden_dim:2*self.hidden_dim]
        w = torch.nn.functional.softmax(params[..., 2*self.hidden_dim:], dim=-1)

        pre_sigm = a * x[..., None] + b # Unsqueeze x for broadcasting
        sigm = torch.sigmoid(pre_sigm)
        x_pre = (w * sigm).sum(dim=-1)

        if self.no_logit:
            return x_pre

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        # Rely on x_pre_clipped to prevent exact 0/1 for logit
        # x_pre_log_safe = torch.clamp(x_pre_clipped, min=EPSILON, max=1.0-EPSILON) # Removed clamp
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped) # Use x_pre_clipped directly
        return xnew