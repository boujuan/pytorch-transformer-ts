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

def smooth_a_cap(a_raw: torch.Tensor, a_max: float) -> torch.Tensor:
    """
    Smooth (rational) cap on the sigmoid-steepness parameter `a`.

    Maps R+ -> (0, a_max) via `a = a_raw / (1 + a_raw / a_max)`.
    - Healthy regime (a_raw << a_max): a ≈ a_raw (essentially identity)
    - Pathological regime (a_raw >> a_max): a -> a_max (asymptote)
    - Gradient is non-zero everywhere (unlike `clamp(., max=a_max)` which has zero
      gradient above the cap and would trap the optimizer).

    Used as defense-in-depth (Fix B) against marginal-flow collapse where DSF's
    `a` parameter exploded to ~720 in the broken epoch=122 model. With a_max=20,
    the broken regime is bounded but the healthy regime (a typically < 5) is
    barely affected. See plan: /user/taed7566/.claude/plans/moonlit-sprouting-gosling.md

    Set a_max=0 (or any non-positive) to disable capping (returns a_raw unchanged).
    """
    if a_max <= 0:
        return a_raw
    return a_raw / (1.0 + a_raw / a_max)


class SigmoidFlow(nn.Module):
    """
    A single layer of the Deep Sigmoid Flow network.
    """
    def __init__(self, hidden_dim: int, no_logit: bool = False, a_max: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.no_logit = no_logit
        # a_max=0 disables the smooth cap (default: backward-compatible / no-op).
        # Set to a positive value (e.g., 20.0) via DeepSigmoidFlow / hparam to enable Fix B.
        self.a_max = a_max
        # Buffer for per-(batch, vars) log-density (logj after log_sum_exp over h).
        # Populated by forward() during training when accessed by the regularizer in
        # lightning_module.training_step. None when not training or after reset.
        self._last_logj_per_datapoint: torch.Tensor = None

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

        # Fix B: optional smooth cap on `a` to bound max sigmoid steepness.
        a_raw = torch.nn.functional.softplus(params[..., : self.hidden_dim])  # b, v, h
        a = smooth_a_cap(a_raw, self.a_max) + EPSILON  # b, v, h
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
        # Fix A: expose per-(batch, vars) log-density for external regularization.
        # This is the actual quantity that runs to infinity during flow collapse.
        # Penalizing this scales correctly with batch size (unlike per-parameter
        # penalties on a_pre, which were 1000x weaker than the NLL pull).
        # Saved as a tensor reference (gradient-attached) so the regularizer
        # contributes to the optimizer's gradients.
        self._last_logj_per_datapoint = logj

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
        # Fix B: keep the smooth cap consistent with forward() so inference matches training.
        a_raw = torch.nn.functional.softplus(params[..., :self.hidden_dim])
        a = smooth_a_cap(a_raw, self.a_max) + EPSILON
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