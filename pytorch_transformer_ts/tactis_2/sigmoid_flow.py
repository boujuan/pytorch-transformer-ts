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
        # --- Roo Debug ---
        # Select a single element to log detailed values for (e.g., batch 0, item 0, sample 0)
        log_idx = (0, 0, 0) if x.dim() == 3 else (0, 0) # Adjust index based on input dims
        log_hidden_idx = 0 # Log first hidden unit details
        try:
            print(f"DEBUG (SigmoidFlow.fwd_no_logdet): Element {log_idx} Input x: {x[log_idx].item():.6f}")
            if torch.isnan(x[log_idx]): print("  WARNING: NaN in input x!")
        except IndexError:
            print(f"DEBUG (SigmoidFlow.fwd_no_logdet): Cannot log element {log_idx}, tensor shape {x.shape} might be smaller.")
        except Exception as e:
            print(f"DEBUG (SigmoidFlow.fwd_no_logdet): Error logging input element {log_idx}: {e}")
        # ---------------
        a = torch.nn.functional.softplus(params[..., :self.hidden_dim]) + EPSILON
        b = params[..., self.hidden_dim:2*self.hidden_dim]
        w = torch.nn.functional.softmax(params[..., 2*self.hidden_dim:], dim=-1)
        # --- Roo Debug ---
        try:
            print(f"  Element {log_idx}, Hidden {log_hidden_idx}: a={a[log_idx][log_hidden_idx].item():.6f}, b={b[log_idx][log_hidden_idx].item():.6f}, w={w[log_idx][log_hidden_idx].item():.6f}")
            if torch.isnan(a[log_idx][log_hidden_idx]): print("    WARNING: NaN in a!")
            if torch.isnan(b[log_idx][log_hidden_idx]): print("    WARNING: NaN in b!")
            if torch.isnan(w[log_idx][log_hidden_idx]): print("    WARNING: NaN in w!")
        except IndexError:
             print(f"  Cannot log hidden element ({log_idx[0]},{log_idx[1]},{log_hidden_idx}), tensor shapes might be smaller (a:{a.shape}, b:{b.shape}, w:{w.shape}).")
        except Exception as e:
             print(f"  Error logging a/b/w element: {e}")
        # ---------------

        pre_sigm = a * x[..., None] + b # Unsqueeze x for broadcasting
        # --- Roo Debug ---
        try:
            print(f"  Element {log_idx}, Hidden {log_hidden_idx}: pre_sigm={pre_sigm[log_idx][log_hidden_idx].item():.6f}")
            if torch.isnan(pre_sigm[log_idx][log_hidden_idx]): print("    WARNING: NaN in pre_sigm!")
        except IndexError:
             print(f"  Cannot log pre_sigm hidden element ({log_idx[0]},{log_idx[1]},{log_hidden_idx}), tensor shape {pre_sigm.shape} might be smaller.")
        except Exception as e:
             print(f"  Error logging pre_sigm element: {e}")
        # ---------------
        sigm = torch.sigmoid(pre_sigm)
        # --- Roo Debug ---
        try:
            print(f"  Element {log_idx}, Hidden {log_hidden_idx}: sigm={sigm[log_idx][log_hidden_idx].item():.6f}")
            if torch.isnan(sigm[log_idx][log_hidden_idx]): print("    WARNING: NaN in sigm!")
        except IndexError:
             print(f"  Cannot log sigm hidden element ({log_idx[0]},{log_idx[1]},{log_hidden_idx}), tensor shape {sigm.shape} might be smaller.")
        except Exception as e:
             print(f"  Error logging sigm element: {e}")
        # ---------------
        x_pre = (w * sigm).sum(dim=-1)
        # --- Roo Debug ---
        try:
            print(f"  Element {log_idx}: x_pre={x_pre[log_idx].item():.6f}")
            if torch.isnan(x_pre[log_idx]): print("    WARNING: NaN in x_pre!")
        except IndexError:
             print(f"  Cannot log x_pre element {log_idx}, tensor shape {x_pre.shape} might be smaller.")
        except Exception as e:
             print(f"  Error logging x_pre element: {e}")
        # ---------------

        if self.no_logit:
            print(f"DEBUG (SigmoidFlow.fwd_no_logdet): Element {log_idx} Returning x_pre (no_logit=True)") # Roo Debug
            return x_pre

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        # --- Roo Debug ---
        try:
            print(f"  Element {log_idx}: x_pre_clipped={x_pre_clipped[log_idx].item():.6f}")
            if torch.isnan(x_pre_clipped[log_idx]): print("    WARNING: NaN in x_pre_clipped!")
        except IndexError:
             print(f"  Cannot log x_pre_clipped element {log_idx}, tensor shape {x_pre_clipped.shape} might be smaller.")
        except Exception as e:
             print(f"  Error logging x_pre_clipped element: {e}")
        # ---------------
        # Rely on x_pre_clipped to prevent exact 0/1 for logit
        # x_pre_log_safe = torch.clamp(x_pre_clipped, min=EPSILON, max=1.0-EPSILON) # Removed clamp
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped) # Use x_pre_clipped directly
        # --- Roo Debug ---
        try:
            print(f"DEBUG (SigmoidFlow.fwd_no_logdet): Element {log_idx} Final xnew: {xnew[log_idx].item():.6f}")
            if torch.isnan(xnew[log_idx]): print("  WARNING: NaN in final xnew!")
        except IndexError:
             print(f"DEBUG (SigmoidFlow.fwd_no_logdet): Cannot log final xnew element {log_idx}, tensor shape {xnew.shape} might be smaller.")
        except Exception as e:
             print(f"DEBUG (SigmoidFlow.fwd_no_logdet): Error logging final xnew element: {e}")
        # ---------------

        return xnew