"""Neural Spline Flow (NSF) marginal — Phase 0i-E.

Drop-in replacement for `DSFMarginal` (in this same package). The DSF version,
trained with NLL on highly autoregressive 15-s wind data, repeatedly converges
to a collapsed-CDF regime where the inverse-DSF Jacobian is small. Empirical
probing in Phases 0i-C and 0i-D showed that ALL parameter-level constraints
(Sa: a-floor; Sw: w-entropy) and ALL sample-based scoring rules (ES, VS,
ES+VS, ES+trajectory-noise) failed to widen F^-1(0.95) - F^-1(0.05) beyond
~0.27 standardized units. The pathology is structural: DSF's softmax mixture
of sigmoids permits arbitrary CDF flatness, and that flatness attenuates the
gradient flowing back through `inverse(...)` from any sample-based loss.

This file uses Neural Spline Flow (Durkan, Bekasov, Murray, Papamakarios 2019,
arXiv:1906.04032) which parametrizes a strictly-monotone CDF as a chain of
rational-quadratic splines with `min_derivative > 0` enforced everywhere. The
collapse pattern that traps DSF cannot occur in NSF's parameter space — every
spline piece has slope ≥ min_derivative by construction.

Reuses `nflows.transforms.splines.rational_quadratic.unconstrained_rational_quadratic_spline`
(MIT, pure PyTorch, pinned to nflows>=0.14).

Public interface matches `DSFMarginal`:
- `forward_logdet(context, x) -> (transformed_x, logdet)`  : training loss path
- `forward_no_logdet(context, x) -> transformed_x`         : inference + Sd sampling
- `inverse(context, u) -> x`                                : sampling

Differences vs DSFMarginal worth noting:
- NSF inverse is ANALYTIC (no binary search). Faster sampling.
- NSF outputs are bounded: outside [-tail_bound, tail_bound] the flow is the
  identity (`tails="linear"`). For standardized data, tail_bound=4.0 covers
  99.99% of N(0, 1).
- Diagnostic buffer `_last_min_derivative_check` reports whether the
  min_derivative floor binds for any (batch, var) — useful for confirming
  the structural anti-collapse property is engaged.
"""
from __future__ import annotations
import logging
from typing import Tuple, List
import torch
from torch import nn

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)

logger = logging.getLogger(__name__)


class _NSFFlowStack(nn.Module):
    """A stack of `num_flow_layers` rational-quadratic spline transforms,
    composed end-to-end. Each layer reads its params from a contiguous slice
    of the conditioner-MLP output. Mirrors `DeepSigmoidFlow`'s role.

    Per-layer params: (num_bins widths) + (num_bins heights) + (num_bins-1
    interior derivatives) = 3*num_bins - 1.
    """

    def __init__(
        self,
        num_flow_layers: int,
        num_bins: int,
        tail_bound: float,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.num_flow_layers = num_flow_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.params_per_layer = 3 * num_bins - 1
        self.total_params_length = num_flow_layers * self.params_per_layer

        # Diagnostic buffer (populated in forward; read by lightning_module
        # if/when an NSF-specific health check is wired). Empty by default.
        self._last_min_derivative_check: List[float] = []

    def _slice_params(self, params: torch.Tensor, layer_idx: int):
        """Slice the conditioner output into (widths, heights, derivatives)
        for the layer_idx-th layer.

        Returns three tensors with shape [..., num_bins], [..., num_bins],
        [..., num_bins - 1] respectively.
        """
        K = self.num_bins
        ofs = layer_idx * self.params_per_layer
        widths = params[..., ofs : ofs + K]
        heights = params[..., ofs + K : ofs + 2 * K]
        derivatives = params[..., ofs + 2 * K : ofs + 3 * K - 1]
        return widths, heights, derivatives

    def forward(
        self, params: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the L spline layers in order, accumulating logdet.

        Inputs:
          params: [..., total_params_length]  (the conditioner output)
          x:      [...]                        (the data, broadcastable to params except last dim)
        Returns:
          transformed_x: [...] same leading shape as x
          logdet: shape (batch,) — summed over all leading dims except batch
        """
        # logdet accumulator on batch dim (consistent with DSF's behavior)
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        self._last_min_derivative_check = []

        for li in range(self.num_flow_layers):
            widths, heights, derivatives = self._slice_params(params, li)
            x, ld_per_elem = unconstrained_rational_quadratic_spline(
                inputs=x,
                unnormalized_widths=widths,
                unnormalized_heights=heights,
                unnormalized_derivatives=derivatives,
                inverse=False,
                tails="linear",
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            # Sum per-element logdet over all non-batch dims to match DSF's [B] return shape.
            extra_dims = tuple(range(1, ld_per_elem.dim()))
            logdet = logdet + ld_per_elem.sum(dim=extra_dims) if extra_dims else logdet + ld_per_elem
            # Cheap diagnostic — softplus(derivatives) + min_derivative is the actual slope
            # at every internal knot. We record the minimum across this layer.
            with torch.no_grad():
                slope_min = (
                    torch.nn.functional.softplus(derivatives) + self.min_derivative
                ).min().item()
            self._last_min_derivative_check.append(slope_min)

        return x, logdet

    def forward_no_logdet(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward without logdet — used during inference / Sd sample loop."""
        for li in range(self.num_flow_layers):
            widths, heights, derivatives = self._slice_params(params, li)
            x, _ = unconstrained_rational_quadratic_spline(
                inputs=x,
                unnormalized_widths=widths,
                unnormalized_heights=heights,
                unnormalized_derivatives=derivatives,
                inverse=False,
                tails="linear",
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
        return x

    def inverse(self, params: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Analytic inverse — apply layers in REVERSE order with `inverse=True`."""
        x = u
        for li in reversed(range(self.num_flow_layers)):
            widths, heights, derivatives = self._slice_params(params, li)
            x, _ = unconstrained_rational_quadratic_spline(
                inputs=x,
                unnormalized_widths=widths,
                unnormalized_heights=heights,
                unnormalized_derivatives=derivatives,
                inverse=True,
                tails="linear",
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
        return x


class NSFMarginal(nn.Module):
    """Neural Spline Flow marginal — drop-in replacement for `DSFMarginal`.

    Public surface (signatures and shape contracts) is identical to DSFMarginal:
    - `forward_logdet(context: [B, N, dim], x: [B, N]) -> ([B, N], [B])`
    - `forward_no_logdet(context: [B, N, dim], x: [B, N] or [B, N, S]) -> same shape as x`
    - `inverse(context: [B, N, dim], u: [B, N] or [B, N, S]) -> same shape as u`

    Constructor differs: NSF needs (num_bins, tail_bound, min_derivative) instead
    of DSF's (flow_hid_dim). flow_layers maps to num_flow_layers.
    """

    def __init__(
        self,
        context_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        num_flow_layers: int = 2,
        num_bins: int = 32,
        tail_bound: float = 4.0,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.marginal_flow = _NSFFlowStack(
            num_flow_layers=num_flow_layers,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_derivative=min_derivative,
        )

        # Conditioner MLP: same shape as DSFMarginal's. Output dim = total_params_length.
        layers = [nn.Linear(context_dim, mlp_dim), nn.ReLU()]
        for _ in range(1, mlp_layers):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, self.marginal_flow.total_params_length)]
        self.marginal_conditioner = nn.Sequential(*layers)

        self.expected_context_dim = context_dim

    def forward_logdet(
        self, context: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.shape[-1] != self.expected_context_dim:
            logger.warning(
                f"Context dim {context.shape[-1]} != expected "
                f"{self.expected_context_dim}. Check calling code."
            )
        marginal_params = self.marginal_conditioner(context)  # [B, N, P]
        # _NSFFlowStack.forward returns transformed_x of shape [B, N], logdet of shape [B]
        transformed_x, logdet = self.marginal_flow.forward(marginal_params, x)
        return transformed_x, logdet

    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] != self.expected_context_dim:
            logger.warning(
                f"Context dim {context.shape[-1]} != expected "
                f"{self.expected_context_dim}. Check calling code."
            )
        marginal_params = self.marginal_conditioner(context)  # [B, N, P]
        # Match DSF semantics: if x has an extra sample axis (B, N, S) while
        # params is (B, N, P), explicitly EXPAND params to (B, N, S, P) so that
        # nflows' rational-quadratic-spline (which requires exact leading-dim
        # match between inputs and widths/heights/derivatives) can broadcast.
        # DSF achieved this implicitly via sigmoid arithmetic; NSF needs it
        # done at the entry point.
        if marginal_params.dim() == x.dim():
            S = x.shape[-1]
            marginal_params = marginal_params.unsqueeze(2).expand(-1, -1, S, -1)
        return self.marginal_flow.forward_no_logdet(marginal_params, x)

    def inverse(
        self,
        context: torch.Tensor,
        u: torch.Tensor,
        max_iter: int = 100,
        precision: float = 1e-6,
    ) -> torch.Tensor:
        """Analytic inverse via NSF (no binary search needed).

        `max_iter` and `precision` are accepted for API compatibility with
        DSFMarginal but ignored — NSF's inverse is closed-form to machine
        precision (typical round-trip error ≈ 1e-6).
        """
        if context.shape[-1] != self.expected_context_dim:
            logger.warning(
                f"Context dim {context.shape[-1]} != expected "
                f"{self.expected_context_dim}. Check calling code."
            )
        marginal_params = self.marginal_conditioner(context)
        # Same expansion pattern as forward_no_logdet — see comment there.
        if marginal_params.dim() == u.dim():
            S = u.shape[-1]
            marginal_params = marginal_params.unsqueeze(2).expand(-1, -1, S, -1)
        return self.marginal_flow.inverse(marginal_params, u)
