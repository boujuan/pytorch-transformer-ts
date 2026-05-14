"""Quantile-head marginal — Phase 0i-G (Path 2).

Drop-in replacement for `DSFMarginal` / `NSFMarginal` in the TACTiS-2 marginal slot.
Instead of a normalizing flow trained with NLL, the conditioner MLP outputs K
predicted quantiles for the target distribution and is trained with **pinball
loss**. Pinball loss has K independent per-quantile terms — none can be jointly
minimised by a delta function, so a model trained this way is structurally
unable to collapse to a near-zero predictive width.

Public interface matches DSFMarginal exactly so the rest of TACTiS-2
(`decoder.py`, `tactis.py`, the copula path, the Stage-2 freeze logic) needs
no plumbing changes beyond the `marginal_flow_type` switch:
  - `forward_logdet(context, x) -> (u, logdet)`     : Stage-1 loss path
  - `forward_no_logdet(context, x) -> u`            : Stage-2 + sampling
  - `inverse(context, u) -> x`                       : sampling

Additionally exposes:
  - `pinball_loss(context, x) -> [B]`                : per-batch pinball loss
    that `decoder.loss` calls when `marginal_flow_type == "quantile"`. The
    return is wired into the existing `marginal_logdet_batch` slot with a
    negated sign so the outer `loss = copula_loss - marginal_logdet` formula
    in `tactis.py:533` becomes `loss = copula_loss + pinball_loss` cleanly.

CDF / ICDF: piecewise-linear interpolation through the K predicted quantile
knots. Adapted from gluonts' `PiecewiseLinear.quantile()` pattern but written
inline here to avoid pulling in the entire gluonts distribution machinery for
one helper. Tail behaviour: for u outside [q_levels[0], q_levels[-1]] (or x
outside [q_pred[0], q_pred[-1]]) we **clip** rather than extrapolate — simpler
and adequate for the standardized data domain.

Crossing fix: pinball loss does NOT guarantee monotonic quantile predictions
(occasionally q_0.05 > q_0.10). We enforce strict monotonicity via the
classical `cumsum(softplus(deltas))` parametrisation: the head emits
(q_min, delta_1, ..., delta_{K-1}); reconstructed q_i = q_min + sum_{j<=i}
softplus(delta_j). This costs ~3 LoC and guarantees ICDF monotonicity by
construction.

Pinball loss math: reused from gluonts' QuantileOutput.loss
(/fs/dss/home/taed7566/Forecasting/gluonts/src/gluonts/torch/distributions/quantile_output.py:51-73):
  2 * mean over q of |(target - q_pred) * ((target <= q_pred).float() - q_level)|
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QuantileMarginal(nn.Module):
    """K-quantile head with pinball loss. See module docstring."""

    def __init__(
        self,
        context_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        quantile_levels: List[float],
        crossing_fix: str = "monotonic_delta",
    ):
        super().__init__()
        if not (1 < len(quantile_levels) <= 64):
            raise ValueError(
                f"quantile_levels must have 2..64 entries, got {len(quantile_levels)}"
            )
        sorted_levels = sorted(quantile_levels)
        if sorted_levels != list(quantile_levels):
            logger.warning(
                "quantile_levels not sorted ascending; auto-sorted to %s", sorted_levels
            )
        for q in sorted_levels:
            if not (0.0 < q < 1.0):
                raise ValueError(f"quantile level {q} must lie strictly in (0,1)")
        if crossing_fix not in {"monotonic_delta", "post_hoc_sort", "none"}:
            raise ValueError(
                f"crossing_fix must be one of monotonic_delta|post_hoc_sort|none, "
                f"got {crossing_fix!r}"
            )

        self.K = len(sorted_levels)
        self.crossing_fix = crossing_fix
        self.register_buffer(
            "q_levels", torch.tensor(sorted_levels, dtype=torch.float32)
        )

        # Conditioner MLP — same shape as DSF/NSF for consistency
        layers = [nn.Linear(context_dim, mlp_dim), nn.ReLU()]
        for _ in range(1, mlp_layers):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, self.K)]
        self.marginal_conditioner = nn.Sequential(*layers)

        self.expected_context_dim = context_dim

    # ------------------------------------------------------------------
    # Quantile prediction with crossing fix
    # ------------------------------------------------------------------

    def _predict_quantiles(self, context: torch.Tensor) -> torch.Tensor:
        """Map context → [..., K] strictly-monotonic quantile predictions.

        Shape contract: if context is [B, N, D] then output is [B, N, K].
        """
        raw = self.marginal_conditioner(context)  # [..., K]

        if self.crossing_fix == "monotonic_delta":
            # raw[..., 0] = q_min; raw[..., 1:] = unconstrained deltas → softplus → cumsum
            base = raw[..., :1]                            # [..., 1]
            deltas = F.softplus(raw[..., 1:])              # [..., K-1], strictly > 0
            cum = torch.cumsum(deltas, dim=-1)             # [..., K-1]
            return torch.cat([base, base + cum], dim=-1)   # [..., K], strictly increasing
        if self.crossing_fix == "post_hoc_sort":
            sorted_q, _ = torch.sort(raw, dim=-1)
            return sorted_q
        return raw  # "none" — caller responsibility

    # ------------------------------------------------------------------
    # Pinball loss (the actual training objective in Stage 1)
    # ------------------------------------------------------------------

    def pinball_loss(
        self, context: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Mean pinball loss across K quantiles, returned per batch element.

        Args:
            context: [B, N, D]
            x:       [B, N]
        Returns:
            [B] — mean pinball over (N, K) for each batch element.
        """
        q_pred = self._predict_quantiles(context)        # [B, N, K]
        x_exp = x.unsqueeze(-1)                          # [B, N, 1]
        diff = x_exp - q_pred                            # [B, N, K]
        levels = self.q_levels.view(1, 1, -1)            # [1, 1, K]
        # gluonts QuantileOutput.loss formula (see file docstring)
        pinball = 2.0 * (
            diff * (levels - (diff < 0).float())
        ).abs()                                          # [B, N, K]
        # Mean over (N, K) → [B]
        return pinball.mean(dim=(-1, -2))

    # ------------------------------------------------------------------
    # Piecewise-linear interpolation helpers (CDF + ICDF)
    # ------------------------------------------------------------------

    @staticmethod
    def _bracket_indices_along_last(
        sorted_values: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """For each query, find the largest i such that sorted_values[..., i] <= query.

        sorted_values: [..., K]
        query:         [...]  (must broadcast against the leading dims of sorted_values)

        Returns: integer indices, same shape as query, in [0, K-2].
        """
        # (q_pred < query) has shape [..., K]; sum along K gives bracket index.
        idx = (sorted_values < query.unsqueeze(-1)).sum(dim=-1) - 1
        K = sorted_values.shape[-1]
        return idx.clamp(0, K - 2)

    def _cdf(self, q_pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear CDF: F(x) given quantile knots q_pred and levels q_levels.

        q_pred: [..., K]; x: [...] (same leading shape as q_pred minus the K dim).
        Returns u ∈ [q_levels[0], q_levels[-1]], shape [...].
        """
        idx = self._bracket_indices_along_last(q_pred, x)         # [...]
        q_lo = torch.gather(q_pred, -1, idx.unsqueeze(-1)).squeeze(-1)
        q_hi = torch.gather(q_pred, -1, (idx + 1).unsqueeze(-1)).squeeze(-1)
        # gather on a 1-D buffer: expand q_levels to broadcast
        lev_lo = self.q_levels[idx]
        lev_hi = self.q_levels[idx + 1]
        denom = (q_hi - q_lo).clamp(min=1e-9)
        u = lev_lo + (lev_hi - lev_lo) * (x - q_lo) / denom
        # Clip to the supported domain [q_levels[0], q_levels[-1]]; do NOT extrapolate.
        return u.clamp(self.q_levels[0].item(), self.q_levels[-1].item())

    def _icdf(self, q_pred: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear inverse-CDF: F^{-1}(u).

        q_pred: [..., K]; u: [...]; returns x with same shape as u.
        For u outside [q_levels[0], q_levels[-1]] we clamp u first.
        """
        u_clamped = u.clamp(self.q_levels[0].item(), self.q_levels[-1].item())
        # Bracket index on q_levels (the support of u), not q_pred.
        # idx s.t. q_levels[idx] <= u_clamped < q_levels[idx+1].
        # q_levels is 1-D, so use searchsorted.
        idx = torch.searchsorted(self.q_levels, u_clamped.contiguous(), right=False) - 1
        K = self.K
        idx = idx.clamp(0, K - 2)
        q_lo = torch.gather(q_pred, -1, idx.unsqueeze(-1)).squeeze(-1)
        q_hi = torch.gather(q_pred, -1, (idx + 1).unsqueeze(-1)).squeeze(-1)
        lev_lo = self.q_levels[idx]
        lev_hi = self.q_levels[idx + 1]
        denom = (lev_hi - lev_lo).clamp(min=1e-9)
        return q_lo + (q_hi - q_lo) * (u_clamped - lev_lo) / denom

    def _slope_logdet(
        self, q_pred: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """log |dF/dx| at each (b, n). Slope of the piecewise-linear CDF is
        (lev_hi - lev_lo) / (q_hi - q_lo) in the bracket containing x.

        q_pred: [B, N, K]; x: [B, N]
        Returns: [B] — sum of log-slope over N.
        """
        idx = self._bracket_indices_along_last(q_pred, x)         # [B, N]
        q_lo = torch.gather(q_pred, -1, idx.unsqueeze(-1)).squeeze(-1)
        q_hi = torch.gather(q_pred, -1, (idx + 1).unsqueeze(-1)).squeeze(-1)
        lev_lo = self.q_levels[idx]
        lev_hi = self.q_levels[idx + 1]
        slope = (lev_hi - lev_lo) / (q_hi - q_lo).clamp(min=1e-9)
        log_slope = torch.log(slope.clamp(min=1e-30))
        # Sum over N → [B]
        return log_slope.sum(dim=tuple(range(1, log_slope.dim())))

    # ------------------------------------------------------------------
    # Public interface methods (mirror DSFMarginal exactly)
    # ------------------------------------------------------------------

    def forward_logdet(
        self, context: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform x → u via piecewise-linear CDF; return (u, logdet).

        NOTE: in quantile mode the loss path uses `pinball_loss` directly,
        NOT this method. This method exists for interface compatibility (some
        downstream code may still call it, e.g. validation-time metric
        callbacks). The returned logdet is the true log-slope of the
        piecewise-linear CDF — usable but never the training signal.
        """
        if context.shape[-1] != self.expected_context_dim:
            logger.warning(
                "Context dim %d != expected %d (QuantileMarginal). Check calling code.",
                context.shape[-1],
                self.expected_context_dim,
            )
        q_pred = self._predict_quantiles(context)                  # [B, N, K]
        u = self._cdf(q_pred, x)                                   # [B, N]
        logdet = self._slope_logdet(q_pred, x)                     # [B]
        return u, logdet

    def forward_no_logdet(
        self, context: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Transform x → u without logdet. Supports optional sample axis on x."""
        if context.shape[-1] != self.expected_context_dim:
            logger.warning(
                "Context dim %d != expected %d (QuantileMarginal). Check calling code.",
                context.shape[-1],
                self.expected_context_dim,
            )
        q_pred = self._predict_quantiles(context)                  # [B, N, K]
        # If x has an extra sample axis (B, N, S) vs q_pred (B, N, K), expand q_pred
        if q_pred.dim() == x.dim():
            S = x.shape[-1]
            q_pred = q_pred.unsqueeze(2).expand(-1, -1, S, -1)     # [B, N, S, K]
        return self._cdf(q_pred, x)

    def inverse(
        self,
        context: torch.Tensor,
        u: torch.Tensor,
        max_iter: int = 100,
        precision: float = 1e-6,
    ) -> torch.Tensor:
        """Piecewise-linear inverse CDF. `max_iter`/`precision` accepted for
        API compat with DSFMarginal but ignored (closed-form)."""
        if context.shape[-1] != self.expected_context_dim:
            logger.warning(
                "Context dim %d != expected %d (QuantileMarginal). Check calling code.",
                context.shape[-1],
                self.expected_context_dim,
            )
        q_pred = self._predict_quantiles(context)                  # [B, N, K]
        if q_pred.dim() == u.dim():
            S = u.shape[-1]
            q_pred = q_pred.unsqueeze(2).expand(-1, -1, S, -1)     # [B, N, S, K]
        return self._icdf(q_pred, u)
