"""Proper scoring rules for TACTiS-2 training (Phase 0i-D / Fix Sd).

Why this file exists
--------------------
After Phase 0i-C pilots Sa (a_floor) and Sw (w_entropy) BOTH mechanically fixed
their target parameter pathologies but neither widened the predictive
distribution F^-1(0.95) - F^-1(0.05) beyond ~0.27 standardized units (vs ~3.3
healthy). The pilots proved the bottleneck is the training OBJECTIVE, not the
parameters. Pure NLL training rewards "concentrate density at training points"
— with highly autoregressive 15-s wind data each context maps near-
deterministically to the next value, so NLL collapses to a sharp spike at the
single training observation. The model has no signal during training that
the TRUE conditional distribution is wide.

This module implements multivariate proper scoring rules whose minimum is
attained ONLY when the predictive distribution matches the true conditional
distribution — they penalize "narrow but lucky" predictions just as much as
"wide but misplaced" ones. References:

- Gneiting & Raftery (2007) — Strictly Proper Scoring Rules, JASA. Foundational
  reference; defines ES and CRPS as strictly proper for continuous distributions.
- Scheuerer & Hamill (2015) — Variogram-Based Proper Scoring Rules, MWR.
  Adds VS to catch correlation errors that ES is weakly sensitive to.
- Gneiting & Katzfuss (2014) — Probabilistic Forecasting, Annual Review of
  Statistics. Practical recommendations for choosing scores for training.

Implementation notes
--------------------
- All three rules below are torch-native (no numpy random); gradients flow
  through samples drawn via `tactis.sample()` for backprop into the encoder
  and marginal flow.
- The reference numpy ES at `stage2_metrics.py:329-380` (used in the existing
  validation callback) cannot be reused for training because `np.random.choice`
  breaks gradient flow.
- Vectorized via `torch.cdist` for ES/VS — no Python loops over samples.
- `mvg_crps_closedform` is currently a stub; will be fleshed out only if the
  E0-E4 sample-based pilot fails (auto-fallback per plan).
"""

from __future__ import annotations

import torch
from torch import Tensor


def energy_score(
    samples: Tensor,
    target: Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Multivariate Energy Score (Gneiting & Raftery 2007, Eq. 21-22).

    ES(F, y) = E_X ||X - y||^beta  -  0.5 * E_{X, X'} ||X - X'||^beta

    where X, X' are independent samples from the predictive distribution F
    and y is the truth. Strictly proper for continuous multivariate
    distributions when beta in (0, 2). Default beta=1 gives the standard ES.

    Parameters
    ----------
    samples : Tensor of shape [N, B, D]
        N samples per batch element, each of dimension D. Must require_grad
        if used as a loss; otherwise pass detached for evaluation.
    target : Tensor of shape [B, D]
        Truth values; one per batch element.
    beta : float, default 1.0
        Distance exponent. beta=1 is standard ES; beta in (0, 2) for strict
        propriety; beta=2 reduces to comparing means/variances and is NOT
        strictly proper.
    eps : float, default 1e-8
        Numerical stabilizer to avoid sqrt(0) backward-pass NaNs when
        sample == target exactly.

    Returns
    -------
    Tensor of shape [B]
        Per-batch-element energy score. Mean to scalar before adding to loss.

    Notes
    -----
    Compute is O(B * N^2 * D) for the pairwise term, dominated by torch.cdist.
    For B=2048, N=16, D=8 (per-turbine pred_len * series), this is ~4M flops
    per batch — negligible vs the rest of the model forward pass.
    """
    if samples.dim() != 3:
        raise ValueError(
            f"samples must be [N, B, D]; got shape {tuple(samples.shape)}"
        )
    if target.dim() != 2:
        raise ValueError(
            f"target must be [B, D]; got shape {tuple(target.shape)}"
        )
    n, b, d = samples.shape
    if target.shape != (b, d):
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match "
            f"samples shape ({b}, {d}) implied dims"
        )

    # Term 1: E_X ||X - y||^beta
    # samples [N, B, D], target [B, D] -> [N, B, D] - [1, B, D] = [N, B, D]
    diffs_to_target = samples - target.unsqueeze(0)  # [N, B, D]
    norms_to_target = (diffs_to_target.pow(2).sum(dim=-1) + eps).sqrt()  # [N, B]
    if beta != 1.0:
        norms_to_target = norms_to_target.pow(beta)
    term1 = norms_to_target.mean(dim=0)  # [B]

    # Term 2: 0.5 * E_{X, X'} ||X - X'||^beta via torch.cdist
    # Permute to [B, N, D] for cdist; pairwise distances [B, N, N].
    samples_bnd = samples.transpose(0, 1).contiguous()  # [B, N, D]
    pairwise = torch.cdist(samples_bnd, samples_bnd, p=2)  # [B, N, N]
    if beta != 1.0:
        pairwise = pairwise.pow(beta)
    # Off-diagonal mean: include all N^2 pairs (diagonal zeros are fine — bias
    # is O(1/N), and consistent across the batch). This matches Gneiting's
    # standard finite-sample estimator.
    term2 = 0.5 * pairwise.mean(dim=(1, 2))  # [B]

    return term1 - term2


def variogram_score(
    samples: Tensor,
    target: Tensor,
    p: float = 0.5,
    weights: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Multivariate Variogram Score of order p (Scheuerer & Hamill 2015, Eq. 3).

    VS_p(F, y) = sum_{i, j} w_{ij} * ( |y_i - y_j|^p - E_X |X_i - X_j|^p )^2

    Penalizes pairwise variable-difference mismatches between the predictive
    distribution and the truth. Crucially, VS catches correlation errors that
    ES is weakly sensitive to. Strictly proper for p in (0, 2). p=0.5 is
    Scheuerer & Hamill's standard recommendation.

    Parameters
    ----------
    samples : Tensor of shape [N, B, D]
    target : Tensor of shape [B, D]
    p : float, default 0.5
        Order of the variogram. p=0.5 gives smaller penalties to large
        differences and avoids over-emphasis on outliers.
    weights : Tensor of shape [D, D] or None
        Optional per-pair weights; defaults to uniform (off-diagonal-only)
        and excludes diagonal (i == j).
    eps : float, default 1e-8
        Numerical stabilizer for the |.|^p computation.

    Returns
    -------
    Tensor of shape [B]
        Per-batch-element variogram score.

    Notes
    -----
    Compute is O(B * N * D^2). For D=8 this is trivial; for D=704 (the full
    multivariate forecast) this becomes O(B * N * 500k), still tractable for
    N=16 and modest B.
    """
    if samples.dim() != 3 or target.dim() != 2:
        raise ValueError(
            f"shapes must be samples=[N, B, D], target=[B, D]; got "
            f"{tuple(samples.shape)} and {tuple(target.shape)}"
        )
    n, b, d = samples.shape

    # Pairwise absolute differences (raised to power p) for samples and target.
    # samples [N, B, D] -> [N, B, D, 1] - [N, B, 1, D] = [N, B, D, D]
    samp_pairs = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs()  # [N, B, D, D]
    samp_pairs = (samp_pairs + eps).pow(p)
    samp_expectation = samp_pairs.mean(dim=0)  # [B, D, D]

    target_pairs = (target.unsqueeze(-1) - target.unsqueeze(-2)).abs()  # [B, D, D]
    target_pairs = (target_pairs + eps).pow(p)

    sq_residuals = (target_pairs - samp_expectation).pow(2)  # [B, D, D]

    # Default: uniform off-diagonal weights (mask diagonal where i == j).
    if weights is None:
        mask = (1.0 - torch.eye(d, device=samples.device, dtype=samples.dtype))
        sq_residuals = sq_residuals * mask
        # Mean over the (D*(D-1)) off-diagonal pairs:
        return sq_residuals.sum(dim=(1, 2)) / (d * (d - 1))

    if weights.shape != (d, d):
        raise ValueError(
            f"weights shape {tuple(weights.shape)} does not match D={d}"
        )
    return (sq_residuals * weights).sum(dim=(1, 2))


def mvg_crps_closedform(
    mean: Tensor,
    cov: Tensor,
    target: Tensor,
) -> Tensor:
    """Multivariate Gaussian CRPS (closed-form) — fallback path stub.

    To be fleshed out only if the sample-based ES/VS pilot fails. The closed
    form for a Gaussian-marginal forecast (after Cholesky-whitening to
    independent components) is computed without sampling, so gradient flow
    is trivially stable and there is nothing to tune for sample count.

    Reference: arXiv:2410.09133 (MVG-CRPS, 2024) — outperforms NLL and ES on
    several multivariate forecasting benchmarks when the marginals are
    approximately Gaussian.

    Parameters
    ----------
    mean : Tensor [B, D]
    cov  : Tensor [B, D, D]
    target : Tensor [B, D]

    Returns
    -------
    Tensor [B]
    """
    raise NotImplementedError(
        "mvg_crps_closedform is a fallback stub. Implement only if the "
        "Phase 0i-D E0-E4 pilot fails to widen F^-1 (auto-trigger per plan)."
    )
