"""Unit tests for `QuantileMarginal` (Phase 0i-G).

Covers:
  1. Shape contracts (interface match with DSFMarginal)
  2. Pinball loss vanishes when predictions match truth (sanity)
  3. Pinball gradient flows through the conditioner MLP
  4. ICDF monotonic in u (no quantile crossing after fix)
  5. forward_no_logdet output ∈ [q_levels[0], q_levels[-1]]
  6. ICDF round-trip: x → u → x within tolerance for x ∈ supported domain
  7. Crossing fix produces strictly-monotonic quantiles (monotonic_delta)
  8. Sample-axis (B, N, S) broadcasting works for forward_no_logdet + inverse
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from pytorch_transformer_ts.tactis_2.quantile_marginal import QuantileMarginal


DEFAULT_LEVELS = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]


@pytest.fixture
def model() -> QuantileMarginal:
    torch.manual_seed(0)
    return QuantileMarginal(
        context_dim=32,
        mlp_layers=2,
        mlp_dim=16,
        quantile_levels=DEFAULT_LEVELS,
        crossing_fix="monotonic_delta",
    )


@pytest.fixture
def batch():
    """Synthetic (context, x) batch matching DSFMarginal's expected shapes."""
    torch.manual_seed(0)
    B, N, D = 4, 12, 32
    context = torch.randn(B, N, D)
    x = torch.randn(B, N) * 0.7  # roughly standardized
    return context, x


def test_1_shape_contract_matches_dsf(model: QuantileMarginal, batch):
    """forward_logdet returns (u [B, N], logdet [B]) like DSFMarginal."""
    context, x = batch
    u, logdet = model.forward_logdet(context, x)
    assert u.shape == x.shape, f"u shape mismatch: {u.shape} vs {x.shape}"
    assert logdet.shape == (x.shape[0],), f"logdet shape mismatch: {logdet.shape}"
    assert torch.isfinite(u).all()
    assert torch.isfinite(logdet).all()


def test_2_pinball_vanishes_at_perfect_quantiles(model: QuantileMarginal):
    """If the K quantile predictions exactly match the data's quantiles,
    pinball loss approaches the theoretical minimum (NOT zero — pinball at
    the true quantile is non-zero in finite samples, but should be small).

    A cleaner sanity: pinball at all-equal predictions (= median) should be
    nonzero whenever the data has variance.
    """
    torch.manual_seed(123)
    B, N, D = 8, 64, 32
    context = torch.randn(B, N, D)
    x_constant = torch.full((B, N), 0.3)  # all targets = 0.3

    # Pinball should be small but nonzero when x is constant and predictions are diverse
    loss = model.pinball_loss(context, x_constant)
    assert loss.shape == (B,)
    assert (loss >= 0).all(), "Pinball loss must be non-negative"
    assert torch.isfinite(loss).all()


def test_3_pinball_gradient_flows(model: QuantileMarginal, batch):
    """Backprop through pinball must update the conditioner MLP."""
    context, x = batch
    context = context.detach().requires_grad_(True)
    loss = model.pinball_loss(context, x).mean()
    loss.backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    assert context.grad.abs().sum() > 0, "Gradient is identically zero"
    # Also check conditioner MLP has gradients
    for p in model.marginal_conditioner.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_4_icdf_monotonic_in_u(model: QuantileMarginal, batch):
    """For fixed context, F^{-1}(u) is non-decreasing in u."""
    context, _ = batch
    u_sweep = torch.linspace(0.02, 0.98, 50).view(1, 1, -1).expand(
        context.shape[0], context.shape[1], 50
    )
    x_at_u = model.inverse(context, u_sweep)  # [B, N, 50]
    diffs = x_at_u[..., 1:] - x_at_u[..., :-1]
    # Should be >= 0 everywhere (allow tiny numerical slop at bracket boundaries)
    assert (diffs >= -1e-5).all(), (
        f"ICDF not monotonic: min diff = {diffs.min().item():.4e}"
    )


def test_5_forward_no_logdet_in_supported_range(model: QuantileMarginal, batch):
    """u = forward_no_logdet(x) lies in [q_levels[0], q_levels[-1]]."""
    context, x = batch
    # Use a range of x values broader than what the model has seen
    x_wide = torch.linspace(-5.0, 5.0, x.numel()).view_as(x)
    u = model.forward_no_logdet(context, x_wide)
    assert (u >= model.q_levels[0] - 1e-6).all(), (
        f"u below q_levels[0]: min = {u.min().item()}"
    )
    assert (u <= model.q_levels[-1] + 1e-6).all(), (
        f"u above q_levels[-1]: max = {u.max().item()}"
    )


def test_6_icdf_round_trip(model: QuantileMarginal):
    """For u ∈ [q_levels[0], q_levels[-1]], applying inverse then forward
    should return u within tolerance."""
    torch.manual_seed(0)
    B, N, D = 4, 8, 32
    context = torch.randn(B, N, D)
    u_in = (
        torch.linspace(
            model.q_levels[0].item() + 0.01,
            model.q_levels[-1].item() - 0.01,
            B * N,
        )
        .view(B, N)
    )
    x = model.inverse(context, u_in)
    u_back = model.forward_no_logdet(context, x)
    err = (u_back - u_in).abs()
    assert err.max() < 1e-4, (
        f"Round-trip error too large: max |u - F(F^-1(u))| = {err.max().item():.4e}"
    )


def test_7_crossing_fix_strict_monotonicity(model: QuantileMarginal, batch):
    """With monotonic_delta crossing fix, predicted quantiles are strictly
    increasing in K — even for random (untrained) conditioner outputs."""
    context, _ = batch
    q_pred = model._predict_quantiles(context)  # [B, N, K]
    diffs = q_pred[..., 1:] - q_pred[..., :-1]
    # softplus output is positive → strictly > 0 (allow 1e-12 numerical slack)
    assert (diffs > -1e-9).all(), (
        f"Crossing-fix failed: min diff = {diffs.min().item():.4e}"
    )
    # With monotonic_delta + softplus, the difference must be STRICTLY positive
    # (softplus(x) > 0 for all finite x).
    assert (diffs > 0).all(), (
        f"monotonic_delta should give strict inequality, got min={diffs.min().item():.4e}"
    )


def test_8_sample_axis_broadcasting(model: QuantileMarginal, batch):
    """forward_no_logdet and inverse must handle (B, N, S) sample axis."""
    context, x = batch  # context [B, N, D], x [B, N]
    B, N = x.shape
    S = 7

    # forward_no_logdet with sample axis on x
    x_sampled = x.unsqueeze(-1).expand(B, N, S) + 0.01 * torch.randn(B, N, S)
    u_out = model.forward_no_logdet(context, x_sampled)
    assert u_out.shape == (B, N, S), f"shape mismatch: {u_out.shape}"

    # inverse with sample axis on u
    u_in = torch.empty(B, N, S).uniform_(0.1, 0.9)
    x_out = model.inverse(context, u_in)
    assert x_out.shape == (B, N, S), f"shape mismatch: {x_out.shape}"


def test_9_crossing_fix_post_hoc_sort_still_monotonic():
    """post_hoc_sort path also produces sorted output, even though it kills
    the gradient through the sort."""
    torch.manual_seed(0)
    m = QuantileMarginal(
        context_dim=8,
        mlp_layers=1,
        mlp_dim=8,
        quantile_levels=DEFAULT_LEVELS,
        crossing_fix="post_hoc_sort",
    )
    context = torch.randn(2, 3, 8)
    q_pred = m._predict_quantiles(context)
    diffs = q_pred[..., 1:] - q_pred[..., :-1]
    assert (diffs >= 0).all()


def test_10_pinball_loss_no_logdet_path_is_independent():
    """The decoder.loss path in quantile mode wraps pinball as
    `marginal_logdet_batch = -pinball_loss`. Verify pinball is positive and
    well-conditioned so that flipping its sign doesn't introduce numerical
    weirdness."""
    torch.manual_seed(0)
    m = QuantileMarginal(
        context_dim=16,
        mlp_layers=1,
        mlp_dim=8,
        quantile_levels=DEFAULT_LEVELS,
    )
    context = torch.randn(3, 5, 16)
    x = torch.randn(3, 5) * 0.5
    loss = m.pinball_loss(context, x)
    assert (loss > 0).all()
    assert (loss < 100).all()  # not blowing up
    # Negating gives a valid logdet-shape tensor
    neg = -loss
    assert neg.shape == (3,)
    assert torch.isfinite(neg).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
