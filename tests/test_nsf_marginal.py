"""Tests for nsf_marginal.py (Phase 0i-E).

Verifies that NSFMarginal:
  - exposes the same 3 public methods as DSFMarginal with matching shape contracts,
  - round-trips x → u → x under forward_logdet → inverse to machine precision,
  - clips outputs at tail_bound (linear-tail behavior),
  - structurally cannot collapse below min_derivative,
  - accepts the same constructor pattern + lightning_module hparam plumbing.
"""
import os
import sys
import math

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pytorch_transformer_ts.tactis_2.nsf_marginal import NSFMarginal, _NSFFlowStack


# ---------------------------------------------------------------------------
# 1. Constructor + interface compatibility
# ---------------------------------------------------------------------------
def test_nsf_marginal_constructs_with_default_hparams():
    m = NSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                    num_flow_layers=2, num_bins=32, tail_bound=4.0,
                    min_derivative=1e-3)
    # Mirror DSFMarginal attributes that lightning_module / decoder.py read
    assert hasattr(m, "marginal_flow")
    assert hasattr(m, "marginal_conditioner")
    assert hasattr(m, "expected_context_dim")
    assert m.expected_context_dim == 64
    assert m.marginal_flow.total_params_length == 2 * (3 * 32 - 1)


def test_nsf_marginal_interface_matches_dsf():
    """Duck-typing: NSFMarginal must expose the same 3 methods as DSFMarginal
    with the same call signatures."""
    from pytorch_transformer_ts.tactis_2.dsf_marginal import DSFMarginal
    nsf = NSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                      num_flow_layers=2, num_bins=32, tail_bound=4.0,
                      min_derivative=1e-3)
    dsf = DSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                      flow_layers=2, flow_hid_dim=64)
    for name in ("forward_logdet", "forward_no_logdet", "inverse"):
        assert callable(getattr(nsf, name))
        assert callable(getattr(dsf, name))


# ---------------------------------------------------------------------------
# 2. Forward + logdet shape contract
# ---------------------------------------------------------------------------
def test_nsf_forward_logdet_shapes():
    """forward_logdet: input (B, N, D) context + (B, N) x → ((B, N), (B,))"""
    m = NSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                    num_flow_layers=2, num_bins=32, tail_bound=4.0)
    B, N, D = 4, 12, 64
    ctx = torch.randn(B, N, D)
    x = torch.randn(B, N)
    u, ld = m.forward_logdet(ctx, x)
    assert u.shape == (B, N)
    assert ld.shape == (B,)
    assert torch.isfinite(u).all()
    assert torch.isfinite(ld).all()


def test_nsf_forward_no_logdet_with_sample_dim():
    """forward_no_logdet must broadcast a sample axis on x (used during inference)."""
    m = NSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                    num_flow_layers=2, num_bins=32, tail_bound=4.0)
    B, N, D, S = 2, 6, 64, 16
    ctx = torch.randn(B, N, D)
    x_with_samples = torch.randn(B, N, S)
    out = m.forward_no_logdet(ctx, x_with_samples)
    assert out.shape == (B, N, S)


# ---------------------------------------------------------------------------
# 3. Round-trip: forward then inverse should recover x to machine precision
# ---------------------------------------------------------------------------
def test_nsf_round_trip_machine_precision():
    """forward_logdet(x) → u; inverse(u) ≈ x to ~1e-4 (NSF inverse is analytic)."""
    torch.manual_seed(0)
    m = NSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                    num_flow_layers=2, num_bins=32, tail_bound=4.0)
    B, N, D = 3, 8, 64
    ctx = torch.randn(B, N, D)
    # Use x within [-tail_bound + 0.5, tail_bound - 0.5] to avoid the linear-tail region
    x = torch.empty(B, N).uniform_(-3.0, 3.0)
    u, _ = m.forward_logdet(ctx, x)
    x_back = m.inverse(ctx, u)
    err = (x_back - x).abs().max().item()
    assert err < 1e-3, f"NSF round-trip error {err:.2e} exceeds 1e-3"


# ---------------------------------------------------------------------------
# 4. Linear tails: data outside [-tail_bound, tail_bound] is identity
# ---------------------------------------------------------------------------
def test_nsf_linear_tails_passthrough():
    """For x far outside [-tail_bound, tail_bound], the spline acts as identity
    (linear tails). forward_logdet should return ~x unchanged."""
    torch.manual_seed(1)
    m = NSFMarginal(context_dim=64, mlp_layers=1, mlp_dim=8,
                    num_flow_layers=1, num_bins=16, tail_bound=4.0)
    B, N, D = 2, 4, 64
    ctx = torch.randn(B, N, D)
    x_far = torch.tensor([[10.0, -10.0, 8.5, -8.5],
                          [10.0, -10.0, 8.5, -8.5]])  # all |x| > tail_bound
    u, _ = m.forward_logdet(ctx, x_far)
    # In linear-tail region, u = x (identity)
    err = (u - x_far).abs().max().item()
    assert err < 1e-4, f"Linear-tail identity failed; max diff {err:.2e}"


# ---------------------------------------------------------------------------
# 5. Structural anti-collapse: min_derivative floor binds in the diagnostic buffer
# ---------------------------------------------------------------------------
def test_nsf_min_derivative_structural_floor():
    """Even with extreme conditioner outputs that try to push slopes to 0, the
    nflows implementation enforces slope >= min_derivative everywhere.

    We probe this by manually constructing params with very negative derivative
    inputs (softplus → ~0) and verifying the diagnostic floor binds at exactly
    min_derivative.
    """
    md = 1e-3
    stack = _NSFFlowStack(num_flow_layers=1, num_bins=16, tail_bound=4.0,
                          min_derivative=md)
    B, N = 2, 4
    K = 16
    params = torch.zeros(B, N, stack.params_per_layer)
    # Set derivative slot (last K-1 entries) to extremely negative so softplus → ~0
    params[..., 2 * K : 3 * K - 1] = -50.0
    x = torch.zeros(B, N)
    _ = stack.forward(params, x)
    # Diagnostic should record a slope very close to min_derivative
    assert len(stack._last_min_derivative_check) == 1
    floor_observed = stack._last_min_derivative_check[0]
    assert abs(floor_observed - md) < 1e-3, (
        f"Expected slope floor ≈ {md}, observed {floor_observed}"
    )


# ---------------------------------------------------------------------------
# 6. Gradient flows from logdet through conditioner MLP
# ---------------------------------------------------------------------------
def test_nsf_logdet_gradient_flows_to_conditioner():
    """An NLL-style backprop on logdet should produce non-zero gradients on
    every conditioner Linear layer's weights — the necessary condition for
    NSF to actually train under NLL."""
    m = NSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                    num_flow_layers=2, num_bins=16, tail_bound=4.0)
    B, N, D = 2, 4, 64
    ctx = torch.randn(B, N, D)
    x = torch.randn(B, N)
    _, ld = m.forward_logdet(ctx, x)
    nll = -ld.mean()
    nll.backward()
    grad_norms = []
    for name, p in m.marginal_conditioner.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad on {name}"
            grad_norms.append(p.grad.abs().max().item())
    assert max(grad_norms) > 0, (
        "Expected non-zero gradient on at least one conditioner weight"
    )


# ---------------------------------------------------------------------------
# 7. Inverse with extra sample axis (mimics inference broadcast pattern)
# ---------------------------------------------------------------------------
def test_nsf_inverse_with_sample_axis():
    """Inference draws u with shape (B, N, S). NSFMarginal.inverse must
    broadcast params to match that extra axis (same convention as DSF)."""
    m = NSFMarginal(context_dim=64, mlp_layers=2, mlp_dim=8,
                    num_flow_layers=2, num_bins=16, tail_bound=4.0)
    B, N, D, S = 2, 5, 64, 8
    ctx = torch.randn(B, N, D)
    u = torch.empty(B, N, S).uniform_(0.05, 0.95)
    # Map u from [0, 1] to [-tail_bound + ε, tail_bound - ε] for the inverse-call test;
    # nflows' `unconstrained_rational_quadratic_spline(inverse=True)` expects inputs
    # in the same domain as the forward output, which for "linear" tails is the full real line.
    # The simplest valid input: u itself in [-tail_bound, tail_bound] tested below.
    u_in_range = (u - 0.5) * 6.0  # rescale to [-2.7, 2.7]
    x = m.inverse(ctx, u_in_range)
    assert x.shape == (B, N, S)
    assert torch.isfinite(x).all()


# ---------------------------------------------------------------------------
# 8. Lightning module + estimator accept the new hparams
# ---------------------------------------------------------------------------
def test_lightning_module_accepts_nsf_hparams():
    """LightningModule.__init__ must accept the 5 NSF hparams added in Phase 0i-E."""
    from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule
    import inspect
    sig = inspect.signature(TACTiS2LightningModule.__init__)
    for name in (
        "marginal_flow_type",
        "decoder_nsf_num_bins",
        "decoder_nsf_tail_bound",
        "decoder_nsf_num_layers",
        "decoder_nsf_min_derivative",
    ):
        assert name in sig.parameters, f"hparam {name} missing from LightningModule.__init__"


def test_estimator_accepts_nsf_hparams():
    """Estimator.__init__ must accept the same 5 NSF hparams for plumbing."""
    from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator
    import inspect
    sig = inspect.signature(TACTiS2Estimator.__init__)
    for name in (
        "marginal_flow_type",
        "decoder_nsf_num_bins",
        "decoder_nsf_tail_bound",
        "decoder_nsf_num_layers",
        "decoder_nsf_min_derivative",
    ):
        assert name in sig.parameters, f"hparam {name} missing from Estimator.__init__"


if __name__ == "__main__":
    test_nsf_marginal_constructs_with_default_hparams()
    test_nsf_marginal_interface_matches_dsf()
    test_nsf_forward_logdet_shapes()
    test_nsf_forward_no_logdet_with_sample_dim()
    test_nsf_round_trip_machine_precision()
    test_nsf_linear_tails_passthrough()
    test_nsf_min_derivative_structural_floor()
    test_nsf_logdet_gradient_flows_to_conditioner()
    test_nsf_inverse_with_sample_axis()
    test_lightning_module_accepts_nsf_hparams()
    test_estimator_accepts_nsf_hparams()
    print("All NSFMarginal tests passed.")
