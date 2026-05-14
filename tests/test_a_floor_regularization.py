"""Tests for the a_floor regularizer (Fix Sa).

The a_floor penalty is `lambda_a_floor * mean(relu(a_floor_threshold - a_post)^2)`
where a_post is the post-softplus + post-smooth-cap `a` value emitted by each
SigmoidFlow layer. This penalizes flat-CDF regions that cause F1 (sample
tightness) and F2 (time-flatness) — see plan moonlit-sprouting-gosling.md.
"""
import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pytorch_transformer_ts.tactis_2.sigmoid_flow import SigmoidFlow, smooth_a_cap, EPSILON
from pytorch_transformer_ts.tactis_2.deep_sigmoid_flow import DeepSigmoidFlow


def _compute_a_floor_reg(a_all: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mirror of the production computation in lightning_module.training_step."""
    return torch.relu(threshold - a_all).pow(2).mean()


def test_a_floor_reg_zero_when_all_above_threshold():
    """When all `a` values exceed the floor threshold, penalty is 0."""
    threshold = 0.5
    a = torch.full((4, 8, 64), 0.7)  # all > 0.5
    reg = _compute_a_floor_reg(a, threshold)
    assert reg.item() == pytest.approx(0.0, abs=1e-9)


def test_a_floor_reg_positive_when_below_threshold():
    """When some `a` values fall below threshold, penalty is positive."""
    threshold = 0.5
    a = torch.tensor([0.1, 0.2, 0.5, 0.7, 1.0])
    reg = _compute_a_floor_reg(a, threshold)
    # relu(0.5 - [0.1, 0.2, 0.5, 0.7, 1.0]) = [0.4, 0.3, 0.0, 0.0, 0.0]
    # squared mean = (0.16 + 0.09 + 0 + 0 + 0) / 5 = 0.25/5 = 0.05
    assert reg.item() == pytest.approx(0.05, rel=1e-6)


def test_a_floor_reg_grows_quadratically():
    """Penalty grows as (threshold - a)^2 in the violating regime."""
    threshold = 0.5
    a1 = torch.tensor([0.4])  # 0.1 below threshold
    a2 = torch.tensor([0.3])  # 0.2 below threshold (2x violation)
    r1 = _compute_a_floor_reg(a1, threshold).item()
    r2 = _compute_a_floor_reg(a2, threshold).item()
    # 2x violation should give 4x penalty (quadratic)
    assert r2 / r1 == pytest.approx(4.0, rel=1e-6)


def test_sigmoid_flow_exposes_a_post_cap():
    """SigmoidFlow.forward and forward_no_logdet must populate _last_a_post_cap."""
    h = 8
    layer = SigmoidFlow(hidden_dim=h, no_logit=False, a_max=3.0)
    params = torch.randn(2, 4, 3 * h)
    x = torch.randn(2, 4)
    logdet = torch.zeros(2)

    # forward path
    layer._last_a_post_cap = None
    _, _ = layer(params, x, logdet)
    assert layer._last_a_post_cap is not None
    assert layer._last_a_post_cap.shape == (2, 4, h)
    # post-cap a should be in (0, a_max + EPSILON]
    assert (layer._last_a_post_cap > 0).all()
    assert (layer._last_a_post_cap <= 3.0 + EPSILON + 1e-5).all()

    # forward_no_logdet path
    layer._last_a_post_cap = None
    _ = layer.forward_no_logdet(params, x)
    assert layer._last_a_post_cap is not None
    assert layer._last_a_post_cap.shape == (2, 4, h)


def test_deep_sigmoid_flow_accumulates_per_layer_a():
    """DeepSigmoidFlow should rebuild _last_a_post_cap_per_layer each forward call."""
    n_layers, h = 3, 16
    flow = DeepSigmoidFlow(n_layers=n_layers, hidden_dim=h, a_max=3.0)
    total_params = n_layers * 3 * h
    params = torch.randn(2, 4, total_params)
    x = torch.randn(2, 4)

    # Initially empty
    assert flow._last_a_post_cap_per_layer == []

    # forward populates per-layer list
    _, _ = flow(params, x)
    assert len(flow._last_a_post_cap_per_layer) == n_layers
    for layer_a in flow._last_a_post_cap_per_layer:
        assert layer_a.shape == (2, 4, h)
        assert (layer_a > 0).all()

    # forward_no_logdet also populates the list
    _ = flow.forward_no_logdet(params, x)
    assert len(flow._last_a_post_cap_per_layer) == n_layers


def test_a_floor_reg_gradient_flows_to_marginal_params():
    """Gradient should flow from a_floor penalty back to the raw params."""
    h = 8
    layer = SigmoidFlow(hidden_dim=h, no_logit=False, a_max=3.0)
    params = torch.randn(2, 4, 3 * h, requires_grad=True)
    x = torch.randn(2, 4)
    logdet = torch.zeros(2)
    _, _ = layer(params, x, logdet)

    a_post = layer._last_a_post_cap
    # Force some `a` values below threshold by using small param values
    threshold = 1.0  # above typical post-softplus values for randn input
    reg = _compute_a_floor_reg(a_post, threshold)
    reg.backward()

    # Gradient should reach the `a_pre` slice (first hidden_dim of each layer block)
    assert params.grad is not None
    a_pre_grad = params.grad[..., :h]
    # Most of the grad should be on the a_pre slice (since b and pre_w don't enter the penalty)
    b_grad = params.grad[..., h:2*h]
    w_grad = params.grad[..., 2*h:]
    # b and w gradients should be exactly zero (not in computational graph for the penalty)
    assert b_grad.abs().max().item() < 1e-9
    assert w_grad.abs().max().item() < 1e-9
    # a_pre gradient should be non-zero (there's a violation)
    assert a_pre_grad.abs().max().item() > 0


def test_a_floor_reg_no_op_when_lambda_zero():
    """When lambda_a_floor=0, the penalty should add nothing to loss (backward compat)."""
    base_loss = torch.tensor(2.5)
    a = torch.tensor([0.05, 0.1])  # below typical thresholds
    threshold = 0.5
    lambda_a_floor = 0.0
    reg = _compute_a_floor_reg(a, threshold)
    final_loss = base_loss + lambda_a_floor * reg
    assert final_loss.item() == base_loss.item()


def test_lightning_module_accepts_a_floor_hparams():
    """Lightning module __init__ accepts the new hparams without error."""
    from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule
    import inspect
    sig = inspect.signature(TACTiS2LightningModule.__init__)
    assert "lambda_a_floor" in sig.parameters
    assert "a_floor_threshold" in sig.parameters
    # Defaults should preserve backward compat
    assert sig.parameters["lambda_a_floor"].default == 0.0
    assert sig.parameters["a_floor_threshold"].default == 0.5


def test_estimator_accepts_a_floor_hparams():
    """Estimator __init__ accepts the new hparams and plumbs them through."""
    from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator
    import inspect
    sig = inspect.signature(TACTiS2Estimator.__init__)
    assert "lambda_a_floor" in sig.parameters
    assert "a_floor_threshold" in sig.parameters


if __name__ == "__main__":
    test_a_floor_reg_zero_when_all_above_threshold()
    test_a_floor_reg_positive_when_below_threshold()
    test_a_floor_reg_grows_quadratically()
    test_sigmoid_flow_exposes_a_post_cap()
    test_deep_sigmoid_flow_accumulates_per_layer_a()
    test_a_floor_reg_gradient_flows_to_marginal_params()
    test_a_floor_reg_no_op_when_lambda_zero()
    test_lightning_module_accepts_a_floor_hparams()
    test_estimator_accepts_a_floor_hparams()
    print("All a_floor regularization tests passed.")
