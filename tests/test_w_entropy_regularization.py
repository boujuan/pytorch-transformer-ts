"""Tests for the w-entropy regularizer (Fix Sw).

The w-entropy penalty is `lambda_w_entropy * mean(relu(w_entropy_min - H(w))^2)`
where H(w) is the per-(b,v) Shannon entropy of softmax weights from each
SigmoidFlow layer. Penalizes one-hot collapse — the actual root cause of
F1/F2 calibration failures per 2026-05-09 diagnostics.
"""
import os, sys, math
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pytorch_transformer_ts.tactis_2.sigmoid_flow import SigmoidFlow
from pytorch_transformer_ts.tactis_2.deep_sigmoid_flow import DeepSigmoidFlow


def _compute_w_entropy_reg(H: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.relu(threshold - H).pow(2).mean()


def test_uniform_w_has_max_entropy():
    """Uniform softmax over n units has entropy log(n)."""
    n = 64
    pre_w = torch.zeros(n)
    w = F.softmax(pre_w, dim=-1)
    H = -(w * (w + 1e-20).log()).sum(dim=-1)
    assert H.item() == pytest.approx(math.log(n), abs=1e-5)


def test_one_hot_w_has_zero_entropy():
    """A one-hot w (one large pre_w, rest small) approaches zero entropy."""
    pre_w = torch.tensor([100.0, -100.0, -100.0])
    w = F.softmax(pre_w, dim=-1)
    H = -(w * (w + 1e-20).log()).sum(dim=-1)
    assert H.item() == pytest.approx(0.0, abs=1e-3)


def test_w_entropy_reg_zero_when_above_threshold():
    """If H is everywhere above the floor, penalty is 0."""
    threshold = 1.0
    H = torch.tensor([1.5, 2.0, 3.0])
    reg = _compute_w_entropy_reg(H, threshold)
    assert reg.item() == pytest.approx(0.0)


def test_w_entropy_reg_quadratic_below_threshold():
    """Penalty grows as (threshold - H)^2 in violation regime."""
    threshold = 2.0
    H1 = torch.tensor([1.0])  # 1.0 below
    H2 = torch.tensor([0.0])  # 2.0 below (2x violation)
    r1 = _compute_w_entropy_reg(H1, threshold).item()
    r2 = _compute_w_entropy_reg(H2, threshold).item()
    assert r2 / r1 == pytest.approx(4.0, rel=1e-6)


def test_sigmoid_flow_exposes_w_entropy():
    """SigmoidFlow.forward and forward_no_logdet must populate _last_w_entropy."""
    h = 8
    layer = SigmoidFlow(hidden_dim=h, no_logit=False, a_max=3.0)
    params = torch.randn(2, 4, 3 * h)
    x = torch.randn(2, 4)
    logdet = torch.zeros(2)

    layer._last_w_entropy = None
    _, _ = layer(params, x, logdet)
    assert layer._last_w_entropy is not None
    assert layer._last_w_entropy.shape == (2, 4)
    assert (layer._last_w_entropy >= 0).all()
    assert (layer._last_w_entropy <= math.log(h) + 1e-3).all()

    layer._last_w_entropy = None
    _ = layer.forward_no_logdet(params, x)
    assert layer._last_w_entropy is not None
    assert layer._last_w_entropy.shape == (2, 4)


def test_deep_sigmoid_flow_accumulates_per_layer_w_entropy():
    """DeepSigmoidFlow must rebuild _last_w_entropy_per_layer each forward call."""
    n_layers, h = 3, 16
    flow = DeepSigmoidFlow(n_layers=n_layers, hidden_dim=h, a_max=3.0)
    total_params = n_layers * 3 * h
    params = torch.randn(2, 4, total_params)
    x = torch.randn(2, 4)

    assert flow._last_w_entropy_per_layer == []

    _, _ = flow(params, x)
    assert len(flow._last_w_entropy_per_layer) == n_layers
    for layer_we in flow._last_w_entropy_per_layer:
        assert layer_we.shape == (2, 4)
        assert (layer_we >= 0).all()
        assert (layer_we <= math.log(h) + 1e-3).all()

    _ = flow.forward_no_logdet(params, x)
    assert len(flow._last_w_entropy_per_layer) == n_layers


def test_w_entropy_reg_gradient_flows_to_pre_w():
    """Gradient should flow from w-entropy penalty back to the pre_w slice."""
    h = 16
    layer = SigmoidFlow(hidden_dim=h, no_logit=False, a_max=3.0)
    # Construct params with EXTREME pre_w values so softmax is one-hot → H close to 0
    pre_w_concentrated = torch.zeros(1, 1, h)
    pre_w_concentrated[..., 0] = 100.0
    a_pre = torch.zeros(1, 1, h)
    b = torch.zeros(1, 1, h)
    params = torch.cat([a_pre, b, pre_w_concentrated], dim=-1).requires_grad_()
    x = torch.zeros(1, 1)
    logdet = torch.zeros(1)
    _, _ = layer(params, x, logdet)
    H = layer._last_w_entropy
    # H should be near zero (one-hot) → big violation of threshold=2.0
    threshold = 2.0
    reg = _compute_w_entropy_reg(H, threshold)
    assert reg.item() > 0
    reg.backward()
    assert params.grad is not None
    a_grad = params.grad[..., :h]
    b_grad = params.grad[..., h:2*h]
    pre_w_grad = params.grad[..., 2*h:]
    # Gradient should ONLY land on pre_w (a and b don't enter the penalty)
    assert a_grad.abs().max().item() < 1e-9
    assert b_grad.abs().max().item() < 1e-9
    assert pre_w_grad.abs().max().item() > 0


def test_w_entropy_reg_no_op_when_lambda_zero():
    """Backward compat: lambda=0 → no contribution to loss."""
    base_loss = torch.tensor(2.5)
    H = torch.tensor([0.0, 0.0])  # one-hot
    threshold = 2.0
    lambda_w = 0.0
    reg = _compute_w_entropy_reg(H, threshold)
    assert (base_loss + lambda_w * reg).item() == base_loss.item()


def test_lightning_module_accepts_w_entropy_hparams():
    from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule
    import inspect
    sig = inspect.signature(TACTiS2LightningModule.__init__)
    assert "lambda_w_entropy" in sig.parameters
    assert "w_entropy_min" in sig.parameters
    assert sig.parameters["lambda_w_entropy"].default == 0.0
    assert sig.parameters["w_entropy_min"].default == 2.0


def test_estimator_accepts_w_entropy_hparams():
    from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator
    import inspect
    sig = inspect.signature(TACTiS2Estimator.__init__)
    assert "lambda_w_entropy" in sig.parameters
    assert "w_entropy_min" in sig.parameters


if __name__ == "__main__":
    test_uniform_w_has_max_entropy()
    test_one_hot_w_has_zero_entropy()
    test_w_entropy_reg_zero_when_above_threshold()
    test_w_entropy_reg_quadratic_below_threshold()
    test_sigmoid_flow_exposes_w_entropy()
    test_deep_sigmoid_flow_accumulates_per_layer_w_entropy()
    test_w_entropy_reg_gradient_flows_to_pre_w()
    test_w_entropy_reg_no_op_when_lambda_zero()
    test_lightning_module_accepts_w_entropy_hparams()
    test_estimator_accepts_w_entropy_hparams()
    print("All w-entropy tests passed.")
