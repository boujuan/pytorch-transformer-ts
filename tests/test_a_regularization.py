"""
Verification tests for the DSF flow-collapse regularization.

The regularization is a soft-hinge L2 penalty on the pre-softplus `a` parameter
of the DeepSigmoidFlow conditioner output, gated to Stage 1 training only.
See pytorch_transformer_ts/tactis_2/lightning_module.py training_step.

These tests verify the math, the forward-hook capture pattern, and the
backward-compatibility default (lambda=0 → identity behavior).
"""
import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _compute_a_reg(marginal_params: torch.Tensor, n_layers: int, hidden_dim: int, threshold: float) -> torch.Tensor:
    """The regularization computation that lives in TACTiS2LightningModule.training_step.

    Replicated here so the test does not depend on instantiating the full module.
    Any divergence between this and the production code is itself a bug.
    """
    params_per_layer = 3 * hidden_dim
    a_pre_slices = [
        marginal_params[..., i * params_per_layer : i * params_per_layer + hidden_dim]
        for i in range(n_layers)
    ]
    a_pre_all = torch.cat(a_pre_slices, dim=-1)
    return torch.relu(a_pre_all - threshold).pow(2).mean()


def test_a_reg_zero_when_all_below_threshold():
    """When `a_pre` is everywhere below the threshold, the soft-hinge penalty is 0."""
    n_layers, hidden_dim = 3, 48
    params_per_layer = 3 * hidden_dim
    total = n_layers * params_per_layer
    # Random params with a_pre values all in [-2, 2] (well below threshold=3)
    torch.manual_seed(0)
    marginal_params = torch.randn(2, 8, total) * 0.5  # std 0.5 → mostly in [-1.5, 1.5]
    threshold = 3.0
    a_reg = _compute_a_reg(marginal_params, n_layers, hidden_dim, threshold)
    assert a_reg.item() == pytest.approx(0.0, abs=1e-9), (
        f"Expected zero regularization for sub-threshold a_pre, got {a_reg.item()}"
    )


def test_a_reg_positive_when_above_threshold():
    """When `a_pre` exceeds threshold, the penalty equals (a_pre - threshold)^2 averaged."""
    n_layers, hidden_dim = 3, 48
    params_per_layer = 3 * hidden_dim
    total = n_layers * params_per_layer
    threshold = 3.0
    # Construct: layer 0 a_pre = 8.0 (well above threshold), b and w arbitrary
    # Layers 1, 2 with small a_pre = 0.5 (below threshold)
    marginal_params = torch.zeros(1, 1, total)
    # Layer 0: a portion (first hidden_dim) = 8.0
    marginal_params[..., :hidden_dim] = 8.0
    # Layer 0: b and w portions = 0.0 (won't be touched by the regularizer)
    # Layers 1, 2: a portion = 0.5 (below threshold)
    for li in [1, 2]:
        marginal_params[..., li * params_per_layer : li * params_per_layer + hidden_dim] = 0.5
    a_reg = _compute_a_reg(marginal_params, n_layers, hidden_dim, threshold)
    # Expected: only layer 0's a contributes. relu(8 - 3) = 5, squared = 25.
    # Mean over n_layers * hidden_dim = 144 values, only 48 (layer 0) are 25, rest 0.
    expected_mean = (48 * 25.0) / 144.0
    assert a_reg.item() == pytest.approx(expected_mean, rel=1e-6), (
        f"Expected a_reg ≈ {expected_mean}, got {a_reg.item()}"
    )


def test_a_reg_grows_quadratically_in_excess():
    """If a_pre - threshold doubles, the penalty quadruples (it's an L2 hinge)."""
    n_layers, hidden_dim = 1, 4
    threshold = 3.0
    # Single layer, small hidden_dim for hand-checkability.
    # First params = a_pre, last 8 = b + w (ignored)
    base = torch.zeros(1, 1, 3 * hidden_dim)
    base[..., :hidden_dim] = 5.0  # excess = 2
    reg_low = _compute_a_reg(base, n_layers, hidden_dim, threshold).item()

    base[..., :hidden_dim] = 7.0  # excess = 4 (doubled)
    reg_high = _compute_a_reg(base, n_layers, hidden_dim, threshold).item()

    # Quadratic: reg_high should be ~4x reg_low
    assert reg_high == pytest.approx(reg_low * 4.0, rel=1e-5), (
        f"Quadratic-in-excess relationship broken: reg_high={reg_high}, reg_low={reg_low}, "
        f"ratio={reg_high / reg_low if reg_low > 0 else float('inf')}"
    )


def test_hook_captures_first_call_only():
    """The forward-hook pattern in training_step must capture only the first call.

    During a TACTiS-2 forward+sample pass, marginal_conditioner is called multiple
    times: once for forward_logdet (the loss path we want to regularize), and again
    for sampling/inverse. The hook stores into a list-as-mutable-cell that we only
    write to when len == 0.
    """
    captured = []  # Mimics the closure pattern in training_step

    def _capture_first(module, inputs, output):
        if len(captured) == 0:
            captured.append(output)

    linear = nn.Linear(4, 8)
    handle = linear.register_forward_hook(_capture_first)
    try:
        x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        x2 = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # Different input
        out1 = linear(x1)
        out2 = linear(x2)
        # Must capture exactly one entry, and it must be the first call
        assert len(captured) == 1, f"Expected 1 capture, got {len(captured)}"
        assert torch.allclose(captured[0], out1), "Captured output must match the first call"
        assert not torch.allclose(captured[0], out2), "Captured output must NOT match the second call"
    finally:
        handle.remove()


def test_lightning_module_accepts_new_hparams():
    """Smoke test: TACTiS2LightningModule accepts lambda_a_reg and a_reg_threshold
    and stores them on self.hparams. Backward compat: default (lambda=0) means
    the regularization branch is skipped.
    """
    from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule

    # Minimal model_config sufficient for instantiation. We don't run forward —
    # just verify the hparams plumbing.
    model_config = {
        "num_series": 2,
        "context_length": 80,
        "prediction_length": 4,
        "flow_series_embedding_dim": 32,
        "copula_series_embedding_dim": 32,
        "flow_input_encoder_layers": 2,
        "copula_input_encoder_layers": 2,
        "marginal_embedding_dim_per_head": 16,
        "marginal_num_heads": 2,
        "marginal_num_layers": 2,
        "copula_embedding_dim_per_head": 16,
        "copula_num_heads": 2,
        "copula_num_layers": 2,
        "decoder_dsf_num_layers": 3,
        "decoder_dsf_hidden_dim": 48,
        "decoder_mlp_num_layers": 2,
        "decoder_mlp_hidden_dim": 32,
        "decoder_transformer_num_layers": 2,
        "decoder_transformer_embedding_dim_per_head": 16,
        "decoder_transformer_num_heads": 2,
        "decoder_num_bins": 50,
        "ac_mlp_num_layers": 2,
        "ac_mlp_dim": 64,
        "bagging_size": 64,
        "input_encoding_normalization": True,
        "loss_normalization": "none",
        "skip_copula": True,
        "lock_skip_copula": True,
        "encoder_type": "standard",
        "dropout_rate": 0.005,
        "cardinality": [88],
        "num_feat_dynamic_real": 2,
        "num_feat_static_real": 0,
        "num_feat_static_cat": 1,
        "embedding_dimension": None,
        "scaling": "std",
        "lags_seq": [0],
        "num_parallel_samples": 10,
    }

    # Instantiate with explicit values to confirm they reach hparams
    lit = TACTiS2LightningModule(
        model_config=model_config,
        stage=1,
        stage2_start_epoch=9999,
        lambda_a_reg=2.5e-3,
        a_reg_threshold=3.5,
    )
    assert lit.hparams.lambda_a_reg == pytest.approx(2.5e-3)
    assert lit.hparams.a_reg_threshold == pytest.approx(3.5)

    # Default values (no kwargs) → backward-compat zero
    lit_default = TACTiS2LightningModule(
        model_config=model_config,
        stage=1,
        stage2_start_epoch=9999,
    )
    assert lit_default.hparams.lambda_a_reg == 0.0, "Default lambda_a_reg must be 0 for backward compat"
    assert lit_default.hparams.a_reg_threshold == 3.0, "Default a_reg_threshold must be 3.0"


if __name__ == "__main__":
    test_a_reg_zero_when_all_below_threshold()
    print("PASS: test_a_reg_zero_when_all_below_threshold")
    test_a_reg_positive_when_above_threshold()
    print("PASS: test_a_reg_positive_when_above_threshold")
    test_a_reg_grows_quadratically_in_excess()
    print("PASS: test_a_reg_grows_quadratically_in_excess")
    test_hook_captures_first_call_only()
    print("PASS: test_hook_captures_first_call_only")
    test_lightning_module_accepts_new_hparams()
    print("PASS: test_lightning_module_accepts_new_hparams")
    print("\nAll regularization tests PASSED")
