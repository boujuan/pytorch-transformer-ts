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


# =============================================================================
# Deeper-fix v2 tests (per-datapoint log-density penalty + smooth a-cap +
# lambda schedule). See plan: /user/taed7566/.claude/plans/moonlit-sprouting-gosling.md
# =============================================================================

def test_smooth_a_cap_bounds_output_with_nonzero_gradient():
    """
    Fix B: smooth_a_cap(a_raw, a_max) must:
      1) Be approximately the identity for a_raw << a_max (healthy regime undisturbed)
      2) Asymptote to a_max for a_raw >> a_max (collapse regime bounded)
      3) Have NON-ZERO gradient everywhere (unlike `clamp(., max=a_max)`, which would
         trap the optimizer in the collapsed regime by zeroing gradient above the cap)
      4) Return a_raw unchanged when a_max <= 0 (disabled mode)
    """
    from pytorch_transformer_ts.tactis_2.sigmoid_flow import smooth_a_cap

    A_MAX = 20.0

    # 1) Healthy regime — a_raw=2 should map to ~2 (small distortion only)
    a_small = torch.tensor(2.0)
    out_small = smooth_a_cap(a_small, A_MAX)
    assert abs(out_small.item() - 2.0) < 0.5, f"Expected ~2.0, got {out_small.item()}"

    # 2) Pathological regime — a_raw=720 (broken-model max) should asymptote to ~A_MAX
    a_huge = torch.tensor(720.0)
    out_huge = smooth_a_cap(a_huge, A_MAX)
    # 720/(1+720/20) = 720/37 ≈ 19.46. Should be < A_MAX but close to it.
    assert out_huge.item() < A_MAX, f"Should asymptote below {A_MAX}, got {out_huge.item()}"
    assert out_huge.item() > 0.95 * A_MAX, f"Should be close to {A_MAX}, got {out_huge.item()}"

    # 3) Non-zero gradient at large a_raw (unlike clamp)
    a_grad = torch.tensor(500.0, requires_grad=True)
    y = smooth_a_cap(a_grad, A_MAX)
    y.backward()
    assert a_grad.grad is not None and a_grad.grad.item() > 0, \
        f"Gradient must flow above the cap; got {a_grad.grad}"
    # Gradient should be small but non-zero
    assert a_grad.grad.item() < 0.01, "Gradient should be small in saturation"

    # 4) a_max=0 disables — pass-through
    a = torch.tensor(50.0)
    assert smooth_a_cap(a, 0.0).item() == 50.0
    assert smooth_a_cap(a, -1.0).item() == 50.0


def test_log_density_reg_fires_above_threshold_zero_below():
    """
    Fix A: per-(batch, vars) log-density penalty equals
        relu(log_density - threshold).pow(2).mean()
    Test the math directly without a full model. Should be:
      - Zero when all log-densities < threshold
      - Positive (== mean of squared excesses) when some log-densities > threshold
    """
    threshold = 3.0

    # Below threshold -> zero
    log_density_below = torch.tensor([[1.0, 2.0, 2.5], [0.5, 1.8, 2.9]])
    reg_below = torch.relu(log_density_below - threshold).pow(2).mean()
    assert reg_below.item() == pytest.approx(0.0), f"Below threshold should give 0, got {reg_below.item()}"

    # Above threshold: log_density=[5, 5, 5, 5] -> excess=[2, 2, 2, 2] -> squared=[4,4,4,4] -> mean=4
    log_density_above = torch.full((2, 2), 5.0)
    reg_above = torch.relu(log_density_above - threshold).pow(2).mean()
    assert reg_above.item() == pytest.approx(4.0), f"Expected 4.0, got {reg_above.item()}"

    # Mixed: half above, half below — only the above contribute to mean (others are 0)
    log_density_mixed = torch.tensor([[5.0, 1.0], [1.0, 5.0]])  # 2 of 4 are excess=2 → squared=4; mean = (4+0+0+4)/4 = 2
    reg_mixed = torch.relu(log_density_mixed - threshold).pow(2).mean()
    assert reg_mixed.item() == pytest.approx(2.0), f"Expected 2.0 (mixed), got {reg_mixed.item()}"


def test_lambda_schedule_multiplier_per_epoch():
    """
    Fix C: lambda schedule should produce:
      - 5.0 for epoch in [0, 4]
      - Linear decay from 5.0 (epoch 5) to 1.0 (epoch 15)
      - 1.0 for epoch >= 15
    Replicates the inline logic from lightning_module.training_step.
    """
    def schedule(cur_epoch: int) -> float:
        if cur_epoch < 5:
            return 5.0
        elif cur_epoch < 15:
            return 5.0 - 4.0 * (cur_epoch - 5) / 10.0
        else:
            return 1.0

    # Boost regime
    assert schedule(0) == 5.0
    assert schedule(4) == 5.0
    # Decay regime
    assert schedule(5) == pytest.approx(5.0)            # exactly at start of decay
    assert schedule(10) == pytest.approx(3.0)           # halfway through decay
    assert schedule(14) == pytest.approx(1.4)           # near end of decay
    # Plateau regime
    assert schedule(15) == 1.0
    assert schedule(50) == 1.0
    # Monotonically non-increasing
    prev = schedule(0)
    for e in range(1, 20):
        cur = schedule(e)
        assert cur <= prev + 1e-9, f"Schedule must be non-increasing; epoch {e}: {cur} > prev {prev}"
        prev = cur


def test_lightning_module_accepts_v2_hparams():
    """
    Smoke test: TACTiS2LightningModule accepts the three new v2 hparams
    (lambda_log_density, log_density_max, a_max) and stores them in self.hparams.
    Default values (0.0 / 3.0 / 0.0) keep the v2 reg disabled (backward-compat).
    """
    from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule

    model_config = {
        "num_series": 2, "context_length": 80, "prediction_length": 4,
        "flow_series_embedding_dim": 32, "copula_series_embedding_dim": 32,
        "flow_input_encoder_layers": 2, "copula_input_encoder_layers": 2,
        "marginal_embedding_dim_per_head": 16, "marginal_num_heads": 2, "marginal_num_layers": 2,
        "copula_embedding_dim_per_head": 16, "copula_num_heads": 2, "copula_num_layers": 2,
        "decoder_dsf_num_layers": 3, "decoder_dsf_hidden_dim": 48,
        "decoder_mlp_num_layers": 2, "decoder_mlp_hidden_dim": 32,
        "decoder_transformer_num_layers": 2, "decoder_transformer_embedding_dim_per_head": 16,
        "decoder_transformer_num_heads": 2, "decoder_num_bins": 50,
        "ac_mlp_num_layers": 2, "ac_mlp_dim": 64, "bagging_size": 64,
        "input_encoding_normalization": True, "loss_normalization": "none",
        "skip_copula": True, "lock_skip_copula": True, "encoder_type": "standard",
        "dropout_rate": 0.005,
        "cardinality": [88], "num_feat_dynamic_real": 2, "num_feat_static_real": 0,
        "num_feat_static_cat": 1, "embedding_dimension": None, "scaling": "std",
        "lags_seq": [0], "num_parallel_samples": 10,
    }

    # Explicit v2 values
    lit = TACTiS2LightningModule(
        model_config=model_config, stage=1, stage2_start_epoch=9999,
        lambda_log_density=0.5, log_density_max=2.0, a_max=20.0,
    )
    assert lit.hparams.lambda_log_density == pytest.approx(0.5)
    assert lit.hparams.log_density_max == pytest.approx(2.0)
    assert lit.hparams.a_max == pytest.approx(20.0)
    # a_max=20 should propagate to all SigmoidFlow layers
    flow = lit.model.tactis.decoder.marginal.marginal_flow
    for layer in flow.layers:
        assert layer.a_max == pytest.approx(20.0), \
            f"a_max should propagate to flow layers; layer.a_max={layer.a_max}"

    # Default values (no kwargs) → all v2 reg disabled
    lit_default = TACTiS2LightningModule(
        model_config=model_config, stage=1, stage2_start_epoch=9999,
    )
    assert lit_default.hparams.lambda_log_density == 0.0
    assert lit_default.hparams.log_density_max == 3.0
    assert lit_default.hparams.a_max == 0.0
    # a_max=0 means SigmoidFlow layers stay at default a_max (no cap applied)
    flow_default = lit_default.model.tactis.decoder.marginal.marginal_flow
    for layer in flow_default.layers:
        assert layer.a_max == 0.0, "a_max=0 leaves layers unmodified"


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
    test_smooth_a_cap_bounds_output_with_nonzero_gradient()
    print("PASS: test_smooth_a_cap_bounds_output_with_nonzero_gradient")
    test_log_density_reg_fires_above_threshold_zero_below()
    print("PASS: test_log_density_reg_fires_above_threshold_zero_below")
    test_lambda_schedule_multiplier_per_epoch()
    print("PASS: test_lambda_schedule_multiplier_per_epoch")
    test_lightning_module_accepts_v2_hparams()
    print("PASS: test_lightning_module_accepts_v2_hparams")
    print("\nAll regularization tests PASSED")
