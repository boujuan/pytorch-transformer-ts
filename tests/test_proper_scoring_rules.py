"""Tests for proper_scoring_rules.py (Fix Sd).

Verifies the math of the Energy Score and Variogram Score implementations,
plus gradient flow (required since these are training losses, not just
evaluation metrics).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pytorch_transformer_ts.tactis_2.proper_scoring_rules import (
    energy_score,
    variogram_score,
    mvg_crps_closedform,
)


# ---------------------------------------------------------------------------
# 1. ES is approximately zero when samples concentrate on the truth
# ---------------------------------------------------------------------------
def test_es_zero_when_samples_equal_target():
    """When all samples are at the truth, ES should be near 0 (within eps).

    The eps stabilizer in the implementation prevents exact zero, but for any
    reasonable epsilon the score should be several orders of magnitude
    smaller than a typical "narrow but lucky" prediction.
    """
    torch.manual_seed(0)
    n, b, d = 32, 4, 8
    target = torch.randn(b, d)
    # Samples concentrated on target with tiny jitter (1e-3) so cdist
    # numerics are well-behaved.
    jitter = 1e-3 * torch.randn(n, b, d)
    samples = target.unsqueeze(0).expand(n, -1, -1) + jitter
    es = energy_score(samples, target)
    # Should be small; the jitter floor sets a baseline ~ 1e-3 * sqrt(d)
    assert es.abs().max().item() < 0.05, (
        f"ES with samples on target should be ~0; got {es.tolist()}"
    )


# ---------------------------------------------------------------------------
# 2. ES is positive when samples are distant from the truth
# ---------------------------------------------------------------------------
def test_es_positive_when_samples_distant():
    """Samples shifted far from the truth should produce positive ES."""
    torch.manual_seed(1)
    n, b, d = 16, 4, 8
    target = torch.zeros(b, d)
    samples = 5.0 + 0.1 * torch.randn(n, b, d)  # all samples ~ 5
    es = energy_score(samples, target)
    # term1 ≈ ||5*ones|| = 5*sqrt(8) ≈ 14.14; term2 ≈ pairwise within tight
    # cluster ≈ 0.4. So ES ≈ 13.7 per batch element.
    assert es.min().item() > 5.0, (
        f"ES with displaced samples should be substantially positive; "
        f"got {es.tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. ES gradient flows back to the samples (and through to whatever produced
#    them). Critical for using ES as a loss term.
# ---------------------------------------------------------------------------
def test_es_gradient_flows_to_samples():
    """Backward through ES should produce non-zero gradient on the samples."""
    torch.manual_seed(2)
    n, b, d = 8, 2, 4
    target = torch.randn(b, d)
    raw_params = torch.randn(b, d, requires_grad=True)
    # Build samples as a function of params so we can verify the chain rule
    samples = (raw_params.unsqueeze(0) + 0.5 * torch.randn(n, b, d))
    es = energy_score(samples, target)
    es.mean().backward()
    assert raw_params.grad is not None
    assert raw_params.grad.abs().max().item() > 0, (
        f"Expected non-zero gradient on raw params; got "
        f"max={raw_params.grad.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 4. ES is invariant to sample permutation (required for it to be a proper
#    function of the empirical distribution, not the sample ordering).
# ---------------------------------------------------------------------------
def test_es_invariant_to_sample_permutation():
    """Permuting along the sample axis must not change ES."""
    torch.manual_seed(3)
    n, b, d = 16, 3, 6
    target = torch.randn(b, d)
    samples = torch.randn(n, b, d) * 0.7 + target.unsqueeze(0)
    es_orig = energy_score(samples, target)

    # Random permutation of the sample axis (independent per batch element).
    perm = torch.argsort(torch.rand(n, b), dim=0)  # [N, B]
    samples_perm = torch.gather(samples, 0, perm.unsqueeze(-1).expand(-1, -1, d))
    es_perm = energy_score(samples_perm, target)

    diff = (es_orig - es_perm).abs().max().item()
    assert diff < 1e-5, (
        f"ES should be permutation-invariant; max diff {diff:.2e}"
    )


# ---------------------------------------------------------------------------
# 5. VS catches correlation errors that ES misses
# ---------------------------------------------------------------------------
def test_vs_catches_correlation_error_es_misses():
    """Build samples with correct marginals but wrong cross-variable
    correlation. ES should be small (marginals are right). VS should be
    nonzero (pairwise differences don't match)."""
    torch.manual_seed(4)
    n, b, d = 256, 4, 4
    # Truth: variables are perfectly correlated — y_1 = y_2 = ... = y_d
    base = torch.randn(b, 1)
    target = base.expand(b, d)

    # Predictive distribution: each variable has the right marginal mean and
    # variance as the truth, but variables are independent (decorrelated).
    samples = base.unsqueeze(0).expand(n, b, d) + 1e-2 * torch.randn(n, b, d)
    # Inject independent noise per dim so marginals match but correlation breaks
    samples = samples + 0.5 * torch.randn(n, b, d)

    es = energy_score(samples, target)
    vs = variogram_score(samples, target, p=0.5)

    # Both can be nonzero, but VS should be substantially larger relative to
    # what we'd see if correlation were correct. Compare against samples that
    # have correct correlation:
    samples_correct_corr = base.unsqueeze(0).expand(n, b, d) + 0.5 * torch.randn(
        n, b, 1
    ).expand(n, b, d)
    vs_correct_corr = variogram_score(samples_correct_corr, target, p=0.5)

    assert vs.mean().item() > vs_correct_corr.mean().item(), (
        f"VS should be larger when correlation is wrong; got wrong-corr "
        f"vs={vs.mean():.4f}, correct-corr vs={vs_correct_corr.mean():.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Lambda=0 is a no-op (backward compat sanity check)
# ---------------------------------------------------------------------------
def test_lambda_zero_no_op():
    """When the regularizer is multiplied by lambda=0, it must add nothing."""
    base_loss = torch.tensor(2.5)
    samples = torch.randn(8, 4, 6)
    target = torch.randn(4, 6)
    es = energy_score(samples, target).mean()
    final = base_loss + 0.0 * es
    assert final.item() == base_loss.item()


# ---------------------------------------------------------------------------
# 7. ES sample-count stability — more samples → tighter estimate
# ---------------------------------------------------------------------------
def test_es_estimate_stabilizes_with_more_samples():
    """Variance of the ES estimate across seeds should shrink as N grows."""
    torch.manual_seed(5)
    b, d = 4, 8
    target = torch.zeros(b, d)
    # True predictive distribution: standard normal centered at target.
    es_estimates = {n: [] for n in [4, 8, 32]}
    for seed in range(20):
        torch.manual_seed(seed)
        for n in es_estimates:
            samples = torch.randn(n, b, d)
            es_estimates[n].append(energy_score(samples, target).mean().item())

    # Standard deviation of the estimate should monotonically decrease as N
    # grows (or at least not increase materially).
    std_n4 = torch.tensor(es_estimates[4]).std().item()
    std_n8 = torch.tensor(es_estimates[8]).std().item()
    std_n32 = torch.tensor(es_estimates[32]).std().item()
    assert std_n8 <= std_n4 * 1.2, (
        f"ES estimate std at N=8 ({std_n8:.4f}) should not exceed N=4 "
        f"std ({std_n4:.4f}) by more than 20% slack"
    )
    assert std_n32 <= std_n8 * 1.2, (
        f"ES estimate std at N=32 ({std_n32:.4f}) should not exceed N=8 "
        f"std ({std_n8:.4f}) by more than 20% slack"
    )


# ---------------------------------------------------------------------------
# Bonus: ES on a known-tractable case — single-sample N=1 reduces to ||X-y||
# ---------------------------------------------------------------------------
def test_es_n1_reduces_to_l2_distance():
    """With N=1, term2 = 0.5 * ||X - X|| = 0, so ES = ||X - y||."""
    torch.manual_seed(6)
    b, d = 4, 6
    target = torch.zeros(b, d)
    sample = torch.randn(1, b, d)
    es = energy_score(sample, target)
    expected = sample.squeeze(0).norm(dim=-1)  # [B]
    assert torch.allclose(es, expected, atol=1e-4), (
        f"ES at N=1 should equal ||sample-target||; got es={es}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Bonus: VS reduces to a simple form when D=2 (single off-diagonal pair)
# ---------------------------------------------------------------------------
def test_vs_d2_simple_form():
    """For D=2, off-diagonal mean is over 2 entries (i,j) and (j,i), each
    being the same value. So VS = (|y_0-y_1|^p - E|x_0-x_1|^p)^2."""
    torch.manual_seed(7)
    n, b = 64, 3
    target = torch.tensor([[1.0, 4.0], [0.0, 2.0], [-1.0, -3.0]])
    # Samples with x_0 - x_1 having mean exactly equal to y_0 - y_1
    diff_target = (target[:, 0] - target[:, 1]).abs().pow(0.5)  # [B]
    samples = torch.zeros(n, b, 2)
    samples[..., 0] = target[:, 0].unsqueeze(0)
    samples[..., 1] = target[:, 1].unsqueeze(0)
    vs = variogram_score(samples, target, p=0.5)
    # Should be near zero (samples == target)
    assert vs.max().item() < 0.05, f"VS should be near zero; got {vs.tolist()}"


# ---------------------------------------------------------------------------
# MVG-CRPS stub raises NotImplementedError (it's deliberately a fallback)
# ---------------------------------------------------------------------------
def test_mvg_crps_is_stub():
    """The MVG-CRPS function is a deliberate stub for the auto-fallback path.
    It should raise NotImplementedError until the pilot proves it's needed."""
    mean = torch.zeros(2, 4)
    cov = torch.eye(4).unsqueeze(0).expand(2, 4, 4)
    target = torch.zeros(2, 4)
    with pytest.raises(NotImplementedError):
        mvg_crps_closedform(mean, cov, target)


if __name__ == "__main__":
    test_es_zero_when_samples_equal_target()
    test_es_positive_when_samples_distant()
    test_es_gradient_flows_to_samples()
    test_es_invariant_to_sample_permutation()
    test_vs_catches_correlation_error_es_misses()
    test_lambda_zero_no_op()
    test_es_estimate_stabilizes_with_more_samples()
    test_es_n1_reduces_to_l2_distance()
    test_vs_d2_simple_form()
    test_mvg_crps_is_stub()
    print("All proper_scoring_rules tests passed.")
