"""
Verification B-2 — Direct test of DSFMarginal.inverse() with the TRAINED
conditioner weights, using U values matching the actual inference distribution.

Why this is sharper than B-1
----------------------------
B-1 plotted F(x|context) for x in [-3, 3] and looked at transition width.
B-2 takes the same trained conditioner and synthetic contexts, but:
  1. Computes F^(-1)(U) for U values matching the empirical inference dist
     ([0.05, 0.3] range from the dumps)
  2. Reports the resulting X values directly

If F^(-1)(U) collapses to ~0 even on SYNTHETIC contexts, then the trained
marginal is pathological for any plausible context — confirming H1 strongly.
If F^(-1)(U) is diverse on synthetic contexts but collapsed on real contexts,
the bug is more subtle (context-dependent collapse) — still H1 but localized.
"""
import os
import sys
import argparse

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_transformer_ts.tactis_2.deep_sigmoid_flow import DeepSigmoidFlow
from pytorch_transformer_ts.tactis_2.dsf_marginal import DSFMarginal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_contexts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]
    mc = hp["model_config"]

    # Build the full DSFMarginal module from checkpoint config + state_dict
    dsf = DSFMarginal(
        context_dim=256,
        mlp_layers=mc["decoder_mlp_num_layers"],
        mlp_dim=mc["decoder_mlp_hidden_dim"],
        flow_layers=mc["decoder_dsf_num_layers"],
        flow_hid_dim=mc["decoder_dsf_hidden_dim"],
    )
    # Load conditioner weights
    prefix = "model.tactis.decoder.marginal.marginal_conditioner"
    relabeled = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            local_k = "marginal_conditioner" + k[len(prefix):]
            relabeled[local_k] = v
    missing, unexpected = dsf.load_state_dict(relabeled, strict=False)
    print(f"[probe2] Loaded conditioner. Missing: {missing}, Unexpected: {unexpected}")
    dsf.eval()

    # Synthetic context: post-LayerNorm-like
    contexts = torch.randn(1, args.num_contexts, 256)
    contexts = (contexts - contexts.mean(dim=-1, keepdim=True)) / (contexts.std(dim=-1, keepdim=True) + 1e-6)

    # U values matching the empirical inference distribution
    # From the deep_diagnosis report: U mean=0.235, std=0.20
    # Distribution: roughly low-end concentrated, range [0.04, 0.32]
    # We'll test multiple regimes:
    print("\n[probe2] === Test A: U values from empirical low-quantile distribution ===")
    u_low = torch.tensor([0.046, 0.088, 0.099, 0.155, 0.220, 0.229, 0.321])
    u_low_b = u_low.unsqueeze(0).unsqueeze(0).expand(1, args.num_contexts, -1)  # [1, num_contexts, 7]
    with torch.no_grad():
        x_low = dsf.inverse(contexts, u_low_b)  # [1, num_contexts, 7]
    print(f"  X values from F^-1 for U in [0.046, 0.321]:")
    print(f"    shape: {tuple(x_low.shape)}")
    print(f"    overall mean: {x_low.mean():.4f}, std: {x_low.std():.4f}")
    print(f"    range: [{x_low.min():.4f}, {x_low.max():.4f}]")
    print(f"    Per-context X for first 5 contexts:")
    for ci in range(min(5, args.num_contexts)):
        print(f"      ctx {ci}: {x_low[0, ci].tolist()}")

    # Compute std ACROSS the U-axis (within a single context) — this measures
    # how well the inverse spreads diverse U values onto diverse X values
    u_axis_std = x_low.std(dim=-1)  # [1, num_contexts]
    print(f"\n  STD across U-axis (per context): mean={u_axis_std.mean():.4f}, "
          f"median={u_axis_std.median():.4f}, max={u_axis_std.max():.4f}, min={u_axis_std.min():.4f}")

    print("\n[probe2] === Test B: Uniform U values [0.1, 0.3, 0.5, 0.7, 0.9] ===")
    u_uniform = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    u_uniform_b = u_uniform.unsqueeze(0).unsqueeze(0).expand(1, args.num_contexts, -1)
    with torch.no_grad():
        x_uniform = dsf.inverse(contexts, u_uniform_b)
    print(f"  X values for U in [0.1, 0.3, 0.5, 0.7, 0.9]:")
    print(f"    overall mean: {x_uniform.mean():.4f}, std: {x_uniform.std():.4f}")
    print(f"    range: [{x_uniform.min():.4f}, {x_uniform.max():.4f}]")
    for ci in range(min(5, args.num_contexts)):
        print(f"      ctx {ci}: {x_uniform[0, ci].tolist()}")

    u_axis_std_uniform = x_uniform.std(dim=-1)
    print(f"\n  STD across U-axis (per context): mean={u_axis_std_uniform.mean():.4f}, "
          f"median={u_axis_std_uniform.median():.4f}, max={u_axis_std_uniform.max():.4f}")

    # === Verdict ===
    print("\n[probe2] === VERDICT ===")
    print(f"Empirical inference X-axis std (from dumps): ~3e-4 (collapsed)")
    print(f"Synthetic-context X-axis std (low-quant U):  {u_axis_std.mean():.4f}")
    print(f"Synthetic-context X-axis std (uniform U):    {u_axis_std_uniform.mean():.4f}")
    print()
    if u_axis_std.mean() < 0.01:
        print("  CONFIRMED: collapse reproduces on SYNTHETIC contexts → trained DSF is broken globally.")
        print("  The inverse cannot spread diverse U values onto diverse X values for ANY plausible context.")
        print("  Implication: the marginal_conditioner has collapsed during training.")
    elif u_axis_std.mean() < 0.1:
        print("  PARTIAL: inverse is constrained but still spreads (synthetic ~mid-magnitude).")
        print("  Likely the REAL inference contexts are particularly bad — context-dependent collapse.")
        print("  Recommend: re-run with REAL context vectors via full inference forward pass.")
    else:
        print("  DSF inverse is healthy on synthetic contexts → the issue is real-context-specific OR upstream.")
        print("  Suspect: encoder output distribution at inference differs from training time.")


if __name__ == "__main__":
    main()
