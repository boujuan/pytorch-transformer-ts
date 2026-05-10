"""
Reproduce Aoife's exact bug-find on a v2 checkpoint.

Aoife's broken-model finding:
  samples[:, 0, 0, :].mean(0) → tensor([0.9431, 0.9431, 0.9431, 0.9431])
  i.e., 200 samples were 200 copies of the same value across the 4-step horizon.

Pass criterion (this script):
  samples[:, 0, 0, :] should be 200 *different* values
  → mean across samples should NOT be 4 identical numbers
  → sample-axis std mean > 0.05

Uses the FULL inference path (Stage 2 with trained copula), not the
Stage-1-only probe shortcut.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dump_path", required=True,
                        help="Path to one of the existing forecast_*.pt dumps")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Match Aoife's 200 samples")
    args = parser.parse_args()

    print(f"[aoife_test] Loading: {args.checkpoint}")
    lit = TACTiS2LightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu", strict=False)
    lit.eval()
    lit.to("cpu")

    # Force Stage 2 — proper copula path
    if hasattr(lit, "model") and hasattr(lit.model, "tactis"):
        lit.model.tactis.set_stage(2)
        lit.model.tactis.skip_copula = False
        print(f"[aoife_test] forced stage=2, skip_copula=False")

    dump = torch.load(args.dump_path, map_location="cpu", weights_only=False)
    past_target = dump["past_target"]
    pred_len = lit.hparams.model_config.get("prediction_length", 4)
    batch, num_series, ctx_len = past_target.shape
    print(f"[aoife_test] past_target shape: {tuple(past_target.shape)}, pred_len={pred_len}")

    # Build hist_time / pred_time
    hist_time = torch.arange(ctx_len).unsqueeze(0).expand(batch, -1).float()
    pred_time = torch.arange(ctx_len, ctx_len + pred_len).unsqueeze(0).expand(batch, -1).float()

    # Use first 2 batch items to keep it tractable on CPU
    n = 2
    print(f"[aoife_test] Sampling {args.num_samples} samples on {n} batch items × "
          f"{num_series} vars × {pred_len} pred steps (CPU — slow)")
    with torch.no_grad():
        samples = lit.model.tactis.sample(
            num_samples=args.num_samples,
            hist_time=hist_time[:n],
            hist_value=past_target[:n],
            pred_time=pred_time[:n],
        )

    # samples shape per probe_real_context convention:
    # (num_samples, batch, vars, pred_len)
    print(f"\n[aoife_test] Sample tensor shape: {tuple(samples.shape)}")

    # === Aoife's exact slice ===
    print("\n" + "=" * 70)
    print("AOIFE TEST: samples[:, 0, 0, :].mean(0)")
    print("=" * 70)
    aoife_slice = samples[:, 0, 0, :]
    aoife_mean = aoife_slice.mean(0)
    aoife_std  = aoife_slice.std(0)
    print(f"  Mean across {samples.shape[0]} samples: {aoife_mean.tolist()}")
    print(f"  Std  across {samples.shape[0]} samples: {aoife_std.tolist()}")
    print(f"  Min sample value: {aoife_slice.min():.4f}")
    print(f"  Max sample value: {aoife_slice.max():.4f}")
    print(f"  Range:            {(aoife_slice.max() - aoife_slice.min()):.4f}")

    # Distinct values check (Aoife's broken model had ALL identical)
    unique_per_step = [len(torch.unique(aoife_slice[:, t])) for t in range(aoife_slice.shape[1])]
    print(f"  Distinct values per pred step: {unique_per_step} (out of {samples.shape[0]})")

    # === Sample-axis std summary across all (batch, var, pred_step) cells ===
    print("\n" + "=" * 70)
    print("FULL SAMPLE-AXIS STD SUMMARY (Aoife's metric)")
    print("=" * 70)
    sa_std = samples.std(dim=0)  # (batch, vars, pred_len)
    print(f"  shape: {tuple(sa_std.shape)}")
    print(f"  mean:   {sa_std.mean():.6f}")
    print(f"  median: {sa_std.median():.6f}")
    print(f"  max:    {sa_std.max():.6f}")
    print(f"  min:    {sa_std.min():.6f}")
    print(f"  fraction > 0.001 (not collapsed):   {(sa_std > 0.001).float().mean().item():.4f}")
    print(f"  fraction > 0.05  (Aoife threshold): {(sa_std > 0.05).float().mean().item():.4f}")

    # === Verdict ===
    print("\n" + "=" * 70)
    pass_aoife = sa_std.mean().item() > 0.05
    pass_distinct = all(u > 1 for u in unique_per_step)
    if pass_aoife and pass_distinct:
        print("✅ PASS — Aoife's bug is FIXED")
        print(f"   sa_std mean = {sa_std.mean():.6f} > 0.05 threshold")
        print(f"   Each pred step has {min(unique_per_step)}+ distinct sample values (was 1 before)")
    else:
        print("❌ FAIL")
        if not pass_aoife:
            print(f"   sa_std mean = {sa_std.mean():.6f} ≤ 0.05 threshold (still collapsed)")
        if not pass_distinct:
            print(f"   Some pred step has only 1 distinct value across {samples.shape[0]} samples")
    print("=" * 70)


if __name__ == "__main__":
    main()
