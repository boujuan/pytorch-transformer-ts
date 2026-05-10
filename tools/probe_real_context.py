"""
Verification B-3 — Probe with REAL inference context vectors (not synthetic).

Loads the broken checkpoint via TACTiS2LightningModule, runs CPU inference on
past_target taken from existing dump tensors, and captures the actual context
that flows into marginal_conditioner. Then probes F^(-1)(U) on those.

If the inverse collapses on REAL contexts but works on synthetic ones, the
trained marginal isn't globally broken — instead, the encoder produces a
specific kind of context at inference that triggers the pathological CDF
region.
"""
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dump_path", required=True,
                        help="Path to one of the existing forecast_*.pt dumps")
    parser.add_argument("--num_contexts_to_show", type=int, default=10)
    args = parser.parse_args()

    print(f"[probe3] Loading checkpoint via LightningModule: {args.checkpoint}")
    lit = TACTiS2LightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu", strict=False)
    lit.eval()
    lit.to("cpu")

    # Force stage 2 (full flow + copula path)
    if hasattr(lit, "model") and hasattr(lit.model, "tactis"):
        lit.model.tactis.stage = 2
        lit.model.tactis.skip_copula = False
        print(f"[probe3] Set stage=2 on lit.model.tactis")
    elif hasattr(lit, "tactis"):
        lit.tactis.stage = 2
        lit.tactis.skip_copula = False

    # Hook into marginal_conditioner to capture its INPUT
    captured_input = []
    def hook(module, args, kwargs):
        # args is a tuple containing the input tensor(s)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            captured_input.append(args[0].detach().cpu())

    decoder = lit.model.tactis.decoder
    handle = decoder.marginal.marginal_conditioner.register_forward_pre_hook(hook, with_kwargs=True)

    # Load past_target from dump
    print(f"[probe3] Loading dump: {args.dump_path}")
    dump = torch.load(args.dump_path, map_location="cpu", weights_only=False)
    past_target = dump["past_target"]  # [batch, series, context_length]
    loc = dump["loc"]                  # [batch, 1, series]
    scale = dump["scale"]              # [batch, 1, series]
    batch, num_series, ctx_len = past_target.shape
    pred_len = lit.model.tactis.decoder.marginal_conditioner.in_features if hasattr(
        lit.model.tactis.decoder, "marginal_conditioner") else None
    pred_len = lit.hparams.model_config.get("prediction_length", 4)
    print(f"[probe3] past_target shape: {tuple(past_target.shape)}, pred_len={pred_len}")

    # Construct minimal inputs that output_params expects
    # We need to call self.tactis.sample directly with hist_time, hist_value, pred_time
    hist_time = torch.arange(ctx_len).unsqueeze(0).expand(batch, -1).float()  # [batch, ctx_len]
    pred_time = torch.arange(ctx_len, ctx_len + pred_len).unsqueeze(0).expand(batch, -1).float()

    # Use only first 2 batch items to keep it fast on CPU
    n = 2
    hist_time = hist_time[:n]
    pred_time = pred_time[:n]
    hist_value = past_target[:n]  # [n, series, ctx_len]

    print(f"[probe3] Running tactis.sample on {n} batch items, num_samples=4 (CPU, will be slow)")
    with torch.no_grad():
        try:
            samples = lit.model.tactis.sample(
                num_samples=4,
                hist_time=hist_time,
                hist_value=hist_value,
                pred_time=pred_time,
            )
            print(f"[probe3] Sample output shape: {tuple(samples.shape)}")
            print(f"[probe3] Sample values (first batch, first series, first sample): {samples[0, 0, 0, :].tolist()}")
        except Exception as e:
            print(f"[probe3] Sampling error: {type(e).__name__}: {e}")

    handle.remove()

    print(f"\n[probe3] Captured {len(captured_input)} marginal_conditioner forward calls")
    if not captured_input:
        print("  No captures — model didn't reach marginal_conditioner. Check stage settings.")
        return

    for ci, ctx in enumerate(captured_input):
        print(f"\n[probe3] Capture {ci}: shape={tuple(ctx.shape)}")
        print(f"  mean={ctx.mean():.4f}, std={ctx.std():.4f}, min={ctx.min():.4f}, max={ctx.max():.4f}")
        if ctx.dim() >= 2:
            # Per-vector statistics
            per_vec_mean = ctx.flatten(0, -2).mean(dim=-1)
            per_vec_std = ctx.flatten(0, -2).std(dim=-1)
            print(f"  per-vector mean: range [{per_vec_mean.min():.4f}, {per_vec_mean.max():.4f}], avg {per_vec_mean.mean():.4f}")
            print(f"  per-vector std:  range [{per_vec_std.min():.4f}, {per_vec_std.max():.4f}], avg {per_vec_std.mean():.4f}")

    # Save real contexts for downstream use
    out_path = os.path.join(os.path.dirname(args.dump_path), "real_contexts_capture.pt")
    torch.save({
        "captured_inputs": captured_input,
    }, out_path)
    print(f"\n[probe3] Saved captured contexts to: {out_path}")

    # Now use first capture to do the inverse probe with real contexts
    if captured_input:
        ctx0 = captured_input[0]  # [batch, ?, 256] or similar
        print(f"\n[probe3] === Real-context inverse probe (using capture 0) ===")
        # Flatten to [N, 256]
        ctx_flat = ctx0.reshape(-1, ctx0.shape[-1])
        # Take first num_contexts_to_show
        contexts = ctx_flat[:args.num_contexts_to_show]
        print(f"[probe3] Using {contexts.shape[0]} real context vectors")

        # Reuse marginal_conditioner from the loaded model
        with torch.no_grad():
            mp = decoder.marginal.marginal_conditioner(contexts)
        print(f"[probe3] marginal_params shape: {tuple(mp.shape)}, "
              f"stats: mean={mp.mean():.3f}, std={mp.std():.3f}, "
              f"min={mp.min():.3f}, max={mp.max():.3f}")

        # Inverse on a few U values
        u_vals = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90])
        # u shape needs to match: marginal expects context [batch, N, dim] and u [batch, N, samples]
        contexts_b = contexts.unsqueeze(0)  # [1, num_contexts, 256]
        u_b = u_vals.unsqueeze(0).unsqueeze(0).expand(1, args.num_contexts_to_show, -1)  # [1, num_contexts, 7]

        with torch.no_grad():
            x_real = decoder.marginal.inverse(contexts_b, u_b)
        print(f"\n[probe3] Real-context F^-1(U):")
        print(f"  shape: {tuple(x_real.shape)}")
        print(f"  overall mean: {x_real.mean():.4f}, std: {x_real.std():.4f}")
        print(f"  range: [{x_real.min():.4f}, {x_real.max():.4f}]")
        for ci in range(min(5, args.num_contexts_to_show)):
            print(f"    ctx {ci}: U={u_vals.tolist()}")
            print(f"           X={x_real[0, ci].tolist()}")

        u_axis_std = x_real.std(dim=-1)
        print(f"\n  STD across U-axis (per real context): mean={u_axis_std.mean():.4f}, "
              f"median={u_axis_std.median():.4f}, max={u_axis_std.max():.4f}, min={u_axis_std.min():.4f}")

        if u_axis_std.mean() < 0.01:
            print("\n  ✗ COLLAPSE REPRODUCED ON REAL CONTEXTS")
            print("  The inference-time encoder output drives the marginal into degenerate territory.")
        else:
            print(f"\n  Real-context inverse spreads OK (std {u_axis_std.mean():.4f}).")
            print("  The collapse must be elsewhere — re-investigate.")


if __name__ == "__main__":
    main()
