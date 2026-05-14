"""
Structured side-by-side checkpoint comparison.

Loads N checkpoints, runs probe_real_context-style F^-1(U) extraction on each,
and prints a comparison table. Anchors to Aoife's downstream metric (sample-axis
std) where possible.

Usage:
    python probe_compare_runs.py \\
        --dump /path/to/forecast_00000.pt \\
        --label v1_epoch99 --ckpt /path/to/v1_ep99.ckpt \\
        --label v2_a3      --ckpt /path/to/v2_amax3_ep14.ckpt \\
        --label v2_a5      --ckpt /path/to/v2_amax5_ep14.ckpt \\
        ...
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule


def probe_one(ckpt_path: str, dump_path: str, label: str):
    """Returns dict of stats for one checkpoint."""
    lit = TACTiS2LightningModule.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
    lit.eval()
    lit.to("cpu")
    if hasattr(lit, "model") and hasattr(lit.model, "tactis"):
        lit.model.tactis.stage = 2
        lit.model.tactis.skip_copula = False

    captured_input = []
    def hook(module, args, kwargs):
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            captured_input.append(args[0].detach().cpu())
    decoder = lit.model.tactis.decoder
    handle = decoder.marginal.marginal_conditioner.register_forward_pre_hook(hook, with_kwargs=True)

    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    past_target = dump["past_target"]
    pred_len = lit.hparams.model_config.get("prediction_length", 4)
    batch, num_series, ctx_len = past_target.shape

    n = 2
    hist_time = torch.arange(ctx_len).unsqueeze(0).expand(batch, -1).float()[:n]
    pred_time = torch.arange(ctx_len, ctx_len + pred_len).unsqueeze(0).expand(batch, -1).float()[:n]
    hist_value = past_target[:n]

    with torch.no_grad():
        try:
            samples = lit.model.tactis.sample(num_samples=4, hist_time=hist_time,
                                              hist_value=hist_value, pred_time=pred_time)
        except Exception as e:
            handle.remove()
            return {"label": label, "error": f"{type(e).__name__}: {e}"}

    handle.remove()

    if not captured_input:
        return {"label": label, "error": "no marginal_conditioner capture"}

    ctx0 = captured_input[0]
    ctx_flat = ctx0.reshape(-1, ctx0.shape[-1])
    contexts = ctx_flat[:10]
    contexts_b = contexts.unsqueeze(0)

    with torch.no_grad():
        mp = decoder.marginal.marginal_conditioner(contexts)
        u_vals = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90])
        u_b = u_vals.unsqueeze(0).unsqueeze(0).expand(1, 10, -1)
        x_real = decoder.marginal.inverse(contexts_b, u_b)

    u_axis_std = x_real.std(dim=-1)  # per-context spread across U values

    # Sample-axis std on the 4-sample tactis.sample output
    # samples shape: (num_samples, batch, var, pred_len) per probe_real_context conventions
    sa_std_full = samples.std(dim=0)  # (batch, var, pred_len)

    return {
        "label": label,
        "ckpt": os.path.basename(ckpt_path),
        "fi_mean": float(u_axis_std.mean()),
        "fi_median": float(u_axis_std.median()),
        "fi_max": float(u_axis_std.max()),
        "fi_min": float(u_axis_std.min()),
        "mp_mean": float(mp.mean()),
        "mp_max": float(mp.max()),
        "mp_min": float(mp.min()),
        "sample_axis_std_mean": float(sa_std_full.mean()),
        "sample_axis_std_max": float(sa_std_full.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="Path to a forecast_*.pt dump")
    parser.add_argument("--label", action="append", default=[], help="Per-ckpt label (repeatable)")
    parser.add_argument("--ckpt", action="append", default=[], help="Per-ckpt path (repeatable, must match --label count)")
    args = parser.parse_args()

    if len(args.label) != len(args.ckpt):
        print(f"ERROR: --label count ({len(args.label)}) != --ckpt count ({len(args.ckpt)})")
        sys.exit(1)

    rows = []
    for label, ckpt in zip(args.label, args.ckpt):
        print(f"[probe_compare] {label}: {ckpt}", flush=True)
        rows.append(probe_one(ckpt, args.dump, label))

    print()
    print("=" * 100)
    print(f"{'label':<28} {'F⁻¹ mean':>10} {'F⁻¹ med':>10} {'F⁻¹ max':>10} {'F⁻¹ min':>10} {'sa_std mean':>14} {'mp_max':>10}")
    print("-" * 100)
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<28} ERROR: {r['error']}")
            continue
        print(f"{r['label']:<28} {r['fi_mean']:>10.4f} {r['fi_median']:>10.4f} {r['fi_max']:>10.4f} {r['fi_min']:>10.4f} {r['sample_axis_std_mean']:>14.6f} {r['mp_max']:>10.2f}")
    print("=" * 100)
    print("Healthy thresholds: F⁻¹ mean > 0.5, F⁻¹ median > 0.5 (broad CDF), sa_std mean > 0.05 (Aoife's)")
    print()


if __name__ == "__main__":
    main()
