"""
Verification B — H1 probe: plot F(x|context) for the trained DSF marginal.

Goal
----
Determine whether the trained DSF marginal_conditioner has collapsed into a
near-step-function CDF — the leading hypothesis (H1) for why
DSFMarginal.inverse() returns ~0 regardless of input quantile u.

Approach
--------
1. Load the broken checkpoint's state_dict
2. Reconstruct ONLY the marginal_conditioner MLP and the DeepSigmoidFlow
3. Feed N synthetic post-LayerNorm context vectors (dim=256, drawn from a
   reasonable distribution that mimics the encoder output)
4. For each context vector, sweep x in [-3, 3] and compute
   F(x | params) = marginal_flow.forward_no_logdet(params, x)
5. Plot the curves on a single figure
6. Interpret: smooth sigmoid = healthy; near-step = collapsed (H1 confirmed)

Why synthetic context vectors are sufficient
--------------------------------------------
If the conditioner has truly collapsed, it produces pathological flow params
(huge a, narrow effective range) for any input distribution similar to its
training inputs. The flow encoder ends in a LayerNorm, so realistic context
vectors are approximately zero-mean unit-variance with skewed/structured
features. Random N(0, 1) is a reasonable proxy for an out-of-distribution but
plausible context.

A complementary test (run AFTER this probe) feeds REAL context vectors via a
full encoder forward pass to confirm that the same pathology shows up on
in-distribution inputs.
"""
import os
import sys
import argparse

import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")  # No display on HPC
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_transformer_ts.tactis_2.deep_sigmoid_flow import DeepSigmoidFlow


def build_marginal_conditioner_from_state_dict(sd: dict, prefix: str = "model.tactis.decoder.marginal.marginal_conditioner"):
    """Reconstruct the marginal_conditioner MLP layers from checkpoint state_dict.

    Layer pattern from checkpoint inspection:
      - layer 0: Linear(256 -> 32)
      - layer 1: ReLU (no params)
      - layer 2: Linear(32 -> 32)
      - layer 3: ReLU (no params)
      - layer 4: Linear(32 -> 432)

    For the broken checkpoint this means: context_dim=256, mlp_dim=32,
    mlp_layers=2 internal MLP linears + 1 output linear, total flow param
    length = 432 = 3 layers * 3 * 48.
    """
    keys = sorted([k for k in sd if k.startswith(prefix)])
    print(f"[builder] Found {len(keys)} marginal_conditioner keys")
    for k in keys:
        print(f"  {k}: {tuple(sd[k].shape)}")

    # Read shapes to figure out architecture
    layers = []
    layer_indices = sorted({int(k.split(".")[-2]) for k in keys})
    for idx in layer_indices:
        w_key = f"{prefix}.{idx}.weight"
        b_key = f"{prefix}.{idx}.bias"
        out_dim, in_dim = sd[w_key].shape
        lin = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            lin.weight.copy_(sd[w_key])
            lin.bias.copy_(sd[b_key])
        layers.append(lin)
        if idx != layer_indices[-1]:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def build_dsf_from_state_dict(sd: dict, n_layers: int = 3, hidden_dim: int = 48):
    """DeepSigmoidFlow has no trainable parameters apart from those in
    SigmoidFlow, which appear to also have none in our local impl. Verify by
    checking state_dict for marginal_flow.* keys."""
    flow_keys = [k for k in sd if "marginal_flow" in k]
    if flow_keys:
        print(f"[builder] WARNING: marginal_flow has trainable params: {flow_keys}")
    flow = DeepSigmoidFlow(n_layers=n_layers, hidden_dim=hidden_dim)
    return flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/aoife_validation")
    parser.add_argument("--num_contexts", type=int, default=12)
    parser.add_argument("--x_min", type=float, default=-4.0)
    parser.add_argument("--x_max", type=float, default=4.0)
    parser.add_argument("--num_x_points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[probe] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]
    mc = hp.get("model_config", {})
    print(f"[probe] Epoch: {ckpt.get('epoch')}, global_step: {ckpt.get('global_step')}")
    print(f"[probe] decoder_dsf_num_layers={mc.get('decoder_dsf_num_layers')}, "
          f"decoder_dsf_hidden_dim={mc.get('decoder_dsf_hidden_dim')}")

    # Reconstruct
    conditioner = build_marginal_conditioner_from_state_dict(sd)
    conditioner.eval()
    flow = build_dsf_from_state_dict(
        sd,
        n_layers=mc.get("decoder_dsf_num_layers", 3),
        hidden_dim=mc.get("decoder_dsf_hidden_dim", 48),
    )
    flow.eval()
    print(f"[probe] total_params_length: {flow.total_params_length}")

    # Get the input dim (= encoder output dim)
    context_dim = conditioner[0].in_features
    print(f"[probe] context_dim = {context_dim}")

    # Generate synthetic post-LayerNorm context vectors
    # Encoder ends with LayerNorm, so outputs are approximately zero-mean unit-variance
    contexts = torch.randn(args.num_contexts, context_dim)
    contexts = (contexts - contexts.mean(dim=-1, keepdim=True)) / (contexts.std(dim=-1, keepdim=True) + 1e-6)

    with torch.no_grad():
        marginal_params = conditioner(contexts)  # [N, 432]

    print(f"[probe] marginal_params shape: {tuple(marginal_params.shape)}")
    print(f"[probe] marginal_params stats: mean={marginal_params.mean():.4f}, "
          f"std={marginal_params.std():.4f}, "
          f"min={marginal_params.min():.4f}, max={marginal_params.max():.4f}")

    # Inspect the per-layer flow params: split into n_layers groups of 3*hidden_dim
    n_layers = flow.n_layers if hasattr(flow, "n_layers") else mc.get("decoder_dsf_num_layers", 3)
    hidden_dim = mc.get("decoder_dsf_hidden_dim", 48)
    params_per_layer = 3 * hidden_dim
    print(f"\n[probe] Per-layer flow param stats (a, b, w_pre):")
    for li in range(n_layers):
        layer_params = marginal_params[:, li * params_per_layer:(li + 1) * params_per_layer]
        a_pre = layer_params[:, :hidden_dim]
        b = layer_params[:, hidden_dim:2 * hidden_dim]
        w_pre = layer_params[:, 2 * hidden_dim:]
        # `a` after softplus determines steepness; the bigger, the sharper the sigmoid
        a_after_softplus = torch.nn.functional.softplus(a_pre) + 1e-6
        print(f"  Layer {li}:")
        print(f"    a_pre:     min={a_pre.min():.3f}, max={a_pre.max():.3f}, mean={a_pre.mean():.3f}")
        print(f"    a (softplus): min={a_after_softplus.min():.3f}, "
              f"max={a_after_softplus.max():.3f}, mean={a_after_softplus.mean():.3f}")
        print(f"    b:         min={b.min():.3f}, max={b.max():.3f}, mean={b.mean():.3f}")
        print(f"    w_pre:     min={w_pre.min():.3f}, max={w_pre.max():.3f}, mean={w_pre.mean():.3f}")

    # Sweep x and compute F(x | params) for each context
    x_grid = torch.linspace(args.x_min, args.x_max, args.num_x_points)  # [num_x_points]
    # forward_no_logdet expects params: [..., flow_param_length], x: [...]
    # Broadcasting: params=[N, 1, P], x=[1, num_x_points] → output [N, num_x_points]
    # But the local DSF expects more specific shapes; let's be explicit.
    # In dsf_marginal.py, marginal_params is [batch, N, param_len] and x is [batch, N].
    # Here we treat each context as a separate "variable" and do a single batch.
    # Reshape: marginal_params -> [1, num_contexts, P], x -> [1, num_contexts] but we need
    # ALL x values per context. Use the sample dimension trick from dsf_marginal:
    # add a singleton dimension to params → [1, num_contexts, 1, P], x → [1, num_contexts, num_x]
    params_b = marginal_params.unsqueeze(0).unsqueeze(2)  # [1, num_contexts, 1, 432]
    x_b = x_grid.unsqueeze(0).unsqueeze(0).expand(1, args.num_contexts, -1)  # [1, num_contexts, num_x]
    with torch.no_grad():
        cdf = flow.forward_no_logdet(params_b, x_b)  # [1, num_contexts, num_x]
    cdf = cdf.squeeze(0)  # [num_contexts, num_x]
    print(f"\n[probe] cdf shape: {tuple(cdf.shape)}")
    print(f"[probe] cdf stats: min={cdf.min():.4f}, max={cdf.max():.4f}")
    print(f"[probe] Per-context CDF range (max - min) — first 6:")
    cdf_range = cdf.max(dim=1).values - cdf.min(dim=1).values
    for ci in range(min(6, args.num_contexts)):
        print(f"  context {ci}: range = {cdf_range[ci].item():.4f}, "
              f"min = {cdf[ci].min().item():.4f}, max = {cdf[ci].max().item():.4f}")

    # Sigmoid transform if the flow output is the logit (n_layers > 0 and the LAST layer
    # has no_logit=True, so output IS the CDF in (0, 1). Confirmed by reading the local DSF code.
    cdf_sigmoid = cdf  # already in (0, 1) per local impl

    # Estimate transition width: the x-range over which CDF crosses 0.1 → 0.9
    transition_widths = []
    for ci in range(args.num_contexts):
        c = cdf_sigmoid[ci].numpy()
        i_low = (c > 0.1).argmax() if (c > 0.1).any() else 0
        i_high = (c > 0.9).argmax() if (c > 0.9).any() else len(c) - 1
        if i_high > i_low:
            tw = (x_grid[i_high] - x_grid[i_low]).item()
            transition_widths.append(tw)
        else:
            transition_widths.append(float("nan"))

    transition_widths = torch.tensor([t for t in transition_widths if t == t])
    if len(transition_widths) > 0:
        print(f"\n[probe] Transition width (x-range from F=0.1 to F=0.9):")
        print(f"  mean: {transition_widths.mean():.4f}")
        print(f"  median: {transition_widths.median():.4f}")
        print(f"  min: {transition_widths.min():.4f}")
        print(f"  max: {transition_widths.max():.4f}")

    # Verdict
    median_tw = transition_widths.median().item() if len(transition_widths) > 0 else float("nan")
    print("\n[probe] === H1 VERDICT ===")
    if median_tw == median_tw:  # not nan
        if median_tw < 0.1:
            print("  H1 CONFIRMED: trained DSF has collapsed to near-step CDF")
            print(f"  (median transition width {median_tw:.4f} < 0.1)")
            print("  Implication: this checkpoint cannot be salvaged by code fixes.")
            print("  Remediation: roll back to earlier epoch OR fine-tune with regularization.")
        elif median_tw < 0.5:
            print(f"  H1 PARTIAL: DSF is sharp but not pathological (median tw {median_tw:.4f})")
        else:
            print(f"  H1 REFUTED: DSF curves are smooth (median tw {median_tw:.4f})")
            print("  Implication: bug is elsewhere — check copula path, U values into inverse, etc.")
    else:
        print("  H1 INDETERMINATE: CDFs don't span [0.1, 0.9] in the probe range")

    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    for ci, ax in enumerate(axes.flat):
        if ci >= args.num_contexts:
            ax.axis("off")
            continue
        ax.plot(x_grid.numpy(), cdf_sigmoid[ci].numpy(), linewidth=2)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5)
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.5)
        ax.set_title(f"Context {ci}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"F(x | context) — DSF marginal probe — epoch={ckpt.get('epoch', '?')}\n"
        f"Median transition width (F=0.1→0.9): {median_tw:.4f}\n"
        "Healthy: smooth sigmoid spans most of the range. Collapsed: near-step at the median.",
        fontsize=11,
    )
    fig.text(0.5, 0.02, "x (in StdScaler-normalized units)", ha="center", fontsize=10)
    fig.text(0.02, 0.5, "F(x | context)", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.93])

    out_path = os.path.join(args.output_dir, "dsf_cdf_probe.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[probe] Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
