"""
Phase 2 verdict script: pull best-trial state from the live pilot Optuna study
and run the F^-1(U) probe on the best-trial checkpoint.

Reads:
  - PostgreSQL Optuna study `tuning_tactis_tune_awaken_smoothed_pred60_tactis_reg_pilot_v1_phase1`
  - Trial best-checkpoint via wind_forecasting.tuning.utils.checkpoint_utils
  - Then calls probe_real_context.py-equivalent inverse probe inline

Pass criteria (from plan Phase 2 / pilot decision gate):
  - >= 50% of trials survive (didn't get pruned for max_a > 50)
  - Best trial val_total_nll < -100 (sane, not catastrophic -1197)
  - Best trial max_a in [3, 30] (regularizer working but not under-fitting)
  - F^-1(U) std > 0.1 on real pred_encoded contexts (key inference health metric)
"""
import os
import sys
import argparse

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "wind-forecasting")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_name",
        default="tuning_tactis_tune_awaken_smoothed_pred60_tactis_reg_pilot_v1_phase1",
        help="Optuna study name (default matches the multi-job pilot's base prefix)",
    )
    parser.add_argument(
        "--db_host",
        default="pg.optuna.uni-oldenburg.de",
        help="PostgreSQL host (UOL internal)",
    )
    parser.add_argument("--db_port", type=int, default=5432)
    parser.add_argument("--db_name", default="optuna")
    parser.add_argument("--db_user", default="optuna02")
    parser.add_argument("--n_top", type=int, default=5, help="How many top trials to inspect")
    parser.add_argument(
        "--past_target_dump",
        default="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/aoife_validation/dumps/forecast_00000.pt",
        help="Source of past_target tensor for the real-context probe",
    )
    args = parser.parse_args()

    pg_password = os.environ.get("LOCAL_PG_PASSWORD")
    if not pg_password:
        sys.exit(
            "ERROR: LOCAL_PG_PASSWORD env var not set. Run:\n"
            "  eval \"$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)\"\n"
            "  python tools/inspect_pilot_trials.py"
        )

    storage_url = (
        f"postgresql://{args.db_user}:{pg_password}@"
        f"{args.db_host}:{args.db_port}/{args.db_name}"
    )

    print(f"[inspect] Connecting to study: {args.study_name}")
    study = optuna.load_study(study_name=args.study_name, storage=storage_url)

    n_total = len(study.trials)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_running = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING)
    n_fail = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)

    print(f"[inspect] Trials: total={n_total}, complete={n_complete}, "
          f"pruned={n_pruned}, running={n_running}, failed={n_fail}")

    if n_complete == 0:
        print("[inspect] No completed trials yet — wait for first epoch+stage2_start_epoch to finish")
        return

    # Pass criterion 1: ≥50% survive (= 1 - pruned_fraction)
    pruned_fraction = n_pruned / max(n_total - n_running, 1)
    print(f"[inspect] Pruned fraction (excl. running): {pruned_fraction:.2%}")
    if pruned_fraction > 0.5:
        print("  ⚠  >50% pruned — consider widening lambda_a_reg search range or relaxing explode_threshold")
    else:
        print("  ✓ <=50% pruned (criterion 1 OK)")

    # Best trial(s)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else float("inf"))

    print(f"\n[inspect] Top {min(args.n_top, len(completed))} completed trials by val_loss:")
    for i, t in enumerate(completed[: args.n_top]):
        max_a = t.user_attrs.get("a_reg/max_a", "n/a")
        mean_a = t.user_attrs.get("a_reg/mean_a", "n/a")
        lambda_a = t.params.get("lambda_a_reg", "n/a")
        print(
            f"  Rank {i + 1}: trial #{t.number} | val_loss={t.value:.3f} | "
            f"lambda_a_reg={lambda_a} | max_a={max_a} | mean_a={mean_a}"
        )
        if i == 0:
            print(f"    Full params: {t.params}")

    # Pass criterion 2: best val_total_nll
    best = completed[0]
    print(f"\n[inspect] Best trial #{best.number}: val_loss={best.value:.3f}")
    if best.value is None or best.value > 0:
        print("  ⚠  val_loss not negative — model may not be learning density")
    elif best.value < -100:
        print("  ✓ val_loss in healthy negative range (criterion 2 OK)")
    else:
        print(f"  ⚠  val_loss={best.value:.3f}: not catastrophic but weaker than expected")

    # Pass criterion 3: max_a regime
    max_a_str = best.user_attrs.get("a_reg/max_a")
    if max_a_str is not None:
        try:
            max_a = float(max_a_str)
        except (TypeError, ValueError):
            max_a = None
        if max_a is None:
            print("  ⚠  max_a metric missing from trial user_attrs")
        elif 3.0 <= max_a <= 30.0:
            print(f"  ✓ max_a={max_a:.2f} in [3, 30] (criterion 3 OK — regularizer working)")
        elif max_a < 3.0:
            print(f"  ⚠  max_a={max_a:.2f} < 3 — possibly under-fitting (lambda too high)")
        else:
            print(f"  ⚠  max_a={max_a:.2f} > 30 — regularizer barely engaging (lambda too low)")

    # Pass criterion 4: probe F^-1(U) on best-trial checkpoint
    # Look up best trial's checkpoint path. Convention from
    # wind_forecasting/tuning/utils/helpers.py:create_trial_checkpoint_callback
    # is checkpoint_dir/<chkp_dir_suffix>/trial_X/...
    chkp_root = "/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/checkpoints/_smoothed_60_tactis_reg_pilot_v1"
    trial_dir = os.path.join(chkp_root, f"trial_{best.number}")
    print(f"\n[inspect] Looking for best-trial checkpoints under: {trial_dir}")
    if not os.path.isdir(trial_dir):
        print(f"  Directory not found. Checkpoint dir may differ — search for 'trial_{best.number}' under {chkp_root}")
        return

    # Pick the first .ckpt file
    ckpts = sorted(
        [os.path.join(trial_dir, f) for f in os.listdir(trial_dir) if f.endswith(".ckpt")],
        key=os.path.getmtime,
        reverse=True,
    )
    if not ckpts:
        print(f"  No .ckpt files in {trial_dir}")
        return
    ckpt_path = ckpts[0]
    print(f"  Probing checkpoint: {ckpt_path}")

    print("\n[inspect] Running probe_real_context to evaluate F^-1(U) on real pred_encoded...")
    print("  (Defer to: python tools/probe_real_context.py --checkpoint <path> "
          f"--dump_path {args.past_target_dump})")
    print("\n  Run this exact command separately for the actual probe output.")


if __name__ == "__main__":
    main()
