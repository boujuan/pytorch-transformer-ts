"""
Custom Lightning callbacks for TACTiS-2 training.

Provides:
- MarginalHealthMonitor: monitors the DeepSigmoidFlow's `a` (sigmoid-steepness)
  parameter during stage 1 training and optionally signals Optuna to prune any
  trial whose `a` explodes (the empirically-observed collapse signature).

Why this callback exists
------------------------
The TACTiS-2 marginal flow training has a known failure mode: without
regularization, the optimizer drives the DSF's `a` parameter unbounded
(empirically: ~0.84 at epoch 2 → ~720 by epoch 90), producing a degenerate
near-step-function CDF. The `val_loss` metric *rewards* this collapse (a near
step CDF gives near-infinite density at training data points → very negative
NLL). So minimizing `val_loss` alone in Optuna will rediscover the collapsed
minimum.

This callback adds a guard: trials whose `a_reg/max_a` (logged by
TACTiS2LightningModule.training_step when `lambda_a_reg > 0`) exceeds a
threshold are pruned, preventing Optuna's sampler from re-exploring the
collapsed region.

For full diagnostic context, see:
/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/tactis_inference_deep_diagnosis.md
"""
import logging
from typing import Any, Optional

import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class MarginalHealthMonitor(pl.Callback):
    """
    Callback that monitors the DSF marginal's `a` parameter during stage 1
    training and prunes Optuna trials whose `a` exceeds a sanity threshold.

    Parameters
    ----------
    optuna_trial
        Optional Optuna `Trial` object. If provided, the callback will call
        `trial.report()` and `trial.should_prune()` at the end of each stage-1
        training epoch. If `should_prune()` returns True OR the explosion
        threshold is exceeded, raises `optuna.TrialPruned`.
    explode_threshold
        Maximum tolerated value of `a_reg/max_a` (post-softplus). Default 50.0.
        Empirical reference: healthy epoch=2 had max_a ≈ 0.84; collapsed
        epoch=90+ had max_a ≈ 720. A threshold of 50 leaves headroom for
        sharp-but-healthy training while rejecting the collapsed minimum.
    metric_name
        Metric key to read from `trainer.callback_metrics`. Default
        `"a_reg/max_a_epoch"` matching the per-epoch logging from
        TACTiS2LightningModule.training_step (Lightning appends "_epoch" by
        default when `on_epoch=True`; if your version doesn't, set this to
        `"a_reg/max_a"`).
    only_stage
        Only fire when `pl_module.stage == only_stage`. Default 1.
        In stage 2 the marginal is frozen so the metric won't change.
    objective_metric
        Metric to report to Optuna's `trial.report()` for sampler convergence
        feedback. Default `"val_total_nll_epoch"`. If missing, falls back to
        `"train_total_nll_epoch"` then `"train_loss_epoch"`.
    """

    def __init__(
        self,
        optuna_trial: Optional[Any] = None,
        explode_threshold: float = 50.0,
        metric_name: str = "a_reg/max_a_epoch",
        only_stage: int = 1,
        objective_metric: str = "val_total_nll_epoch",
    ):
        super().__init__()
        self.optuna_trial = optuna_trial
        self.explode_threshold = explode_threshold
        self.metric_name = metric_name
        self.only_stage = only_stage
        self.objective_metric = objective_metric

    @staticmethod
    def _get_metric(metrics: dict, key: str) -> Optional[float]:
        """Pull a scalar from Lightning's callback_metrics dict, handling tensors."""
        if key not in metrics:
            return None
        val = metrics[key]
        try:
            return float(val.item()) if hasattr(val, "item") else float(val)
        except (TypeError, ValueError):
            return None

    def _resolve_objective(self, metrics: dict) -> Optional[float]:
        for key in (self.objective_metric, "train_total_nll_epoch", "train_loss_epoch"):
            v = self._get_metric(metrics, key)
            if v is not None:
                return v
        return None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Stage gate
        current_stage = getattr(pl_module, "stage", None)
        if current_stage != self.only_stage:
            return

        metrics = trainer.callback_metrics or {}

        # Read a_reg/max_a (post-softplus). If the user disabled regularization
        # (lambda_a_reg=0), this metric won't be logged → nothing to monitor.
        max_a = self._get_metric(metrics, self.metric_name)
        if max_a is None:
            # Try the non-_epoch key as fallback (Lightning version differences)
            max_a = self._get_metric(metrics, self.metric_name.replace("_epoch", ""))

        if max_a is None:
            # Regularization is disabled or metric unavailable — nothing to do.
            return

        # Log a stable metric name for downstream consumers
        pl_module.log("marginal_health/max_a", max_a, on_epoch=True, prog_bar=False)

        # Optuna pruning logic
        if self.optuna_trial is None:
            return

        try:
            import optuna
        except ImportError:
            logger.warning("MarginalHealthMonitor: optuna_trial provided but optuna not importable")
            return

        epoch = trainer.current_epoch
        objective = self._resolve_objective(metrics)
        if objective is not None:
            self.optuna_trial.report(objective, step=epoch)

        # Hard explode-prune: if max_a above threshold, this trial has rediscovered
        # the collapsed minimum. Prune unconditionally (do NOT rely on the sampler
        # to figure this out from val_loss, which rewards collapse).
        if max_a > self.explode_threshold:
            logger.info(
                f"MarginalHealthMonitor: pruning trial at epoch {epoch} — "
                f"max_a={max_a:.2f} > threshold={self.explode_threshold:.2f} "
                f"(flow collapse signature)"
            )
            raise optuna.TrialPruned(
                f"Marginal flow `a` exceeded {self.explode_threshold} "
                f"(observed {max_a:.2f}) — collapsed CDF detected"
            )

        # Soft sampler-driven pruning: defer to the sampler's pruning policy
        if self.optuna_trial.should_prune():
            logger.info(
                f"MarginalHealthMonitor: pruning trial at epoch {epoch} per "
                f"Optuna sampler policy (objective={objective})"
            )
            raise optuna.TrialPruned()
