# Use the newer namespace consistent with Lightning > v2.0
import lightning.pytorch as pl
import torch
# from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from typing import Optional
from pytorch_transformer_ts.autoformer.module import AutoformerModel

import logging
logger = logging.getLogger(__name__)

class AutoformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        # loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        gradient_clip_val: float = 1000.0,  # Gradient clipping
        steps_to_decay: Optional[float] = 0.9,  # Optional manual T_max value for CosineAnnealingLR
        warmup_steps: float = 0.1, # Number of warmup steps for LR
        eta_min_fraction: float = 0.01, # Fraction of initial LR for eta_min in cosine decay
        num_batches_per_epoch: int = None, # Number of batches per epoch for scheduler calculations
        batch_size: int = 2048, # Current trial's batch size
        base_batch_size_for_scheduler_steps: int = 2048, # Base batch size for scheduler step calculations
        base_limit_train_batches: int = None, # Base limit train batches - if set, disables batch size scaling
    ) -> None:
        super().__init__()
        # if isinstance(model_config, dict):
        self.model = AutoformerModel(**model_config)
        self.save_hyperparameters("model_config", "lr", "weight_decay", "gradient_clip_val",
                                  "steps_to_decay", "warmup_steps", "eta_min_fraction", 
                                  "num_batches_per_epoch", "batch_size", 
                                  "base_batch_size_for_scheduler_steps", "base_limit_train_batches")
        # else:
        #     self.model = model
        #     self.save_hyperparameters(ignore=["model"])
        # self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Use resolved absolute values directly (estimator calculated from fractions)
        # The estimator has already calculated absolute steps from fractions based on
        # actual training schedule: stage epochs × num_batches_per_epoch
        # No additional batch size scaling needed since we have actual step counts
        self.warmup_steps = self.hparams.warmup_steps or 0
        self.steps_to_decay = self.hparams.steps_to_decay
        
        # Initialize scheduler reference attributes
        self.warmup_scheduler_ref = None
        self.cosine_scheduler_ref = None
        self.sequential_scheduler_ref = None

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        train_loss = self(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        with torch.inference_mode():
            val_loss = self(batch)
        self.log("val_loss", 
                 val_loss, 
                 on_epoch=True, 
                 on_step=True, 
                 prog_bar=True,
                 sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Configure gradient clipping directly in the lightning module
        # This helps with numerical stability by preventing extreme gradient values
        self.automatic_optimization = True # Explicitly set to True
        
        # Get steps_per_epoch from hyperparameters
        steps_per_epoch = self.hparams.num_batches_per_epoch
        
        # Check if steps_per_epoch is valid
        if steps_per_epoch is None or steps_per_epoch <= 0:
            logger.warning(f"Invalid num_batches_per_epoch: {steps_per_epoch}. Using 100 as default.")
            steps_per_epoch = 100
            
        # Log diagnostic information
        logger.info(f"Scheduler setup - num_batches_per_epoch: {steps_per_epoch}, warmup_steps: {self.hparams.warmup_steps}")
        
        # Create warmup scheduler
        if self.warmup_steps > 0:
            logger.info(f"Configuring LR scheduler: Linear warmup for {self.warmup_steps} steps.")
            
            # Define linear warmup function
            def lr_lambda_func(current_step: int):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                return 1.0
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)
            self.warmup_scheduler_ref = warmup_scheduler  # Store reference
            
            # Calculate T_max for cosine annealing
            # Check if manual steps_to_decay is configured
            if self.steps_to_decay is not None and self.steps_to_decay > 0:
                T_max = self.steps_to_decay
                logger.info(f"Using scaled steps_to_decay={T_max} as T_max for CosineAnnealingLR.")
            else:
                # Calculate based on epochs and steps per epoch
                total_epochs_in_run = self.trainer.max_epochs
                T_max_calculated = (steps_per_epoch * total_epochs_in_run) - self.warmup_steps
                T_max = T_max_calculated
                logger.info(f"steps_to_decay not configured or invalid. Calculating T_max = ({steps_per_epoch} * {total_epochs_in_run}) - {self.warmup_steps} = {T_max}")
            
            # Ensure T_max is valid
            if T_max <= 0:
                logger.warning(f"Cosine Annealing duration (T_max) is not positive ({T_max}). Only applying warmup scheduler.")
                # Return only the warmup scheduler if T_max is not positive
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": warmup_scheduler,
                        "interval": "step",  # Step scheduler every training step
                        "frequency": 1,
                        "name": "lr_scheduler_warmup_only",
                    },
                }
            
            # Calculate eta_min for
            lr = self.hparams.lr
            eta_frac = self.hparams.eta_min_fraction
            eta_min = lr * eta_frac
            
            logger.info(f"Configuring CosineAnnealingLR: T_max={T_max}. Inputs: lr={lr}, eta_min_fraction={eta_frac}. Calculated eta_min={eta_min}")
            
            # Create cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            self.cosine_scheduler_ref = cosine_scheduler  # Store reference
            
            # Create sequential scheduler that combines warmup and cosine annealing
            sequential_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
            self.sequential_scheduler_ref = sequential_scheduler  # Store reference
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sequential_scheduler,
                    "interval": "step",  # Step scheduler every training step
                    "frequency": 1,
                    "name": "lr_scheduler",
                },
            }
        else:
            # No warmup, just cosine annealing
            # Check if manual steps_to_decay is configured
            if self.steps_to_decay is not None and self.steps_to_decay > 0:
                T_max = self.steps_to_decay
                logger.info(f"Using scaled steps_to_decay={T_max} as T_max for CosineAnnealingLR (no warmup).")
            else:
                # Calculate based on epochs and steps per epoch
                total_epochs_in_run = self.trainer.max_epochs
                T_max = steps_per_epoch * total_epochs_in_run
                logger.info(f"steps_to_decay not configured or invalid. Calculating T_max = {steps_per_epoch} * {total_epochs_in_run} = {T_max}")
            
            # Ensure T_max is valid
            if T_max <= 0:
                logger.warning(f"Cosine Annealing duration (T_max) is not positive ({T_max}). Using constant LR for.")
                # Return a constant LR scheduler (identity function)
                identity_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": identity_scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "name": "lr_scheduler_constant",
                    },
                }
            
            # Calculate eta_min
            eta_min = self.hparams.lr * self.hparams.eta_min_fraction
            
            logger.info(f"Configuring CosineAnnealingLR (no warmup): T_max={T_max}, calculated eta_min={eta_min} (lr={self.hparams.lr}, eta_min_fraction={self.hparams.eta_min_fraction})")
            
            # Create cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            self.cosine_scheduler_ref = cosine_scheduler  # Store reference
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": cosine_scheduler,
                    "interval": "step",  # Step scheduler every training step
                    "frequency": 1,
                    "name": "lr_scheduler",
                },
            }
            
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        """
        Configure gradient clipping.
        This method is called by PyTorch Lightning during training.
        """
        # Use stage-specific gradient clipping
        clip_val = self.hparams.gradient_clip_val
        
        # Only clip if value is greater than 0
        if clip_val > 0:
            self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm)
    
      
    def forward(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        (
            autoformer_inputs,
            loc,
            scale,
            dynamic_features,
            _,
        ) = self.model.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        params = self.model.output_params(autoformer_inputs, dynamic_features)
        loss_values = self.model.output_loss(params, future_target, loc=loc, scale=scale) # TODO HIGH is this averaged across batches or summed?

        # loss_values = self.loss(distr, future_target)

        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)
