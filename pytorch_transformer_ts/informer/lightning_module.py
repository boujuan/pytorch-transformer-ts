# Use the newer namespace consistent with Lightning > v2.0
import lightning.pytorch as pl
import torch
# from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from typing import Optional
# CHANGE
from pytorch_transformer_ts.informer.module import InformerModel

import logging
logger = logging.getLogger(__name__)

class InformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        # loss: DistributionLoss = NegativeLogLikelihood(), CHANGE
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        gradient_clip_val: float = 1000.0,  # Gradient clipping
        steps_to_decay: Optional[int] = None,  # Optional manual T_max value for CosineAnnealingLR
        warmup_steps: int = 1000, # Number of warmup steps for LR
        eta_min_fraction: float = 0.01, # Fraction of initial LR for eta_min in cosine decay
        num_batches_per_epoch: int = None, # Number of batches per epoch for scheduler calculations
        batch_size: int = 2048, # Current trial's batch size
        base_batch_size_for_scheduler_steps: int = 2048, # Base batch size for scheduler step calculations
        base_limit_train_batches: int = None, # Base limit train batches - if set, disables batch size scaling
    ) -> None:
        super().__init__()
        
        # if isinstance(model_config, dict):
        self.model = InformerModel(**model_config)
        self.save_hyperparameters("model_config", "lr", "weight_decay", "gradient_clip_val",
                                  "steps_to_decay", "warmup_steps", "eta_min_fraction", 
                                  "num_batches_per_epoch", "batch_size", 
                                  "base_batch_size_for_scheduler_steps", "base_limit_train_batches")
        # else:
        #     self.model = model
        #     self.save_hyperparameters(ignore=["model"])
        # self.loss = loss CHANGE
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Check if dynamic limit_train_batches scaling is enabled
        if self.hparams.base_limit_train_batches is not None:
            # When base_limit_train_batches is set, scale scheduler steps based on
            # the dynamic limit_train_batches relative to the base configuration
            
            # Calculate the steps per epoch scaling factor
            # This accounts for the dynamic limit_train_batches calculation:
            # limit_train_batches = base_limit_train_batches * base_batch_size / current_batch_size
            steps_per_epoch_scaling_factor = self.hparams.base_batch_size_for_scheduler_steps / self.hparams.batch_size
            
            # Scale scheduler step parameters to maintain proportional relationships
            self.scaled_warmup_steps = round(self.hparams.warmup_steps * steps_per_epoch_scaling_factor)
            self.scaled_steps_to_decay = round(self.hparams.steps_to_decay * steps_per_epoch_scaling_factor) if self.hparams.steps_to_decay is not None else None
            
            logger.info(f"Dynamic limit_train_batches scaling ENABLED: base_limit_train_batches={self.hparams.base_limit_train_batches}")
            logger.info(f"Steps per epoch scaling: base_batch_size={self.hparams.base_batch_size_for_scheduler_steps}, "
                       f"current_batch_size={self.hparams.batch_size}, scaling_factor={steps_per_epoch_scaling_factor}")
            logger.info(f"This maintains proportional scheduler timing across different batch sizes with dynamic limit_train_batches")
            logger.info(f"Original scheduler steps: warmup={self.hparams.warmup_steps}, "
                       f"decay={self.hparams.steps_to_decay}, ")
            logger.info(f"Scaled scheduler steps: warmup={self.scaled_warmup_steps}, "
                       f"decay={self.scaled_steps_to_decay}, ")
        else:
            # Legacy batch size scaling for cases without dynamic limit_train_batches
            scaling_factor = 1.0
            if self.hparams.batch_size > 0:  # Avoid division by zero
                scaling_factor = self.hparams.base_batch_size_for_scheduler_steps / self.hparams.batch_size
            else:
                logger.warning("Batch size is zero or negative. Using default scaling factor of 1.0.")
                
            # Scale scheduler step parameters
            self.scaled_warmup_steps = round(self.hparams.warmup_steps * scaling_factor)
            self.scaled_steps_to_decay = round(self.hparams.steps_to_decay * scaling_factor) if self.hparams.steps_to_decay is not None else None
            
            # Log the scaled values
            logger.info(f"Legacy batch size scaling ENABLED: base_batch_size={self.hparams.base_batch_size_for_scheduler_steps}, "
                       f"current_batch_size={self.hparams.batch_size}, scaling_factor={scaling_factor}")
            logger.info(f"Original scheduler steps: warmup={self.hparams.warmup_steps}, "
                       f"decay={self.hparams.steps_to_decay}, ")
            logger.info(f"Scaled scheduler steps: warmup={self.scaled_warmup_steps}, "
                       f"decay={self.scaled_steps_to_decay}, ")

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
            sync_dist=True # CHANGE
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        with torch.inference_mode():
            val_loss = self(batch)
        self.log(
            "val_loss", 
            val_loss,
            on_epoch=True,
            on_step=True, # CHANGE 
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
        if self.scaled_warmup_steps > 0:
            logger.info(f"Configuring LR scheduler: Linear warmup for {self.scaled_warmup_steps} steps.")
            
            # Define linear warmup function
            def lr_lambda_func(current_step: int):
                if current_step < self.scaled_warmup_steps:
                    return float(current_step) / float(max(1, self.scaled_warmup_steps))
                return 1.0
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)
            self.warmup_scheduler_ref = warmup_scheduler  # Store reference
            
            # Calculate T_max for cosine annealing
            # Check if manual steps_to_decay is configured
            if self.scaled_steps_to_decay is not None and self.scaled_steps_to_decay > 0:
                T_max = self.scaled_steps_to_decay
                logger.info(f"Using scaled steps_to_decay={T_max} as T_max for CosineAnnealingLR.")
            else:
                # Calculate based on epochs and steps per epoch
                total_epochs_in_run = self.trainer.max_epochs
                T_max_calculated = (steps_per_epoch * total_epochs_in_run) - self.scaled_warmup_steps
                T_max = T_max_calculated
                logger.info(f"steps_to_decay not configured or invalid. Calculating T_max = ({steps_per_epoch} * {total_epochs_in_run}) - {self.scaled_warmup_steps} = {T_max}")
            
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
                milestones=[self.scaled_warmup_steps],
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
            if self.scaled_steps_to_decay is not None and self.scaled_steps_to_decay > 0:
                T_max = self.scaled_steps_to_decay
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
    
      
    # for training
    def forward(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]
        
        # past_time_feat = torch.broadcast_to(torch.linspace(0, past_time_feat.shape[-1], past_time_feat.shape[-1]), past_time_feat.shape)
        # past_target = torch.broadcast_to(torch.linspace(0, past_target.shape[-1], past_target.shape[-1]), past_target.shape)
        # future_time_feat = torch.broadcast_to(torch.linspace(0, future_time_feat.shape[-1], future_time_feat.shape[-1]), future_time_feat.shape)
        # future_target = torch.broadcast_to(torch.linspace(0, future_target.shape[-1], future_target.shape[-1]), future_target.shape)
        # past_time_feat = torch.broadcast_to(torch.linspace(0, past_time_feat.shape[-1], past_time_feat.shape[-1]), past_time_feat.shape)
         
        transformer_inputs, loc, scale, _ = self.model.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        params = self.model.output_params(transformer_inputs)
        # distr = self.model.output_distribution(params, loc=loc, scale=scale)
        
        # loss_values = self.loss(distr, future_target) CHANGE
        loss_values = self.model.output_loss(params, future_target, loc=loc, scale=scale)

        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)
