# Use the newer namespace consistent with Lightning > v2.0
import logging
import lightning.pytorch as pl
import torch
from gluonts.torch.util import weighted_average
# from gluonts.dataset.field_names import FieldName
import sys
from torch.optim.lr_scheduler import LambdaLR # Import LambdaLR
from .module import TACTiS2Model

# Set up logging
logger = logging.getLogger(__name__)

class TACTiS2LightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the TACTiS2 model.
    """
    
    def __init__(
        self,
        model_config: dict, # Accept model configuration dictionary
        lr_stage1: float = 1.8e-3,
        lr_stage2: float = 7.0e-4,
        weight_decay_stage1: float = 0.0,
        weight_decay_stage2: float = 0.0,
        stage: int = 1,  # Start with stage 1 (flow-only)
        stage2_start_epoch: int = 10,  # When to start stage 2
        warmup_steps: int = 1000, # Number of warmup steps for Stage 1 LR
    ) -> None:
        """
        Initialize the TACTiS2 Lightning Module.
        
        Parameters
        ----------
        model
            Dictionary containing the configuration for the TACTiS2Model.
        lr_stage1
            Learning rate for stage 1 optimizer.
        lr_stage2
            Learning rate for stage 2 optimizer.
        weight_decay_stage1
            Weight decay for stage 1 optimizer.
        weight_decay_stage2
            Weight decay for stage 2 optimizer.
        stage
            Initial training stage (1 for flow-only, 2 for flow+copula).
        stage2_start_epoch
            Epoch at which to switch to stage 2 (flow+copula) if starting with stage 1.
        warmup_steps
            Number of linear warmup steps for the learning rate during Stage 1.
        """
        super().__init__()
        # Instantiate the model internally using the provided config
        # Extract Attentional Copula specific parameters with ac_ prefix
        ac_params = {}
        model_direct_params = {}
        
        # Map from config ac_ prefix to the actual parameter names in AttentionalCopula
        ac_param_mapping = {
            'ac_mlp_num_layers': 'mlp_layers',
            'ac_mlp_dim': 'mlp_dim',
            'ac_activation_function': 'activation_function'
        }
        
        # Separate parameters into proper dictionaries
        for key, value in model_config.items():
            if key.startswith('ac_') and key in ac_param_mapping:
                # Store in ac_params with proper mapping
                mapped_key = ac_param_mapping[key]
                ac_params[mapped_key] = value
                logger.info(f"Extracted AttentionalCopula parameter: {key} -> {mapped_key}={value}")
            else:
                # All other parameters go to the direct params dictionary
                model_direct_params[key] = value
        
        # Include the stage parameter in the direct params
        model_direct_params['stage'] = stage
        
        # Pass both direct parameters and AC parameters to TACTiS2Model
        # This ensures AC parameters are properly handled without interfering with other parameters
        self.model = TACTiS2Model(
            **model_direct_params,
            attentional_copula_kwargs=ac_params if ac_params else None
        )
        # Save hyperparameters, including the model_config
        self.save_hyperparameters("model_config", "lr_stage1", "lr_stage2",
                                  "weight_decay_stage1", "weight_decay_stage2",
                                  "stage", "stage2_start_epoch",
                                  "warmup_steps") # Add warmup_steps here
        # Store stage-specific optimizer parameters
        # Note: These are already saved by save_hyperparameters() if passed to __init__
        # self.lr_stage1 = lr_stage1
        # self.lr_stage2 = lr_stage2
        # Access hyperparameters via self.hparams
        self.lr_stage1 = self.hparams.lr_stage1
        self.lr_stage2 = self.hparams.lr_stage2
        self.weight_decay_stage1 = self.hparams.weight_decay_stage1
        self.weight_decay_stage2 = self.hparams.weight_decay_stage2
        self.stage = self.hparams.stage
        self.stage2_start_epoch = self.hparams.stage2_start_epoch
        self.warmup_steps = self.hparams.warmup_steps # Store warmup steps

        # Set the stage in the model
        if hasattr(self.model.tactis, "set_stage"):
            self.model.tactis.set_stage(self.stage)
            
    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        
        Check if we need to switch to stage 2.
        """
        current_epoch = self.current_epoch
        if self.stage == 1 and current_epoch >= self.stage2_start_epoch:
            logger.info(f"Epoch {current_epoch}: Entering Stage 2 transition.")
            self.stage = 2

            # Update the stage in the model - this no longer initializes components
            # but just updates the flags and enables already initialized components
            if hasattr(self.model.tactis, "set_stage"):
                self.model.tactis.set_stage(self.stage)
                logger.info(f"Epoch {current_epoch}: Called model.tactis.set_stage(2)")
            else:
                 logger.warning("model.tactis does not have set_stage method.")
                 # Cannot proceed with freezing/optimizer update if stage cannot be set in model

            # 2. Freeze flow/marginal parameters, unfreeze copula parameters
            logger.info("Freezing flow/marginal parameters and unfreezing copula parameters...")
            frozen_count = 0
            unfrozen_count = 0
            for name, param in self.model.tactis.named_parameters():
                if name.startswith("flow_") or name.startswith("marginal"):
                    param.requires_grad = False
                    frozen_count += 1
                elif name.startswith("copula_") or name.startswith("copula."): # Check for direct attribute 'copula' too
                    param.requires_grad = True
                    unfrozen_count += 1
                else:
                    # Default: Keep requires_grad as is, but log a warning if it's unexpected
                    logger.debug(f"Parameter '{name}' not explicitly frozen/unfrozen.")

            logger.info(f"Froze {frozen_count} flow/marginal parameters. Ensured {unfrozen_count} copula parameters are trainable.")

            # 3. Update optimizer settings for the (potentially new) set of trainable parameters
            optimizer = self.optimizers()
            if isinstance(optimizer, list): # Handle cases with LR schedulers
                 optimizer = optimizer[0]

            if optimizer:
                 # Update LR and WD for all parameter groups.
                 # The requires_grad=False flags will prevent updates to frozen parameters.
                 for param_group in optimizer.param_groups:
                     param_group['lr'] = self.lr_stage2
                     param_group['weight_decay'] = self.weight_decay_stage2
                 self.log_dict({"stage": 2, "learning_rate": self.lr_stage2, "weight_decay": self.weight_decay_stage2})
                 logger.info(f"Epoch {current_epoch}: Switched to Stage 2. Updated optimizer lr={self.lr_stage2}, weight_decay={self.weight_decay_stage2}. Parameter freezing applied.")
            else:
                 logger.warning(f"Epoch {current_epoch}: Tried to switch to Stage 2, but optimizer not found. Cannot update LR/WD.")

            self.log("stage", self.stage, on_step=False, on_epoch=True)
                
    def training_step(self, batch, batch_idx: int):
        """
        Training step.
        
        Parameters
        ----------
        batch
            The input batch.
        batch_idx
            The index of the batch.
            
        Returns
        -------
        The loss.
        """
        # Automatic optimization handles optimizer steps, zero_grad, backward
        
        # Extract data from the batch
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        future_target = batch["future_target"]
        
        # Get time features
        past_time_feat = batch["past_time_feat"]
        future_time_feat = batch["future_time_feat"]
        
        # Get static features if available
        feat_static_cat = batch.get("feat_static_cat", torch.zeros((past_target.shape[0], 1), device=self.device, dtype=torch.long))
        feat_static_real = batch.get("feat_static_real", torch.zeros((past_target.shape[0], 1), device=self.device, dtype=torch.float32))
        
        # Get dynamic features if available
        feat_dynamic_real = batch.get("feat_dynamic_real", None)
        
        # Process with model
        model_output = self.model(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,  # For teacher forcing or loss computation
        )
        
        # Check if model returns a tuple (common in TACTiS-2 where the model returns both predictions and loss)
        if isinstance(model_output, tuple):
            # The second element of the tuple is typically the loss
            predictions, loss = model_output
            logger.debug(f"Training - Model returned tuple: predictions shape={predictions.shape if hasattr(predictions, 'shape') else 'N/A'}, loss={loss}")
        else:
            # If it's not a tuple, assume it's just the loss
            loss = model_output
            logger.debug(f"Training - Model returned scalar loss: {loss}")
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            logger.warning("NaN detected in loss! Replacing with large value to continue training.")
            loss = torch.nan_to_num(loss, nan=1000.0)  # Use a large value but not too large
        
        # Loss is returned for automatic optimization

        # Log the loss
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        
        # Manual LR logging removed as LearningRateMonitor is now working
        
        return loss
        
    def validation_step(self, batch, batch_idx: int):
        """
        Validation step.
        
        Parameters
        ----------
        batch
            The input batch.
        batch_idx
            The index of the batch.
            
        Returns
        -------
        The loss.
        """
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        future_target = batch["future_target"]
        
        # Get time features
        past_time_feat = batch["past_time_feat"]
        future_time_feat = batch["future_time_feat"]
        
        # Get static features if available
        feat_static_cat = batch.get("feat_static_cat", torch.zeros((past_target.shape[0], 1), device=self.device, dtype=torch.long))
        feat_static_real = batch.get("feat_static_real", torch.zeros((past_target.shape[0], 1), device=self.device, dtype=torch.float32))
        
        # Get dynamic features if available
        feat_dynamic_real = batch.get("feat_dynamic_real", None)
        
        # Process with model - no gradients needed for validation
        with torch.no_grad():
            model_output = self.model(
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_time_feat=past_time_feat,
                past_target=past_target,
                past_observed_values=past_observed_values,
                future_time_feat=future_time_feat,
                future_target=future_target,  # For teacher forcing or loss computation
            )
        
        # Check if model returns a tuple (common in TACTiS-2 where the model returns both predictions and loss)
        if isinstance(model_output, tuple):
            # The second element of the tuple is typically the loss
            predictions, loss = model_output
            logger.debug(f"Validation - Model returned tuple: predictions shape={predictions.shape if hasattr(predictions, 'shape') else 'N/A'}, loss={loss}")
        else:
            # If it's not a tuple, assume it's just the loss
            loss = model_output
            logger.debug(f"Validation - Model returned scalar loss: {loss}")
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            logger.warning("NaN detected in validation loss! Replacing with large value for logging.")
            loss = torch.nan_to_num(loss, nan=1000.0)  # Use a large value but not too large
        
        # Log the validation loss
        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        Returns
        -------
        The optimizer to use.
        """
        # Initialize optimizer with parameters for the current stage
        current_lr = self.lr_stage1 if self.stage == 1 else self.lr_stage2
        current_weight_decay = self.weight_decay_stage1 if self.stage == 1 else self.weight_decay_stage2

        logger.info(f"Configuring optimizer for Stage {self.stage} with lr={current_lr}, weight_decay={current_weight_decay}")

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=current_lr,
            weight_decay=current_weight_decay,
        )
        
        # Configure gradient clipping directly in the lightning module
        # This helps with numerical stability by preventing extreme gradient values
        # self.automatic_optimization = False # Use automatic optimization loop
        self.automatic_optimization = True # Explicitly set to True
        
        # Add LR Warmup Scheduler ONLY for Stage 1 and if warmup_steps > 0
        if self.stage == 1 and self.warmup_steps > 0:
            logger.info(f"Configuring LR scheduler: Linear warmup for {self.warmup_steps} steps.")
            
            # Define linear warmup function
            def lr_lambda_func(current_step: int):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                return 1.0
                
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Step scheduler every training step
                    "frequency": 1,
                },
            }
        else:
            # No scheduler for Stage 2 or if warmup_steps is 0
            logger.info(f"No LR scheduler configured for Stage {self.stage} (warmup_steps={self.warmup_steps}).")
            return optimizer
            
        # Example for ReduceLROnPlateau (keep commented out)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=10
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",
        #     },
        # }
        
        return optimizer
    
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        **kwargs, # Allow for extra arguments if any
    ):
        """
        Forward pass through the model for inference (prediction).
        Accepts keyword arguments directly as passed by the GluonTS predictor.
        
        Parameters
        ----------
        feat_static_cat
            Static categorical features.
        feat_static_real
            Static real features.
        past_time_feat
            Past time features.
        past_target
            Past target values.
        past_observed_values
            Indicator for observed values in the past.
        future_time_feat
            Future time features.
            
        Returns
        -------
        The output of the model (predictions).
        """
        if feat_static_cat is None:
             feat_static_cat = torch.zeros((past_target.shape[0], 1), device=self.device, dtype=torch.long)
        if feat_static_real is None:
             feat_static_real = torch.zeros((past_target.shape[0], 1), device=self.device, dtype=torch.float32)

        # Call model's forward method, which expects these arguments
        # Pass future_target=None explicitly for inference mode
        model_output = self.model(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=None, # Ensure future_target is None for inference
        )
        
        # Handle tuple return value from the model during inference
        # TACTiS2Model forward should return samples directly in inference mode
        if isinstance(model_output, tuple):
             # This case might indicate an issue if it happens during inference,
             # as the model should ideally return only samples. Log a warning.
             logger.warning(f"Inference - Model returned a tuple unexpectedly. Using the first element as predictions.")
             predictions = model_output[0]
             return predictions
        else:
             # Assume the direct output is the predictions/samples
             logger.debug(f"Inference - Model returned single output (predictions/samples)")
             return model_output
