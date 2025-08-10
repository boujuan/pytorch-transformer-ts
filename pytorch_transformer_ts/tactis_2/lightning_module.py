# Use the newer namespace consistent with Lightning > v2.0
import logging
import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from .module import TACTiS2Model
from typing import Optional

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
        gradient_clip_val_stage1: float = 1000.0,  # Gradient clipping for stage 1
        gradient_clip_val_stage2: float = 1000.0,  # Gradient clipping for stage 2
        stage: int = 1,  # Start with stage 1 (flow-only)
        stage2_start_epoch: int = 10,  # When to start stage 2
        warmup_steps_s1: Optional[int] = 1000, # Number of warmup steps for Stage 1 LR (resolved absolute value)
        warmup_steps_s2: Optional[int] = 500,  # Number of warmup steps for Stage 2 LR (resolved absolute value)
        steps_to_decay_s1: Optional[int] = None,  # T_max value for Stage 1 CosineAnnealingLR (resolved absolute value)
        steps_to_decay_s2: Optional[int] = None,  # T_max value for Stage 2 CosineAnnealingLR (resolved absolute value)
        stage1_activation_function: str = "ReLU", # Added direct parameter
        stage2_activation_function: str = "ReLU", # Added direct parameter
        eta_min_fraction_s1: float = 0.01, # Fraction of initial LR for eta_min in Stage 1 cosine decay
        eta_min_fraction_s2: float = 0.01, # Fraction of initial LR for eta_min in Stage 2 cosine decay
        num_batches_per_epoch: int = None, # Number of batches per epoch for scheduler calculations
        batch_size: int = 2048, # Current trial's batch size
        base_batch_size_for_scheduler_steps: int = 2048, # Base batch size for scheduler step calculations
        base_limit_train_batches: int = None, # Base limit train batches - if set, disables batch size scaling
    ) -> None:
        """
        Initialize the TACTiS2 Lightning Module.
        
        Parameters
        ----------
        model_config
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
        stage1_activation_function
            Activation function for stage 1.
        stage2_activation_function
            Activation function for stage 2.
        """
        super().__init__()
        
        # Save hyperparameters first, so self.hparams is populated
        self.save_hyperparameters("model_config", "lr_stage1", "lr_stage2",
                                   "weight_decay_stage1", "weight_decay_stage2",
                                   "gradient_clip_val_stage1", "gradient_clip_val_stage2",
                                   "stage", "stage2_start_epoch",
                                   "warmup_steps_s1", "warmup_steps_s2",
                                   "steps_to_decay_s1", "steps_to_decay_s2",
                                   "stage1_activation_function", "stage2_activation_function",
                                   "eta_min_fraction_s1", "eta_min_fraction_s2",
                                   "num_batches_per_epoch", "batch_size",
                                   "base_batch_size_for_scheduler_steps", "base_limit_train_batches")

        # Instantiate the model internally using the provided config
        # Separate Attentional Copula parameters from the main model config
        ac_params = {}
        model_direct_params = {}
        # Mapping from config keys (with ac_ prefix) to AttentionalCopula internal arg names
        ac_param_mapping = {
            'ac_mlp_num_layers': 'mlp_layers',
            'ac_mlp_dim': 'mlp_dim',
            'ac_activation_function': 'activation_function'
            # Add other ac_ mappings here if needed in the future
        }

        # Use self.hparams.model_config as it's now saved
        for key, value in self.hparams.model_config.items():
            if key.startswith('ac_') and key in ac_param_mapping:
                # Map and store in ac_params
                mapped_key = ac_param_mapping[key]
                ac_params[mapped_key] = value
                logger.debug(f"Extracted AttentionalCopula parameter: {key} -> {mapped_key}={value}")
            elif key == 'stage1_activation_function' or key == 'stage2_activation_function':
                # These are now direct hparams, so skip them from model_config when passing to TACTiS2Model
                logger.debug(f"Skipping {key} from model_config for model_direct_params, as it's a direct hparam.")
                pass
            else:
                # Keep other non-ac_ parameters in the direct dictionary
                model_direct_params[key] = value

        # Include the stage parameter in the direct params (from hparams)
        model_direct_params['stage'] = self.hparams.stage
        
        # Instantiate the model with separated parameters
        self.model = TACTiS2Model(
            **model_direct_params, # Pass only direct model params here
            stage1_activation_function=self.hparams.stage1_activation_function, # Use direct hparam
            stage2_activation_function=self.hparams.stage2_activation_function, # Use direct hparam
            attentional_copula_kwargs=ac_params if ac_params else None # Pass mapped AC params separately
        )
        
        # Store stage-specific optimizer parameters (already in self.hparams)
        # self.lr_stage1 = lr_stage1
        # self.lr_stage2 = lr_stage2
        # Access hyperparameters via self.hparams
        self.lr_stage1 = self.hparams.lr_stage1
        self.lr_stage2 = self.hparams.lr_stage2
        self.weight_decay_stage1 = self.hparams.weight_decay_stage1
        self.weight_decay_stage2 = self.hparams.weight_decay_stage2
        self.stage = self.hparams.stage
        self.stage2_start_epoch = self.hparams.stage2_start_epoch
        
        # Use resolved absolute values directly (estimator calculated from fractions)
        # The estimator has already calculated absolute steps from fractions based on
        # actual training schedule: stage epochs × num_batches_per_epoch
        # No additional batch size scaling needed since we have actual step counts
        self.warmup_steps_s1 = self.hparams.warmup_steps_s1 or 0
        self.warmup_steps_s2 = self.hparams.warmup_steps_s2 or 0
        self.steps_to_decay_s1 = self.hparams.steps_to_decay_s1
        self.steps_to_decay_s2 = self.hparams.steps_to_decay_s2
        
        logger.info(f"Using resolved scheduler steps from estimator (no additional scaling):")
        logger.info(f"  warmup_s1={self.warmup_steps_s1}, warmup_s2={self.warmup_steps_s2}")
        logger.info(f"  decay_s1={self.steps_to_decay_s1}, decay_s2={self.steps_to_decay_s2}")
        logger.info(f"  (These values were calculated from fractions × actual training steps)")

        # Initialize scheduler reference attributes
        self.warmup_scheduler_ref = None
        self.cosine_scheduler_ref = None
        self.sequential_scheduler_ref = None

        # Set the stage in the model
        if hasattr(self.model.tactis, "set_stage"):
            self.model.tactis.set_stage(self.stage)
            
    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        
        Check if we need to switch to stage 2.
        """
        super().on_train_epoch_start()
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
            frozen_tensors = 0
            unfrozen_tensors = 0
            frozen_elements = 0
            unfrozen_elements = 0
            other_tensors = 0
            other_elements = 0
            
            for name, param in self.model.tactis.named_parameters():
                # FIXED: Include decoder.marginal and decoder.copula patterns
                is_marginal = (name.startswith("flow_") or name.startswith("marginal") or "decoder.marginal" in name)
                is_copula = (name.startswith("copula_") or name.startswith("copula.") or "decoder.copula" in name)
                
                if is_marginal:
                    param.requires_grad = False  # Freeze marginal/flow parameters in stage 2
                    frozen_tensors += 1
                    frozen_elements += param.numel()
                elif is_copula:
                    param.requires_grad = True   # Unfreeze copula parameters in stage 2  
                    unfrozen_tensors += 1
                    unfrozen_elements += param.numel()
                else:
                    # Default: Keep requires_grad as is, but log a warning if it's unexpected
                    logger.debug(f"Parameter '{name}' not explicitly frozen/unfrozen.")
                    other_tensors += 1
                    other_elements += param.numel()

            # Provide clear, accurate logging
            logger.info(f"Stage 2 Parameter Summary:")
            logger.info(f"  Frozen (marginal/flow): {frozen_tensors} tensors, {frozen_elements:,} parameters")
            logger.info(f"  Trainable (copula): {unfrozen_tensors} tensors, {unfrozen_elements:,} parameters")
            if other_tensors > 0:
                logger.info(f"  Other: {other_tensors} tensors, {other_elements:,} parameters")
            logger.info(f"  Total model: {frozen_tensors + unfrozen_tensors + other_tensors} tensors, {frozen_elements + unfrozen_elements + other_elements:,} parameters")

            # 3. Update optimizer settings for the (potentially new) set of trainable parameters
            optimizer = self.optimizers()
            if isinstance(optimizer, list): # Handle cases with LR schedulers
                 optimizer = optimizer[0]

            if optimizer:
                 # Update LR, initial_lr, and weight_decay for all parameter groups
                 # This is crucial for proper scheduler behavior in Stage 2
                 for param_group in optimizer.param_groups:
                     param_group['initial_lr'] = self.lr_stage2  # Critical for LambdaLR reference point
                     param_group['lr'] = self.lr_stage2
                     param_group['weight_decay'] = self.weight_decay_stage2
                 self.log_dict({"stage": 2, "learning_rate": self.lr_stage2, "weight_decay": self.weight_decay_stage2})
                 logger.info(f"Epoch {current_epoch}: Switched to Stage 2. Updated optimizer lr={self.lr_stage2}, weight_decay={self.weight_decay_stage2}. Parameter freezing applied.")
                 
                 # 4. Set up Stage 2 CosineAnnealingLR scheduler
                 # Get steps_per_epoch from hyperparameters
                 steps_per_epoch = self.hparams.num_batches_per_epoch
                 
                 # Check if steps_per_epoch is valid
                 if steps_per_epoch is None or steps_per_epoch <= 0:
                     logger.warning(f"Invalid num_batches_per_epoch: {steps_per_epoch}. Using 100 as default.")
                     steps_per_epoch = 100
                 
                 # Calculate T_max for Stage 2 cosine annealing
                 total_epochs_in_run = self.trainer.max_epochs
                 epochs_in_stage2 = total_epochs_in_run - self.hparams.stage2_start_epoch
                 
                 # Log diagnostic information
                 logger.info(f"Stage 2 scheduler setup - total_epochs: {total_epochs_in_run}, stage2_start_epoch: {self.hparams.stage2_start_epoch}, epochs_in_stage2: {epochs_in_stage2}")
                 
                 # Check if manual steps_to_decay_s2 is configured
                 if self.steps_to_decay_s2 is not None and self.steps_to_decay_s2 > 0:
                     T_max_s2 = self.steps_to_decay_s2
                     logger.info(f"Using steps_to_decay_s2={T_max_s2} as T_max for Stage 2 CosineAnnealingLR.")
                 else:
                     # Calculate based on epochs and steps per epoch
                     T_max_s2_calculated = steps_per_epoch * epochs_in_stage2
                     T_max_s2 = T_max_s2_calculated
                     logger.info(f"steps_to_decay_s2 not configured or invalid. Calculating T_max_s2 = {steps_per_epoch} * {epochs_in_stage2} = {T_max_s2}")
                 
                 # Ensure T_max is valid
                 if T_max_s2 <= 0:
                     logger.warning(f"Stage 2 Cosine Annealing duration (T_max_s2) is not positive ({T_max_s2}). Using constant LR for Stage 2.")
                     
                     # Update the existing warmup scheduler to a constant LR if it exists
                     if self.warmup_scheduler_ref is not None:
                         # Define a constant lambda function
                         constant_lambda = lambda _: 1.0
                         self.warmup_scheduler_ref.lr_lambdas = [constant_lambda]
                         logger.info(f"Updated warmup scheduler to constant LR for Stage 2.")
                         return
                     else:
                         logger.error("No scheduler reference available to update for Stage 2.")
                         return
                 
                 # Calculate eta_min for Stage 2
                 lr_s2 = self.hparams.lr_stage2
                 eta_frac_s2 = self.hparams.eta_min_fraction_s2
                 eta_min_s2 = lr_s2 * eta_frac_s2
                 
                 logger.info(f"Configuring Stage 2 LR scheduler: Linear warmup for {self.warmup_steps_s2} steps.")
                 
                 # Update Warmup Scheduler if it exists
                 if self.warmup_scheduler_ref is not None:
                     # Define new warmup function that starts from 0 and scales to 1.0
                     final_warmup_steps_s2 = max(1, int(self.warmup_steps_s2))
                     def lr_lambda_func_s2(current_step: int):
                         if current_step < final_warmup_steps_s2 - 1:
                             return float(current_step + 1) / float(final_warmup_steps_s2)  # current_step is 0-indexed
                         return 1.0  # Ensure we reach exactly 1.0 at the step before milestone
                     
                     # Update the lambda function and reset scheduler
                     self.warmup_scheduler_ref.lr_lambdas = [lr_lambda_func_s2]
                     self.warmup_scheduler_ref.last_epoch = -1  # Reset for fresh warmup start
                     
                     # Update the base_lrs of the LambdaLR to the new lr_stage2
                     # Fetch the optimizer again as it might be a new list/object after reconfigure
                     current_optimizer = self.optimizers()
                     if isinstance(current_optimizer, list):
                         current_optimizer = current_optimizer[0]
                     
                     if current_optimizer:
                         self.warmup_scheduler_ref.base_lrs = [pg['initial_lr'] for pg in current_optimizer.param_groups]
                         logger.info(f"Updated warmup_scheduler_ref.base_lrs to: {[pg['initial_lr'] for pg in current_optimizer.param_groups]} for Stage 2.")
                     else:
                         logger.error("Failed to retrieve optimizer to update warmup_scheduler_ref.base_lrs for Stage 2.")
                     
                     logger.info(f"Updated warmup scheduler with new warmup steps: {final_warmup_steps_s2} and reset epoch counter")
                 
                 # Adjust T_max for cosine annealing to account for warmup steps
                 # Only adjust if we're using the calculated value, not the manual value
                 if self.steps_to_decay_s2 is None or self.steps_to_decay_s2 <= 0:
                     T_max_s2 = T_max_s2 - self.warmup_steps_s2
                     logger.info(f"Stage 2 Cosine Annealing adjusted for warmup: T_max_s2 = {T_max_s2} (after subtracting {self.warmup_steps_s2} warmup steps)")
                 
                 if T_max_s2 <= 0:
                     logger.warning(f"Adjusted T_max_s2 after warmup is not positive ({T_max_s2}). Using warmup-only scheduler for Stage 2.")
                     # In this case, we'll let the warmup scheduler handle everything
                     # and just set cosine to have minimal effect if it exists
                     if self.cosine_scheduler_ref is not None:
                         self.cosine_scheduler_ref.T_max = 1
                         self.cosine_scheduler_ref.eta_min = lr_s2  # No decay
                         self.cosine_scheduler_ref.last_epoch = -1  # Reset internal counter
                         logger.info(f"Set cosine scheduler to constant LR (T_max=1, eta_min=lr_s2)")
                     
                     # Update sequential scheduler if it exists
                     if self.sequential_scheduler_ref is not None:
                         self.sequential_scheduler_ref.milestones = [self.warmup_steps_s2]
                         self.sequential_scheduler_ref.last_epoch = -1  # Reset internal counter
                         logger.info(f"Updated sequential scheduler milestone to {self.warmup_steps_s2}")
                     return
                 
                 logger.info(f"Configuring CosineAnnealingLR for Stage 2: T_max={T_max_s2}, eta_min={eta_min_s2}")
                 
                 # Update Cosine Scheduler if it exists
                 if self.cosine_scheduler_ref is not None:
                     # Update existing cosine scheduler parameters
                     # Update cosine scheduler parameters for Stage 2
                     self.cosine_scheduler_ref.base_lrs = [self.hparams.lr_stage2]  # Set correct base LR
                     self.cosine_scheduler_ref.T_max = T_max_s2
                     self.cosine_scheduler_ref.eta_min = eta_min_s2
                     self.cosine_scheduler_ref.last_epoch = -1  # Reset internal counter
                     logger.info(f"Updated cosine scheduler with T_max={T_max_s2}, eta_min={eta_min_s2}")
                 
                 # Update Sequential Scheduler if it exists
                 if self.sequential_scheduler_ref is not None:
                     # Update the milestone - add 1 to ensure warmup completes before transition
                     self.sequential_scheduler_ref.milestones = [self.warmup_steps_s2]
                     # Reset internal counter
                     self.sequential_scheduler_ref.last_epoch = -1
                     logger.info(f"Updated sequential scheduler with new milestone: {self.warmup_steps_s2}")
                 else:
                     logger.error("No sequential scheduler reference available to update.")
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

        optimizer = torch.optim.AdamW( # INFO: Changed from Adam to AdamW
            self.parameters(),
            lr=current_lr,
            weight_decay=current_weight_decay,
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
        logger.info(f"Scheduler setup - num_batches_per_epoch: {steps_per_epoch}, stage2_start_epoch: {self.hparams.stage2_start_epoch}, warmup_steps_s1: {self.warmup_steps_s1}")
        
        # For Stage 1, implement warmup + cosine annealing
        if self.stage == 1:
            # Create warmup scheduler
            if self.warmup_steps_s1 > 0:
                logger.info(f"Configuring LR scheduler: Linear warmup for {self.warmup_steps_s1} steps.")
                
                # Define linear warmup function
                def lr_lambda_func(current_step: int):
                    if current_step < self.warmup_steps_s1:
                        return float(current_step) / float(max(1, self.warmup_steps_s1))
                    return 1.0
                
                warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)
                self.warmup_scheduler_ref = warmup_scheduler  # Store reference
                
                # Calculate T_max for cosine annealing in Stage 1
                # Check if manual steps_to_decay_s1 is configured
                if self.steps_to_decay_s1 is not None and self.steps_to_decay_s1 > 0:
                    T_max_s1 = self.steps_to_decay_s1
                    logger.info(f"Using steps_to_decay_s1={T_max_s1} as T_max for Stage 1 CosineAnnealingLR.")
                else:
                    # Calculate based on epochs and steps per epoch
                    epochs_in_stage1 = self.hparams.stage2_start_epoch
                    T_max_s1_calculated = (steps_per_epoch * epochs_in_stage1) - self.warmup_steps_s1
                    T_max_s1 = T_max_s1_calculated
                    logger.info(f"steps_to_decay_s1 not configured or invalid. Calculating T_max_s1 = ({steps_per_epoch} * {epochs_in_stage1}) - {self.warmup_steps_s1} = {T_max_s1}")
                
                # Ensure T_max is valid
                if T_max_s1 <= 0:
                    logger.warning(f"Stage 1 Cosine Annealing duration (T_max_s1) is not positive ({T_max_s1}). Only applying warmup for Stage 1.")
                    # Return only the warmup scheduler if T_max is not positive
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": warmup_scheduler,
                            "interval": "step",  # Step scheduler every training step
                            "frequency": 1,
                            "name": "lr_scheduler_stage1_warmup_only",
                        },
                    }
                
                # Calculate eta_min for Stage 1
                lr_s1 = self.hparams.lr_stage1
                eta_frac_s1 = self.hparams.eta_min_fraction_s1
                eta_min_s1 = lr_s1 * eta_frac_s1
                
                logger.info(f"Configuring Stage 1 CosineAnnealingLR: T_max={T_max_s1}. Inputs: lr_stage1={lr_s1}, eta_min_fraction_s1={eta_frac_s1}. Calculated eta_min={eta_min_s1}")
                
                # Create cosine annealing scheduler for Stage 1
                cosine_scheduler_s1 = CosineAnnealingLR(optimizer, T_max=T_max_s1, eta_min=eta_min_s1)
                self.cosine_scheduler_ref = cosine_scheduler_s1  # Store reference
                
                # Create sequential scheduler that combines warmup and cosine annealing
                sequential_scheduler_s1 = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler_s1],
                    milestones=[self.warmup_steps_s1],
                )
                self.sequential_scheduler_ref = sequential_scheduler_s1  # Store reference
                
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": sequential_scheduler_s1,
                        "interval": "step",  # Step scheduler every training step
                        "frequency": 1,
                        "name": "lr_scheduler_stage1",
                    },
                }
            else:
                # No warmup, just cosine annealing
                # Check if manual steps_to_decay_s1 is configured
                if self.steps_to_decay_s1 is not None and self.steps_to_decay_s1 > 0:
                    T_max_s1 = self.steps_to_decay_s1
                    logger.info(f"Using scaled steps_to_decay_s1={T_max_s1} as T_max for Stage 1 CosineAnnealingLR (no warmup).")
                else:
                    # Calculate based on epochs and steps per epoch
                    epochs_in_stage1 = self.hparams.stage2_start_epoch
                    T_max_s1 = steps_per_epoch * epochs_in_stage1
                    logger.info(f"steps_to_decay_s1 not configured or invalid. Calculating T_max_s1 = {steps_per_epoch} * {epochs_in_stage1} = {T_max_s1}")
                
                # Ensure T_max is valid
                if T_max_s1 <= 0:
                    logger.warning(f"Stage 1 Cosine Annealing duration (T_max_s1) is not positive ({T_max_s1}). Using constant LR for Stage 1.")
                    # Return a constant LR scheduler (identity function)
                    identity_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": identity_scheduler,
                            "interval": "step",
                            "frequency": 1,
                            "name": "lr_scheduler_stage1_constant",
                        },
                    }
                
                # Calculate eta_min for Stage 1
                eta_min_s1 = self.hparams.lr_stage1 * self.hparams.eta_min_fraction_s1
                
                logger.info(f"Configuring Stage 1 CosineAnnealingLR (no warmup): T_max={T_max_s1}, calculated eta_min={eta_min_s1} (lr_stage1={self.hparams.lr_stage1}, eta_min_fraction_s1={self.hparams.eta_min_fraction_s1})")
                
                # Create cosine annealing scheduler for Stage 1
                cosine_scheduler_s1 = CosineAnnealingLR(optimizer, T_max=T_max_s1, eta_min=eta_min_s1)
                self.cosine_scheduler_ref = cosine_scheduler_s1  # Store reference
                
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": cosine_scheduler_s1,
                        "interval": "step",  # Step scheduler every training step
                        "frequency": 1,
                        "name": "lr_scheduler_stage1",
                    },
                }
        else:
            # For Stage 2, the scheduler will be set in on_train_epoch_start
            logger.info(f"Stage 2 scheduler will be configured during on_train_epoch_start.")
            return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        """
        Configure gradient clipping based on the current training stage.
        This method is called by PyTorch Lightning during training.
        """
        # Use stage-specific gradient clipping
        if self.stage == 1:
            clip_val = self.hparams.gradient_clip_val_stage1
        else:
            clip_val = self.hparams.gradient_clip_val_stage2
        
        # Only clip if value is greater than 0
        if clip_val > 0:
            self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm)
    
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
