# Use the newer namespace consistent with Lightning > v2.0
import logging
import lightning.pytorch as pl
import torch
from gluonts.torch.util import weighted_average
# from gluonts.dataset.field_names import FieldName
import sys
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
        stage: int = 1,  # Start with stage 1 (flow-only)
        stage2_start_epoch: int = 10,  # When to start stage 2
        warmup_steps_s1: int = 1000, # Number of warmup steps for Stage 1 LR
        warmup_steps_s2: int = 500,  # Number of warmup steps for Stage 2 LR
        steps_to_decay_s1: Optional[int] = None,  # Optional manual T_max value for Stage 1 CosineAnnealingLR
        steps_to_decay_s2: Optional[int] = None,  # Optional manual T_max value for Stage 2 CosineAnnealingLR
        stage1_activation_function: str = "ReLU", # Added direct parameter
        stage2_activation_function: str = "ReLU", # Added direct parameter
        eta_min_fraction_s1: float = 0.01, # Fraction of initial LR for eta_min in Stage 1 cosine decay
        eta_min_fraction_s2: float = 0.01, # Fraction of initial LR for eta_min in Stage 2 cosine decay
        num_batches_per_epoch: int = None, # Number of batches per epoch for scheduler calculations
        batch_size: int = 2048, # Current trial's batch size
        base_batch_size_for_scheduler_steps: int = 2048, # Base batch size for scheduler step calculations
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
                                   "stage", "stage2_start_epoch",
                                   "warmup_steps_s1", "warmup_steps_s2",
                                   "steps_to_decay_s1", "steps_to_decay_s2",
                                   "stage1_activation_function", "stage2_activation_function",
                                   "eta_min_fraction_s1", "eta_min_fraction_s2",
                                   "num_batches_per_epoch", "batch_size",
                                   "base_batch_size_for_scheduler_steps")

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
        
        # Ensure 'use_gradient_checkpointing_copula' is also passed if present in model_config
        if 'use_gradient_checkpointing_copula' in self.hparams.model_config:
            model_direct_params['use_gradient_checkpointing_copula'] = self.hparams.model_config['use_gradient_checkpointing_copula']


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
        
        # Calculate scaling factor for scheduler steps based on batch size
        scaling_factor = 1.0
        if self.hparams.batch_size > 0:  # Avoid division by zero
            scaling_factor = self.hparams.base_batch_size_for_scheduler_steps / self.hparams.batch_size
        else:
            logger.warning("Batch size is zero or negative. Using default scaling factor of 1.0.")
            
        # Scale scheduler step parameters
        self.scaled_warmup_steps_s1 = round(self.hparams.warmup_steps_s1 * scaling_factor)
        self.scaled_warmup_steps_s2 = round(self.hparams.warmup_steps_s2 * scaling_factor)
        self.scaled_steps_to_decay_s1 = round(self.hparams.steps_to_decay_s1 * scaling_factor) if self.hparams.steps_to_decay_s1 is not None else None
        self.scaled_steps_to_decay_s2 = round(self.hparams.steps_to_decay_s2 * scaling_factor) if self.hparams.steps_to_decay_s2 is not None else None
        
        # Log the scaled values
        logger.info(f"Batch size scaling: base_batch_size={self.hparams.base_batch_size_for_scheduler_steps}, "
                   f"current_batch_size={self.hparams.batch_size}, scaling_factor={scaling_factor}")
        logger.info(f"Original scheduler steps: warmup_s1={self.hparams.warmup_steps_s1}, "
                   f"warmup_s2={self.hparams.warmup_steps_s2}, "
                   f"decay_s1={self.hparams.steps_to_decay_s1}, "
                   f"decay_s2={self.hparams.steps_to_decay_s2}")
        logger.info(f"Scaled scheduler steps: warmup_s1={self.scaled_warmup_steps_s1}, "
                   f"warmup_s2={self.scaled_warmup_steps_s2}, "
                   f"decay_s1={self.scaled_steps_to_decay_s1}, "
                   f"decay_s2={self.scaled_steps_to_decay_s2}")

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

        if self.stage == 1 and current_epoch >= self.hparams.stage2_start_epoch:
            logger.info(f"Epoch {current_epoch}: Entering Stage 2 transition.")
            self.stage = 2

            if hasattr(self.model.tactis, "set_stage"):
                self.model.tactis.set_stage(self.stage) # This initializes copula components if needed
                logger.info(f"Epoch {current_epoch}: Called model.tactis.set_stage(2)")
            else:
                logger.warning("model.tactis does not have set_stage method. Cannot transition stage in model.")
                return

            logger.info("Freezing flow/marginal parameters and unfreezing copula parameters...")
            frozen_count = 0
            unfrozen_count = 0
            # Ensure all parameters are iterated, including those potentially added in stage 2 by model.tactis.set_stage()
            for name, param in self.model.named_parameters(): # Iterate over self.model to catch all params
                if name.startswith("tactis.flow_") or name.startswith("tactis.marginal") or \
                   name.startswith("model.tactis.flow_") or name.startswith("model.tactis.marginal"): # More specific paths
                    param.requires_grad = False
                    frozen_count += 1
                elif name.startswith("tactis.copula_") or name.startswith("tactis.decoder.copula") or \
                     name.startswith("model.tactis.copula_") or name.startswith("model.tactis.decoder.copula"): # More specific paths
                    param.requires_grad = True
                    unfrozen_count += 1
            logger.info(f"Froze {frozen_count} flow/marginal parameters. Ensured {unfrozen_count} copula parameters are trainable.")

            # Reconfigure optimizer and schedulers for Stage 2
            # This will create a new optimizer with only trainable (copula) parameters
            # and new schedulers configured for Stage 2.
            if self.trainer is not None:
                logger.info("Reconfiguring optimizer and LR schedulers for Stage 2.")
                # PyTorch Lightning's way to re-initialize optimizers/schedulers:
                # 1. Update self.trainer.optimizers
                # 2. Update self.trainer.lr_schedulers
                # This requires configure_optimizers to be robust to being called mid-training for stage 2.
                
                # Force PTL to re-evaluate configure_optimizers by clearing old ones.
                # This is a bit of a heavy-handed way, but ensures clean re-initialization.
                # A more direct PTL API for this would be ideal if available.
                self.trainer.strategy.optimizers = [] # Clear internal optimizer reference in strategy
                
                # PTL will call configure_optimizers() again in the next optimization step
                # or we can try to force it more directly.
                # Forcing re-configuration by directly updating trainer's optimizer and scheduler list:
                new_optimizer_config = self.configure_optimizers() # This will now use self.stage = 2
                
                if "optimizer" not in new_optimizer_config:
                    logger.error("Failed to get optimizer from configure_optimizers for Stage 2.")
                    return

                new_optimizer = new_optimizer_config["optimizer"]
                new_lr_scheduler_config = new_optimizer_config.get("lr_scheduler")

                self.trainer.optimizers = [new_optimizer]
                
                # The lr_scheduler_configs must be in the format PTL expects
                if new_lr_scheduler_config:
                    # Ensure it's a list of PTL scheduler config dictionaries
                    if isinstance(new_lr_scheduler_config, dict) and "scheduler" in new_lr_scheduler_config:
                         self.trainer.lr_schedulers = [new_lr_scheduler_config]
                    elif isinstance(new_lr_scheduler_config, list): # If configure_optimizers returns a list
                         self.trainer.lr_schedulers = new_lr_scheduler_config
                    else:
                         logger.warning(f"LR scheduler config for Stage 2 has unexpected format: {new_lr_scheduler_config}")
                         self.trainer.lr_schedulers = []
                else:
                    self.trainer.lr_schedulers = []
                
                logger.info(f"Epoch {current_epoch}: Optimizer and LR schedulers reconfigured for Stage 2.")
                self.log_dict({
                    "stage": float(self.stage), # Ensure stage is float for logging
                    "learning_rate": self.hparams.lr_stage2, # Log the target LR for stage 2
                    "weight_decay": self.hparams.weight_decay_stage2
                }, on_step=False, on_epoch=True)

            else:
                logger.warning("Trainer not available in on_train_epoch_start. Cannot reconfigure optimizer for Stage 2.")

            self.log("stage", float(self.stage), on_step=False, on_epoch=True, prog_bar=True)
                
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
        This method is now responsible for configuring for Stage 1 or Stage 2
        based on the current self.stage.
        """
        steps_per_epoch = self.hparams.num_batches_per_epoch
        if steps_per_epoch is None or steps_per_epoch <= 0:
            logger.warning(f"Invalid num_batches_per_epoch: {steps_per_epoch} in hparams. Using 100 as default for scheduler calculations.")
            steps_per_epoch = 100

        if self.stage == 1:
            current_lr = self.hparams.lr_stage1
            current_weight_decay = self.hparams.weight_decay_stage1
            params_to_optimize = self.parameters() # All parameters for Stage 1
            logger.info(f"Configuring optimizer and scheduler for Stage 1: lr={current_lr}, wd={current_weight_decay}")

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=current_lr,
                weight_decay=current_weight_decay,
            )

            # Stage 1 Scheduler Logic (Warmup + Cosine Annealing)
            if self.scaled_warmup_steps_s1 > 0:
                logger.info(f"Stage 1: Linear warmup for {self.scaled_warmup_steps_s1} steps.")
                def lr_lambda_s1(current_step: int):
                    if current_step < self.scaled_warmup_steps_s1:
                        return float(current_step + 1) / float(max(1, self.scaled_warmup_steps_s1))
                    return 1.0
                warmup_scheduler_s1 = LambdaLR(optimizer, lr_lambda=lr_lambda_s1)

                T_max_s1_calc = (steps_per_epoch * self.hparams.stage2_start_epoch) - self.scaled_warmup_steps_s1
                T_max_s1 = self.scaled_steps_to_decay_s1 if self.scaled_steps_to_decay_s1 is not None and self.scaled_steps_to_decay_s1 > 0 else T_max_s1_calc
                
                if T_max_s1 <= 0:
                    logger.warning(f"Stage 1 Cosine Annealing duration (T_max_s1={T_max_s1}) is not positive. Only applying warmup.")
                    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": warmup_scheduler_s1, "interval": "step", "frequency": 1, "name": "lr_scheduler_stage1_warmup_only"}}

                eta_min_s1 = self.hparams.lr_stage1 * self.hparams.eta_min_fraction_s1
                logger.info(f"Stage 1: CosineAnnealingLR with T_max={T_max_s1}, eta_min={eta_min_s1}")
                cosine_scheduler_s1 = CosineAnnealingLR(optimizer, T_max=T_max_s1, eta_min=eta_min_s1)
                
                sequential_scheduler_s1 = SequentialLR(optimizer, schedulers=[warmup_scheduler_s1, cosine_scheduler_s1], milestones=[self.scaled_warmup_steps_s1])
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sequential_scheduler_s1, "interval": "step", "frequency": 1, "name": "lr_scheduler_stage1_sequential"}}
            else: # No warmup for Stage 1
                T_max_s1_calc = steps_per_epoch * self.hparams.stage2_start_epoch
                T_max_s1 = self.scaled_steps_to_decay_s1 if self.scaled_steps_to_decay_s1 is not None and self.scaled_steps_to_decay_s1 > 0 else T_max_s1_calc
                if T_max_s1 <= 0:
                    logger.warning(f"Stage 1 Cosine Annealing duration (T_max_s1={T_max_s1}) is not positive (no warmup). Using constant LR.")
                    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": LambdaLR(optimizer, lr_lambda=lambda _: 1.0), "interval": "step", "frequency": 1, "name": "lr_scheduler_stage1_constant"}}

                eta_min_s1 = self.hparams.lr_stage1 * self.hparams.eta_min_fraction_s1
                logger.info(f"Stage 1: CosineAnnealingLR (no warmup) with T_max={T_max_s1}, eta_min={eta_min_s1}")
                cosine_scheduler_s1 = CosineAnnealingLR(optimizer, T_max=T_max_s1, eta_min=eta_min_s1)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": cosine_scheduler_s1, "interval": "step", "frequency": 1, "name": "lr_scheduler_stage1_cosine_only"}}

        elif self.stage == 2:
            current_lr = self.hparams.lr_stage2
            current_weight_decay = self.hparams.weight_decay_stage2
            # Key change: Optimize only parameters that require gradients
            params_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters()) # Use self.model.parameters()
            logger.info(f"Configuring optimizer and scheduler for Stage 2: lr={current_lr}, wd={current_weight_decay} for TRAINABLE parameters.")

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=current_lr,
                weight_decay=current_weight_decay,
            )

            # Stage 2 Scheduler Logic (Warmup + Cosine Annealing)
            if self.scaled_warmup_steps_s2 > 0:
                logger.info(f"Stage 2: Linear warmup for {self.scaled_warmup_steps_s2} steps.")
                def lr_lambda_s2(current_step: int):
                    # current_step is relative to the start of Stage 2 scheduler
                    if current_step < self.scaled_warmup_steps_s2:
                        return float(current_step + 1) / float(max(1, self.scaled_warmup_steps_s2))
                    return 1.0
                warmup_scheduler_s2 = LambdaLR(optimizer, lr_lambda=lr_lambda_s2)

                total_epochs_in_run = self.trainer.max_epochs if self.trainer else self.hparams.model_config.get("max_epochs", 100) # Fallback
                epochs_in_stage2 = total_epochs_in_run - self.hparams.stage2_start_epoch
                
                T_max_s2_calc = (steps_per_epoch * epochs_in_stage2) - self.scaled_warmup_steps_s2
                T_max_s2 = self.scaled_steps_to_decay_s2 if self.scaled_steps_to_decay_s2 is not None and self.scaled_steps_to_decay_s2 > 0 else T_max_s2_calc
                
                if T_max_s2 <= 0:
                    logger.warning(f"Stage 2 Cosine Annealing duration (T_max_s2={T_max_s2}) is not positive. Only applying warmup.")
                    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": warmup_scheduler_s2, "interval": "step", "frequency": 1, "name": "lr_scheduler_stage2_warmup_only"}}

                eta_min_s2 = self.hparams.lr_stage2 * self.hparams.eta_min_fraction_s2
                logger.info(f"Stage 2: CosineAnnealingLR with T_max={T_max_s2}, eta_min={eta_min_s2}")
                cosine_scheduler_s2 = CosineAnnealingLR(optimizer, T_max=T_max_s2, eta_min=eta_min_s2)
                
                sequential_scheduler_s2 = SequentialLR(optimizer, schedulers=[warmup_scheduler_s2, cosine_scheduler_s2], milestones=[self.scaled_warmup_steps_s2])
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sequential_scheduler_s2, "interval": "step", "frequency": 1, "name": "lr_scheduler_stage2_sequential"}}
            else: # No warmup for Stage 2
                total_epochs_in_run = self.trainer.max_epochs if self.trainer else self.hparams.model_config.get("max_epochs", 100)
                epochs_in_stage2 = total_epochs_in_run - self.hparams.stage2_start_epoch
                T_max_s2_calc = steps_per_epoch * epochs_in_stage2
                T_max_s2 = self.scaled_steps_to_decay_s2 if self.scaled_steps_to_decay_s2 is not None and self.scaled_steps_to_decay_s2 > 0 else T_max_s2_calc

                if T_max_s2 <= 0:
                    logger.warning(f"Stage 2 Cosine Annealing duration (T_max_s2={T_max_s2}) is not positive (no warmup). Using constant LR.")
                    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": LambdaLR(optimizer, lr_lambda=lambda _: 1.0), "interval": "step", "frequency": 1, "name": "lr_scheduler_stage2_constant"}}

                eta_min_s2 = self.hparams.lr_stage2 * self.hparams.eta_min_fraction_s2
                logger.info(f"Stage 2: CosineAnnealingLR (no warmup) with T_max={T_max_s2}, eta_min={eta_min_s2}")
                cosine_scheduler_s2 = CosineAnnealingLR(optimizer, T_max=T_max_s2, eta_min=eta_min_s2)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": cosine_scheduler_s2, "interval": "step", "frequency": 1, "name": "lr_scheduler_stage2_cosine_only"}}
        else:
            logger.error(f"Unknown stage {self.stage} in configure_optimizers. Returning default optimizer.")
            return torch.optim.AdamW(self.parameters(), lr=1e-3) # Fallback
    
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
