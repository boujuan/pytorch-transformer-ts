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
        
        # Store reference to Stage 2 optimizer for manual optimization
        self._stage2_optimizer = None
        self._stage2_scheduler = None
        
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
            
    def _get_marginal_parameters(self):
        """
        Get marginal/flow parameters that should be optimized in Stage 1.
        
        Returns:
            list: List of marginal/flow parameters that have requires_grad=True
        """
        marginal_parameter_names = [
            "flow_series_encoder",
            "flow_time_encoding", 
            "flow_input_encoder",
            "flow_encoder",
            "decoder.marginal",
        ]
        
        marginal_params = []
        marginal_param_count = 0
        
        for name, param in self.model.tactis.named_parameters():
            if any(pname in name for pname in marginal_parameter_names):
                if param.requires_grad:
                    marginal_params.append(param)
                    marginal_param_count += param.numel()
                    logger.debug(f"Marginal parameter: {name} ({param.numel()} elements)")
        
        logger.info(f"Found {len(marginal_params)} marginal parameter tensors, {marginal_param_count:,} total parameters")
        return marginal_params

    def _get_copula_parameters(self):
        """
        Get copula parameters that should be optimized in Stage 2.
        
        Returns:
            list: List of copula parameters that have requires_grad=True
        """
        copula_parameter_names = [
            "copula_series_encoder",
            "copula_time_encoding",
            "copula_input_encoder", 
            "copula_encoder",
            "decoder.copula",
        ]
        
        copula_params = []
        copula_param_count = 0
        
        for name, param in self.model.tactis.named_parameters():
            if any(pname in name for pname in copula_parameter_names):
                if param.requires_grad:
                    copula_params.append(param)
                    copula_param_count += param.numel()
                    logger.debug(f"Copula parameter: {name} ({param.numel()} elements)")
        
        logger.info(f"Found {len(copula_params)} copula parameter tensors, {copula_param_count:,} total parameters")
        return copula_params
    
    def _create_stage2_optimizer(self):
        """
        Create a fresh optimizer with ONLY copula parameters for Stage 2.
        This follows the original TACTiS-2 design of creating a new optimizer 
        rather than updating the existing one.
        
        Returns:
            torch.optim.Optimizer: Fresh optimizer configured for Stage 2 with only copula parameters
        """
        # Get only copula parameters that should be trained in Stage 2
        copula_params = self._get_copula_parameters()
        
        if not copula_params:
            logger.error("No copula parameters found for Stage 2 optimizer!")
            # Fallback to all parameters to prevent crash
            copula_params = list(self.model.tactis.parameters())
        
        # Create fresh optimizer with Stage 2 configuration
        optimizer_stage2 = torch.optim.AdamW(
            copula_params,
            lr=self.lr_stage2,
            weight_decay=self.weight_decay_stage2,
        )
        
        logger.info(f"Created fresh Stage 2 optimizer with {len(copula_params)} parameter tensors")
        logger.info(f"Stage 2 optimizer config: lr={self.lr_stage2}, weight_decay={self.weight_decay_stage2}")
        
        return optimizer_stage2
    
    def _create_stage2_scheduler(self, optimizer):
        """
        Create a fresh scheduler for Stage 2 optimizer.
        
        Parameters:
            optimizer: The Stage 2 optimizer to create scheduler for
        """
        # Get the scheduler configuration from configure_optimizers
        steps_per_epoch = self.hparams.num_batches_per_epoch
        if steps_per_epoch is None or steps_per_epoch <= 0:
            logger.warning(f"Invalid num_batches_per_epoch: {steps_per_epoch}. Using 100 as default.")
            steps_per_epoch = 100

        warmup_steps = self.warmup_steps_s2
        cosine_steps = self.steps_to_decay_s2
        
        if cosine_steps is None or cosine_steps <= 0:
            logger.warning(f"Invalid steps_to_decay_s2: {cosine_steps}. Using 1000 as default.")
            cosine_steps = 1000

        # Create warmup scheduler (if warmup_steps > 0)
        if warmup_steps > 0:
            def warmup_lr_lambda(step):
                return min(1.0, step / warmup_steps)
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
        else:
            warmup_scheduler = None

        # Create cosine annealing scheduler
        eta_min = self.lr_stage2 * self.hparams.eta_min_fraction_s2
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cosine_steps, 
            eta_min=eta_min
        )

        # Combine schedulers if warmup is used
        if warmup_scheduler is not None:
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = cosine_scheduler

        # Store scheduler for manual optimization use
        self._stage2_scheduler = scheduler
        
        # Replace in trainer's scheduler list
        if hasattr(self.trainer, 'lr_scheduler_configs') and self.trainer.lr_scheduler_configs:
            self.trainer.lr_scheduler_configs[0].scheduler = scheduler
            logger.info("Created and registered fresh Stage 2 scheduler")
        else:
            logger.warning("Could not register Stage 2 scheduler with trainer")
        
    def _apply_stage2_parameter_freezing(self):
        """
        Apply parameter freezing for Stage 2: freeze marginal/flow parameters, unfreeze copula parameters.
        """
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
                # Default: Keep requires_grad as is, but log for transparency
                logger.debug(f"Parameter '{name}' not explicitly frozen/unfrozen.")
                other_tensors += 1
                other_elements += param.numel()

        # Provide clear, accurate logging
        logger.info(f"Stage 2 Parameter Freezing Summary:")
        logger.info(f"  Frozen (marginal/flow): {frozen_tensors} tensors, {frozen_elements:,} parameters")
        logger.info(f"  Trainable (copula): {unfrozen_tensors} tensors, {unfrozen_elements:,} parameters")
        if other_tensors > 0:
            logger.info(f"  Other: {other_tensors} tensors, {other_elements:,} parameters")
        logger.info(f"  Total model: {frozen_tensors + unfrozen_tensors + other_tensors} tensors, {frozen_elements + unfrozen_elements + other_elements:,} parameters")
            
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

            # 1. Update the stage in the model 
            if hasattr(self.model.tactis, "set_stage"):
                self.model.tactis.set_stage(self.stage)
                logger.info(f"Epoch {current_epoch}: Called model.tactis.set_stage(2)")
            else:
                logger.warning("model.tactis does not have set_stage method.")

            # 2. Apply parameter freezing for Stage 2
            logger.info("Applying Stage 2 parameter freezing...")
            self._apply_stage2_parameter_freezing()

            # 3. MANUAL OPTIMIZATION: Create fresh optimizer with ONLY copula parameters
            logger.info("Creating fresh Stage 2 optimizer with only copula parameters...")
            try:
                # Create fresh optimizer with only copula parameters (original TACTiS design)
                fresh_optimizer = self._create_stage2_optimizer()
                
                # Store the fresh optimizer for manual optimization use
                self._stage2_optimizer = fresh_optimizer
                
                # Replace optimizer in trainer's internal list (for Lightning compatibility)
                if hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
                    self.trainer.optimizers[0] = fresh_optimizer
                    logger.info("Successfully replaced Stage 1 optimizer with fresh Stage 2 optimizer")
                
                # Log successful Stage 2 transition
                self.log_dict({
                    "stage": 2, 
                    "learning_rate": self.lr_stage2, 
                    "weight_decay": self.weight_decay_stage2,
                    "optimizer_param_count": sum(p.numel() for p in fresh_optimizer.param_groups[0]['params'])
                })
                
                logger.info(f"Epoch {current_epoch}: Successfully created fresh Stage 2 optimizer")
                logger.info(f"Stage 2 optimizer trains {sum(p.numel() for p in fresh_optimizer.param_groups[0]['params']):,} parameters (copula only)")
                
                # Create new scheduler for Stage 2 if needed
                self._create_stage2_scheduler(fresh_optimizer)
                
            except Exception as e:
                logger.error(f"Failed to create fresh Stage 2 optimizer: {e}")
                logger.warning("Continuing with existing optimizer configuration - Stage 2 may not work correctly")

            self.log("stage", self.stage, on_step=False, on_epoch=True)
                
    def training_step(self, batch, batch_idx: int):
        """
        Training step with manual optimization for two-stage training.
        
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
        # Manual optimization for proper two-stage training
        
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
        
        # Manual optimization - get the correct optimizer based on current stage
        if self.stage == 2 and hasattr(self, '_stage2_optimizer') and self._stage2_optimizer is not None:
            # Use Stage 2 fresh optimizer with only copula parameters
            opt = self._stage2_optimizer
            sch = getattr(self, '_stage2_scheduler', None)
        else:
            # Use Stage 1 optimizer (all parameters)
            opt = self.optimizers()
            sch = self.lr_schedulers()
        
        # Zero gradients
        opt.zero_grad()
        
        # Backward pass
        self.manual_backward(loss)
        
        # Apply gradient clipping based on current stage
        if self.stage == 1:
            clip_val = self.hparams.gradient_clip_val_stage1
        else:
            clip_val = self.hparams.gradient_clip_val_stage2
            
        if clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
        
        # Optimizer step
        opt.step()
        
        # Update learning rate scheduler
        if sch is not None:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(loss)
            else:
                sch.step()

        # TRAINING METRIC: Log what's being optimized for training (stage-specific)
        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        
        # OPTUNA METRICS: Always log components for consistent optimization
        tactis_model = getattr(self.model, 'tactis', None)
        if tactis_model and hasattr(tactis_model, 'copula_loss') and hasattr(tactis_model, 'marginal_logdet'):
            if tactis_model.copula_loss is not None and tactis_model.marginal_logdet is not None:
                # DIAGNOSTIC: Marginal stability (should be constant in Stage 2)
                self.log("train_marginal_logdet", tactis_model.marginal_logdet.detach(), 
                       on_step=False, on_epoch=True, prog_bar=False)
                
                # OPTUNA TARGET: Consistent metric across both stages
                total_nll = tactis_model.copula_loss - tactis_model.marginal_logdet
                self.log("train_total_nll", total_nll.detach(), 
                       on_step=False, on_epoch=True, prog_bar=False)
        
        # Return loss for logging (not used for backprop anymore)
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
        
        # TRAINING METRIC: Log what's being optimized for training (stage-specific)
        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        
        # OPTUNA METRICS: Always log components for consistent optimization
        tactis_model = getattr(self.model, 'tactis', None)
        if tactis_model and hasattr(tactis_model, 'copula_loss') and hasattr(tactis_model, 'marginal_logdet'):
            if tactis_model.copula_loss is not None and tactis_model.marginal_logdet is not None:
                # DIAGNOSTIC: Marginal stability (should be constant in Stage 2)
                self.log("val_marginal_logdet", tactis_model.marginal_logdet.detach(), 
                       on_step=False, on_epoch=True, prog_bar=False)
                
                # OPTUNA TARGET: Consistent metric across both stages
                total_nll = tactis_model.copula_loss - tactis_model.marginal_logdet
                self.log("val_total_nll", total_nll.detach(), 
                       on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
        
    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        STAGE-AWARE APPROACH: 
        - Stage 1: Create optimizer with ALL parameters (matches original TACTiS design)
        - Stage 2: Fresh optimizer with ONLY copula parameters created during transition
        
        This method creates the initial Stage 1 optimizer. Stage 2 transition 
        replaces this with a fresh optimizer containing only copula parameters.
        """
        # Always start with Stage 1 configuration
        # Stage 2 transition will replace this with fresh copula-only optimizer
        initial_lr = self.lr_stage1
        initial_weight_decay = self.weight_decay_stage1

        logger.info(f"Configuring Stage 1 optimizer with lr={initial_lr}, weight_decay={initial_weight_decay}")

        # Create Stage 1 optimizer with ALL model parameters (original TACTiS design)
        optimizer = torch.optim.AdamW(
            self.model.tactis.parameters(),  # All parameters for Stage 1
            lr=initial_lr,
            weight_decay=initial_weight_decay,
        )
        
        logger.info(f"Created optimizer with {len(list(self.model.tactis.parameters()))} parameters")
        
        # Use manual optimization for two-stage training with optimizer switching
        self.automatic_optimization = False
        
        # Get steps_per_epoch from hyperparameters
        steps_per_epoch = self.hparams.num_batches_per_epoch
        if steps_per_epoch is None or steps_per_epoch <= 0:
            logger.warning(f"Invalid num_batches_per_epoch: {steps_per_epoch}. Using 100 as default.")
            steps_per_epoch = 100
            
        # Create comprehensive scheduler that handles both stages
        logger.info(f"Scheduler setup - steps_per_epoch: {steps_per_epoch}, stage2_start_epoch: {self.hparams.stage2_start_epoch}")
        
        # Create warmup scheduler for Stage 1
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
            if self.steps_to_decay_s1 is not None and self.steps_to_decay_s1 > 0:
                T_max_s1 = self.steps_to_decay_s1
                logger.info(f"Using steps_to_decay_s1={T_max_s1} as T_max for Stage 1 CosineAnnealingLR.")
            else:
                epochs_in_stage1 = self.hparams.stage2_start_epoch
                T_max_s1_calculated = (steps_per_epoch * epochs_in_stage1) - self.warmup_steps_s1
                T_max_s1 = T_max_s1_calculated
                logger.info(f"Calculating T_max_s1 = ({steps_per_epoch} * {epochs_in_stage1}) - {self.warmup_steps_s1} = {T_max_s1}")
            
            # Ensure T_max is valid
            if T_max_s1 <= 0:
                logger.warning(f"Stage 1 T_max ({T_max_s1}) not positive. Using warmup only.")
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": warmup_scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "name": "lr_scheduler_warmup_only",
                    },
                }
            
            # Calculate eta_min for Stage 1
            eta_min_s1 = self.hparams.lr_stage1 * self.hparams.eta_min_fraction_s1
            logger.info(f"Stage 1 CosineAnnealingLR: T_max={T_max_s1}, eta_min={eta_min_s1}")
            
            # Create cosine annealing scheduler for Stage 1
            cosine_scheduler_s1 = CosineAnnealingLR(optimizer, T_max=T_max_s1, eta_min=eta_min_s1)
            self.cosine_scheduler_ref = cosine_scheduler_s1  # Store reference
            
            # Create sequential scheduler
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
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr_scheduler_main",
                },
            }
        else:
            # No warmup, just cosine annealing for total training duration
            total_epochs = self.hparams.max_epochs if hasattr(self.hparams, 'max_epochs') else 100
            T_max_total = steps_per_epoch * total_epochs
            eta_min = self.hparams.lr_stage1 * self.hparams.eta_min_fraction_s1
            
            logger.info(f"No warmup CosineAnnealingLR: T_max={T_max_total}, eta_min={eta_min}")
            
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max_total, eta_min=eta_min)
            self.cosine_scheduler_ref = cosine_scheduler
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": cosine_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr_scheduler_main",
                },
            }
    
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
