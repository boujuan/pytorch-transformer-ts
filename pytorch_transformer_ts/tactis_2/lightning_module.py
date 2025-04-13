# Use the newer namespace consistent with Lightning > v2.0
import lightning.pytorch as pl
import torch
# from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average
from gluonts.dataset.field_names import FieldName

from .module import TACTiS2Model

class TACTiS2LightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the TACTiS2 model.
    """
    
    def __init__(
        self,
        model: TACTiS2Model,
        lr_stage1: float = 1.8e-3,
        lr_stage2: float = 7.0e-4,
        weight_decay_stage1: float = 0.0,
        weight_decay_stage2: float = 0.0,
        stage: int = 1,  # Start with stage 1 (flow-only)
        stage2_start_epoch: int = 10,  # When to start stage 2
    ) -> None:
        """
        Initialize the TACTiS2 Lightning Module.
        
        Parameters
        ----------
        model
            The TACTiS2 model to train.
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
        """
        super().__init__()
        if isinstance(model, dict):
            self.model = TACTiS2Model(**model)
            self.save_hyperparameters()
        else:
            self.model = model
            self.save_hyperparameters(ignore=["model"])
        # Store stage-specific optimizer parameters
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.weight_decay_stage1 = weight_decay_stage1
        self.weight_decay_stage2 = weight_decay_stage2
        self.stage = stage
        self.stage2_start_epoch = stage2_start_epoch
        
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
            self.stage = 2
            # Update optimizer parameters for stage 2
            optimizer = self.optimizers()
            # Check if optimizer is a list (e.g., with scheduler) or single instance
            if isinstance(optimizer, list):
                 optimizer = optimizer[0] # Assume first element is the optimizer

            if optimizer:
                 for param_group in optimizer.param_groups:
                     param_group['lr'] = self.lr_stage2
                     param_group['weight_decay'] = self.weight_decay_stage2
                 logging.info(f"Epoch {current_epoch}: Switched to Stage 2. Updated optimizer lr={self.lr_stage2}, weight_decay={self.weight_decay_stage2}")
            else:
                 logging.warning(f"Epoch {current_epoch}: Tried to switch to Stage 2, but optimizer not found.")

            if hasattr(self.model.tactis, "set_stage"):
                self.model.tactis.set_stage(self.stage)
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
        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        
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
            print(f"Training - Model returned tuple: predictions shape={predictions.shape if hasattr(predictions, 'shape') else 'N/A'}, loss={loss}")
        else:
            # If it's not a tuple, assume it's just the loss
            loss = model_output
            print(f"Training - Model returned scalar loss: {loss}")
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            print("WARNING: NaN detected in loss! Replacing with large value to continue training.")
            loss = torch.nan_to_num(loss, nan=1000.0)  # Use a large value but not too large
        
        # Manual backward pass
        self.manual_backward(loss)
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Step the optimizer
        opt.step()
        
        # Log the loss
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        
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
            print(f"Validation - Model returned tuple: predictions shape={predictions.shape if hasattr(predictions, 'shape') else 'N/A'}, loss={loss}")
        else:
            # If it's not a tuple, assume it's just the loss
            loss = model_output
            print(f"Validation - Model returned scalar loss: {loss}")
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            print("WARNING: NaN detected in validation loss! Replacing with large value for logging.")
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

        logging.info(f"Configuring optimizer for Stage {self.stage} with lr={current_lr}, weight_decay={current_weight_decay}")

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=current_lr,
            weight_decay=current_weight_decay,
        )
        
        # Configure gradient clipping directly in the lightning module
        # This helps with numerical stability by preventing extreme gradient values
        self.automatic_optimization = False
        
        # Can add learning rate scheduler here if needed
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
    
    def forward(self, batch):
        """
        Forward pass through the model for inference.
        
        Parameters
        ----------
        batch
            The input batch.
            
        Returns
        -------
        The output of the model (predictions).
        """
        # Fields for prediction are different from training
        feat_static_cat = batch.get("feat_static_cat", torch.zeros((batch["past_target"].shape[0], 1), device=self.device, dtype=torch.long))
        feat_static_real = batch.get("feat_static_real", torch.zeros((batch["past_target"].shape[0], 1), device=self.device, dtype=torch.float32))
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        future_time_feat = batch["future_time_feat"]
        
        # Call model's forward
        model_output = self.model(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
        )
        
        # Handle tuple return value from the model during inference
        if isinstance(model_output, tuple):
            # For inference, we want the predictions (first element of the tuple)
            predictions, _ = model_output
            print(f"Inference - Model returned tuple, using predictions with shape={predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
            return predictions
        else:
            # If it's not a tuple, return as is
            print(f"Inference - Model returned single output")
            return model_output
