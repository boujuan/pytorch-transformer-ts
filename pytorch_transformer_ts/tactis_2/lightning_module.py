import pytorch_lightning as pl
import torch
# from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from .module import TACTiS2Model

class TACTiS2LightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the TACTiS2 model.
    """
    
    def __init__(
        self,
        model: TACTiS2Model,
        # loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        stage: int = 1,  # Start with stage 1 (flow-only)
        stage2_start_epoch: int = 10,  # When to start stage 2
    ) -> None:
        """
        Initialize the Lightning module.
        
        Parameters
        ----------
        model: TACTiS2Model
            The TACTiS2 model to train
        loss: DistributionLoss
            Loss function to use
        lr: float
            Learning rate
        weight_decay: float
            Weight decay coefficient
        stage: int
            Starting training stage (1=flow only, 2=full model)
        stage2_start_epoch: int
            Epoch at which to transition to stage 2
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.stage = stage
        self.stage2_start_epoch = stage2_start_epoch
        
        # Set initial stage
        self.model.set_training_stage(stage)

    def training_step(self, batch, batch_idx: int):
        """
        Execute training step.
        
        Parameters
        ----------
        batch: Dict
            Batch of data
        batch_idx: int
            Batch index
            
        Returns
        -------
        torch.Tensor
            Training loss
        """
        # Check if we should transition to stage 2
        current_epoch = self.trainer.current_epoch
        if self.stage == 1 and current_epoch >= self.stage2_start_epoch:
            self.stage = 2
            self.model.set_training_stage(2)
            self.log("training_stage", self.stage)
        
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
        """
        Execute validation step.
        
        Parameters
        ----------
        batch: Dict
            Batch of data
        batch_idx: int
            Batch index
            
        Returns
        -------
        torch.Tensor
            Validation loss
        """
        with torch.inference_mode():
            val_loss = self(batch)
        self.log(
            "val_loss", 
            val_loss, 
            on_epoch=True, 
            on_step=True, 
            prog_bar=True,
            sync_dist=True
        )
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        
        Returns
        -------
        torch.optim.Optimizer
            Optimizer
        """
        # In stage 1, only optimize flow parameters
        if self.stage == 1:
            params = []
            for name, param in self.model.named_parameters():
                if 'flow' in name and not 'copula' in name:
                    params.append(param)
        else:
            # In stage 2, optimize all parameters
            params = self.model.parameters()
        
        return torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def forward(self, batch):
        """
        Forward pass for training.
        
        Parameters
        ----------
        batch: Dict
            Batch of data
            
        Returns
        -------
        torch.Tensor
            Loss value
        """
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        inputs, loc, scale, static_feat, _ = self.model.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        
        # For TACTiS, we can directly use its own loss calculation
        _, loss = self.model.tactis(
            hist_time=inputs[0],
            hist_value=inputs[1],
            pred_time=inputs[2],
            pred_value=inputs[3],
            permute_series=True,  # Enable permutation during training
        )
        
        # Apply observation weights
        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss.unsqueeze(-1), weights=loss_weights)
