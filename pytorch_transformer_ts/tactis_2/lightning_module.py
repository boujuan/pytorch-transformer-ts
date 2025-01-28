import pytorch_lightning as pl
import torch
from gluonts.torch.util import weighted_average

from pytorch_transformer_ts.tactis_2.module import TACTiSModel # Import TACTiSModel from module.py

class TACTiSLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: dict, # Expecting model params as dict now
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.model = TACTiSModel(**model) # Instantiate TACTiSModel here using params dict
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.current_batch = None # Store current batch for loss calculation in module

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        past_static_feat = batch["past_static_feat"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]

        self.current_batch = batch # Store batch

        # Calculate output and loss by forward pass
        outputs = self.model( # Call forward pass of TACTiSModel
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            past_static_feat=past_static_feat,
            future_time_feat=future_time_feat,
            future_target=future_target, # Pass future_target for potential conditioning or loss calculation
        )

        # Calculate loss using output_loss from TACTiSModel
        # Pass future_target and future_observed_values for loss calculation
        train_loss = self.model.output_loss(
            params=outputs, # Pass the output from forward pass as 'params'
            future_target=future_target,
            future_observed_values=future_observed_values,
        )

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        past_static_feat = batch["past_static_feat"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]

        self.current_batch = batch # Store batch

        # Calculate output and loss (similar to training_step)
        outputs = self.model( # Call forward pass of TACTiSModel
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            past_static_feat=past_static_feat,
            future_time_feat=future_time_feat,
            future_target=future_target, # Pass future_target for potential conditioning or loss calculation
        )

        # Calculate loss using output_loss from TACTiSModel
        val_loss = self.model.output_loss(
            params=outputs, # Pass the output from forward pass as 'params'
            future_target=future_target,
            future_observed_values=future_observed_values,
        )

        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr, # TODO: Check if this is correct
            weight_decay=self.weight_decay,
        )
        return optimizer