import pytorch_lightning as pl
import torch
from gluonts.torch.util import weighted_average

class TACTiSLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: TACTiSModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        past_static_feat = batch["past_static_feat"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]

        # Calculate output and loss
        outputs = self.model(
            past_target,
            past_observed_values,
            past_time_feat,
            past_static_feat,
            future_time_feat,
            future_target,
        )
        
        # Call TACTiS loss function (make sure arguments are correct)
        hist_time = past_time_feat.permute(0, 2, 1)
        hist_value = past_target.unsqueeze(1)
        pred_time = future_time_feat.permute(0, 2, 1)
        pred_value = future_target.unsqueeze(1)
        
        loss = self.model.tactis.loss(
            hist_time,
            hist_value,
            pred_time,
            pred_value,
            outputs,
        )

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        past_static_feat = batch["past_static_feat"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]

        # Calculate output and loss (similar to training_step)
        outputs = self.model(
            past_target,
            past_observed_values,
            past_time_feat,
            past_static_feat,
            future_time_feat,
            future_target,
        )

        hist_time = past_time_feat.permute(0, 2, 1)
        hist_value = past_target.unsqueeze(1)
        pred_time = future_time_feat.permute(0, 2, 1)
        pred_value = future_target.unsqueeze(1)

        loss = self.model.tactis.loss(
            hist_time,
            hist_value,
            pred_time,
            pred_value,
            outputs,
        )

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer