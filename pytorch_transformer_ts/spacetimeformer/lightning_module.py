# Use the newer namespace consistent with Lightning > v2.0
import lightning.pytorch as pl
import torch
# from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from pytorch_transformer_ts.spacetimeformer.module import SpacetimeformerModel

class SpacetimeformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        # model: SpacetimeformerModel,
        model_config: dict, # CHANGE
        # loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-4,
        weight_decay: float = 1e-8
    ) -> None:
        super().__init__()
        self.model = SpacetimeformerModel(**model_config)
        self.save_hyperparameters("model_config", "lr", "weight_decay")
        # self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

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
        self.log(
            "val_loss", 
            val_loss, 
                on_epoch=True, 
                on_step=True, 
                prog_bar=True,
                sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def forward(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

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
        loss_values = self.model.output_loss(params, future_target, loc=loc, scale=scale)

        # loss_values = self.loss(distr, future_target)
        
        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)