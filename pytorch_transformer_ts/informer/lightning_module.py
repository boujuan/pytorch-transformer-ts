# Use the newer namespace consistent with Lightning > v2.0
import lightning.pytorch as pl
import torch
# from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

# CHANGE
from pytorch_transformer_ts.informer.module import InformerModel


class InformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        # loss: DistributionLoss = NegativeLogLikelihood(), CHANGE
        lr: float = 1e-4,
        weight_decay: float = 1e-8
    ) -> None:
        super().__init__()
        
        # if isinstance(model_config, dict):
        self.model = InformerModel(**model_config)
        self.save_hyperparameters("model_config", "lr", "weight_decay")
        # else:
        #     self.model = model
        #     self.save_hyperparameters(ignore=["model"])
        # self.loss = loss CHANGE
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
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
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
