import torch
import torch.nn as nn
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.core.component import validated

# Import TACTiS from the new tactis.py file
from .tactis import TACTiS

class TACTiSModel(nn.Module):

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        # TACTiS arguments
        num_series: int,
        num_layers_encoder: int,
        num_layers_decoder: int,
        num_parallel_samples: int = 100,
        embedding_dimension: Optional[List[int]] = None,
        scaling: Optional[str] = "mean",
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape
        self.num_parallel_samples = num_parallel_samples
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        if num_feat_static_cat > 0:
            self.static_feat_embedding = nn.Embedding(
                num_embeddings=sum(cardinality), embedding_dim=embedding_dimension[0] if embedding_dimension else 32
            )
        else:
            self.static_feat_embedding = None

        self.tactis = TACTiS(
            num_series=num_series,
            flow_series_embedding_dim=embedding_dimension,
            copula_series_embedding_dim=embedding_dimension,
            flow_input_encoder_layers=num_layers_encoder,
            copula_input_encoder_layers=num_layers_encoder,
            bagging_size=None,
            input_encoding_normalization=True,
            data_normalization="standardization",
            loss_normalization="series",
            positional_encoding={
                "embedding_dim": 32,
                "max_len": 2048,
            },
            flow_encoder={
                "embedding_dim": 32,
                "num_heads": 4,
                "num_layers": num_layers_encoder,
                "dropout": 0.1,
            },
            copula_encoder={
                "embedding_dim": 32,
                "num_heads": 4,
                "num_layers": num_layers_encoder,
                "dropout": 0.1,
            },
            flow_temporal_encoder=None,
            copula_temporal_encoder=None,
            copula_decoder={
                "marginal_dist_type": "piecewise_linear",
                "marginal_dist_args": {
                    "num_pieces": 32,
                },
                "attentional_copula": {
                    "input_dim": 32,
                    "num_heads": 4,
                    "num_layers": num_layers_decoder,
                    "dropout": 0.1,
                },
                "copula_type": "gaussian",
                "copula_args": {},
            },
            skip_copula=False,
            experiment_mode="forecasting",
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_static_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        past_target_scaled, loc, scale = self.scaler(past_target, past_observed_values)

        # 1. Define hist_time, hist_value, pred_time, pred_value
        hist_time = past_time_feat.permute(0, 2, 1)  # Assuming (batch, time, feat) -> (batch, series, time)
        hist_value = past_target_scaled.unsqueeze(1) # Assuming shape (batch, time) -> (batch, series=1, time)
        pred_time = future_time_feat.permute(0, 2, 1)  # (batch, time, feat) -> (batch, series, time)

        if future_target is not None:
            future_target_scaled, _, _ = self.scaler(future_target, torch.ones_like(future_target))
            pred_value = future_target_scaled.unsqueeze(1)  # (batch, time) -> (batch, series=1, time)
        else:
            pred_value = None

        # 2. Handle static features (if needed)
        # Example: Embed and concatenate to hist_value
        if self.num_feat_static_cat > 0:
            static_feat_emb = self.static_feat_embedding(past_static_feat) # Assuming you add an embedding layer
            static_feat_emb = static_feat_emb.unsqueeze(2).expand(-1, -1, hist_value.shape[2]) # Expand to match time dimension
            hist_value = torch.cat([hist_value, static_feat_emb], dim=1) # Concatenate along series dimension

        # 3. Call tactis.forward with the correct arguments
        nn_out = self.tactis(
            hist_time,
            hist_value,
            pred_time,
            pred_value
        )

        # Slice output to obtain only the prediction length
        sliced_nn_out = nn_out[:, -self.prediction_length:, :]

        # Scale the output back to the original scale
        loc = loc.unsqueeze(1).expand(-1, self.prediction_length, -1)
        scale = scale.unsqueeze(1).expand(-1, self.prediction_length, -1)
        
        samples = self.scaler.inv_scale(sliced_nn_out, scale, loc)

        return samples.unsqueeze(1)

    def output_distribution(self, params, scale=None, loc=None):
        """
        Construct the output distribution.

        Args:
            params: Output of the model.
            scale: Scale of the target.
            loc: Location of the target.

        Returns:
            DistributionOutput: Output distribution.
        """
        return self.distr_output.distribution(params, scale=scale, loc=loc)

    def output_loss(self, params, future_target, loc=None, scale=None):
        """
        Compute the loss for the output distribution.

        Args:
            params: Output of the model.
            future_target: Target.
            loc: Location of the target.
            scale: Scale of the target.

        Returns:
            torch.Tensor: Loss.
        """
        distr = self.output_distribution(params, loc=loc, scale=scale)
        loss = distr.loss(future_target)
        return loss
    
    def output_params(self, tactis_outputs):
        """
        Compute the parameters for the output distribution.

        Args:
            tactis_outputs: Output of the TACTiS model.

        Returns:
            Tuple[torch.Tensor, ...]: Parameters for the output distribution.
        """
        return self.distr_output.args_from_values(tactis_outputs)