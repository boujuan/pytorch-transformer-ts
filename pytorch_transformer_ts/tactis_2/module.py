import torch
import torch.nn as nn
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.core.component import validated
from typing import List, Optional, Dict, Any

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
        num_series: int, # Redundant, already inferred from cardinality ideally, but kept for compatibility
        flow_series_embedding_dim: int,
        copula_series_embedding_dim: int,
        flow_input_encoder_layers: int,
        copula_input_encoder_layers: int,
        num_layers_encoder: int,
        num_decoder_layers: int,
        num_parallel_samples: int = 100,
        embedding_dimension: Optional[List[int]] = None, # Redundant, embedding dims should be infered from model_params
        scaling: Optional[str] = "mean",
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None, # Not used in TACTIS directly, remove if not needed
        model_parameters: Optional[Dict[str, Any]] = None, # To pass model params as dict for clarity
        **kwargs,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality
        self.num_parallel_samples = num_parallel_samples
        self.lags_seq = lags_seq # Remove if not needed

        if scaling == "mean":
            self.scaler = MeanScaler(dim=-1, keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(dim=-1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=-1, keepdim=True)

        if num_feat_static_cat > 0:
            self.static_feat_embedding = nn.Embedding(
                num_embeddings=sum(cardinality), embedding_dim=embedding_dimension[0] if embedding_dimension else 32
            )
        else:
            self.static_feat_embedding = None

        # Determine time feature dimension based on estimator config
        time_feature_dim = 0
        if time_features_from_frequency_str(freq): # Assuming default time features are used if time_features is None
            time_feature_dim += len(time_features_from_frequency_str(freq))
        if True: # Assuming age feature is always added, adjust if not
            time_feature_dim += 1
        # Add dynamic real features if used
        time_feature_dim += num_feat_dynamic_real

        # Initialize TACTiS model, passing model_parameters or individual params
        if model_parameters is None:
            model_params = { # Default values, adjust as needed or get from kwargs
                'num_series': num_series,
                'flow_series_embedding_dim': flow_series_embedding_dim,
                'copula_series_embedding_dim': copula_series_embedding_dim,
                'flow_input_encoder_layers': flow_input_encoder_layers,
                'copula_input_encoder_layers': copula_input_encoder_layers,
                'num_layers_encoder': num_layers_encoder,
                'num_decoder_layers': num_decoder_layers,
            }
        else:
            model_params = model_parameters

        self.tactis = TACTiS(**model_params) # Pass params dict directly

    def forward(self, past_target, past_observed_values, past_time_feat, past_static_feat, future_time_feat, future_target = None):
        """
        Computes the forward pass of the TACTiS model.

        Args:
            past_target: Past target values.
            past_observed_values: Past observed values indicator.
            past_time_feat: Past time features.
            past_static_feat: Past static features.
            future_time_feat: Future time features.
            future_target: Future target values (optional, for training).

        Returns:
            The output of the TACTiS model.
        """
        # Scaling past target
        scaled_past_target, past_target_scale = self.scaler(past_target, past_observed_values)

        # Static features (if any) - currently not used in forward pass as per provided code, but kept for potential future use
        # static_feat = torch.cat(
        #     [past_static_feat, past_static_cat],
        #     dim=1,
        # )

        # Call the core TACTiS model
        samples = self.tactis(
            hist_value=scaled_past_target, # Use scaled past target
            hist_time=past_time_feat.permute(0, 2, 1), # Permute time features to [B, feature_dim, time] -> [B, time, feature_dim]
            pred_time=future_time_feat.permute(0, 2, 1), # Permute future time features similarly
            num_samples=self.num_parallel_samples, # Pass num_samples for sampling in TACTiS
        )

        # Unscale the samples - IMPORTANT: unscale using the *past_target_scale*
        if self.scaler is not None: # Only unscale if scaler is used
            samples = self.scaler.inverse_transform(samples, past_target_scale)

        return samples # Return samples from TACTiS

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

    def output_loss(self, params, future_target, future_observed_values=None, loc=None, scale=None): # Added future_observed_values, loc, scale - might not be used but kept for consistency
        """
        Compute the loss for the output distribution.

        Args:
            params: Output of the model (samples from TACTiS in this case).
            future_target: Target.
            future_observed_values: Future observed values (optional).
            loc: Location parameter (optional).
            scale: Scale parameter (optional).

        Returns:
            torch.Tensor: Loss.
        """
        # 'params' are samples from TACTiS. We need to calculate the TACTiS loss here.
        # Assuming self.tactis.loss() calculates and returns marginal_logdet and copula_loss
        hist_time = self.current_batch['past_time_feat'].permute(0, 2, 1) # Get hist_time from current batch
        hist_value = self.current_batch['past_target'].unsqueeze(1) # Get hist_value from current batch
        pred_time = self.current_batch['future_time_feat'].permute(0, 2, 1) # Get pred_time from current batch
        pred_value = future_target.unsqueeze(1) # Use provided future_target as pred_value

        marginal_logdet, copula_loss = self.tactis.loss(
            hist_time=hist_time,
            hist_value=hist_value,
            pred_time=pred_time,
            pred_value=pred_value,
        )
        loss_values = copula_loss - marginal_logdet

        if future_observed_values is not None: # Check if future_observed_values is provided
            if len(self.target_shape) == 0:
                loss_weights = future_observed_values
            else:
                loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)
            return weighted_average(loss_values, weights=loss_weights)
        else:
            return loss_values # Return loss directly if no weights are needed

    def output_params(self, tactis_outputs):
        """
        Compute the parameters for the output distribution.

        Args:
            tactis_outputs: Output of the TACTiS model.

        Returns:
            Tuple[torch.Tensor, ...]: Parameters for the output distribution.
        """
        return self.distr_output.args_from_values(tactis_outputs)