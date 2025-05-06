from typing import List, Optional
import logging

import torch
import torch.nn as nn
# import numpy as np

from gluonts.core.component import validated
# from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import get_lags_for_frequency
# from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, StdScaler, NOPScaler

from .tactis import TACTiS

# Set up logging
logger = logging.getLogger(__name__)

class TACTiS2Model(nn.Module):
    """
    Module connecting TACTiS2 model to GluonTS framework, handling data transformations.
    
    This module bridges between the core TACTiS2 model and the GluonTS API expectations,
    managing input/output transformations, scaling, and distribution parameters.
    """
    
    @validated()
    def __init__(
        self,
        # Data dimensions
        num_series: int,
        context_length: int,
        prediction_length: int,
        # --- TACTiS specific parameters ---
        # Passed directly from Estimator
        flow_series_embedding_dim: int,
        copula_series_embedding_dim: int,
        flow_input_encoder_layers: int, # Marginal CDF Encoder input layers
        copula_input_encoder_layers: int, # Attentional Copula Encoder input layers
        marginal_embedding_dim_per_head: int,
        marginal_num_heads: int,
        marginal_num_layers: int, # Marginal CDF Encoder transformer layers
        copula_embedding_dim_per_head: int,
        copula_num_heads: int,
        copula_num_layers: int, # Attentional Copula Encoder transformer layers
        decoder_dsf_num_layers: int,
        decoder_dsf_hidden_dim: int,
        decoder_mlp_num_layers: int,
        decoder_mlp_hidden_dim: int,
        decoder_transformer_num_layers: int,
        decoder_transformer_embedding_dim_per_head: int,
        decoder_transformer_num_heads: int,
        decoder_num_bins: int, # Corresponds to AttentionalCopula resolution
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        loss_normalization: str = "series",
        encoder_type: str = "standard",
        dropout_rate: float = 0.1,
        # GluonTS compatability parameters
        cardinality: List[int] = [1],
        num_feat_dynamic_real: int = 0,
        num_feat_static_real: int = 0,
        num_feat_static_cat: int = 0,
        embedding_dimension: Optional[List[int]] = None,
        scaling: Optional[str] = "std", # Note: TACTiS handles internal scaling/normalization
        lags_seq: Optional[List[int]] = None,
        num_parallel_samples: int = 100,
        stage: int = 1,  # Add stage parameter with default value 1
        attentional_copula_kwargs: Optional[dict] = None,  # Parameters for AttentionalCopula component
    ) -> None:
        """
        Initialize the TACTiS2Model.
        
        Parameters
        ----------
        num_series
            Number of time series (variables) to forecast.
        context_length
            Length of the input context (history).
        prediction_length
            Length of the forecast horizon.
        flow_series_embedding_dim
            Dimension of the flow series embedding.
        copula_series_embedding_dim
            Dimension of the copula series embedding.
        flow_input_encoder_layers
            Number of layers in the flow input encoder.
        copula_input_encoder_layers
            Number of layers in the copula input encoder.
        bagging_size
            Size of the bagging ensemble. If None, no bagging is performed.
        input_encoding_normalization
            Whether to normalize the input encoding.
        loss_normalization
            Type of loss normalization to apply. Options: "none", "series".
        cardinality
            List of cardinalities of the categorical features.
        num_feat_dynamic_real
            Number of dynamic real features.
        num_feat_static_real
            Number of static real features.
        num_feat_static_cat
            Number of static categorical features.
        embedding_dimension
            List of embedding dimensions for categorical features.
        scaling
            Type of scaling to apply to the target. Options: "mean", "std", None.
        lags_seq
            Sequence of lags to use as features.
        num_parallel_samples
            Number of samples to generate in parallel during inference.
        attentional_copula_kwargs
            Optional dictionary containing specific arguments for the AttentionalCopula.
        """
        super().__init__()
        
        self.num_series = num_series
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # GluonTS compatibility parameters
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_real = num_feat_static_real
        self.num_feat_static_cat = num_feat_static_cat
        self.embedding_dimension = embedding_dimension
        self.cardinality = cardinality
        self.num_parallel_samples = num_parallel_samples
        
        # Set up scaling
        self.scaling = scaling
        if scaling == "mean":
            self.scaler = MeanScaler(dim=1, keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
            
        # Create embeddings for static categorical features if needed
        if self.num_feat_static_cat > 0:
            self.embedder = nn.ModuleList([
                nn.Embedding(cardinality, embedding_dim)
                for cardinality, embedding_dim in zip(
                    self.cardinality, 
                    self.embedding_dimension if self.embedding_dimension is not None else
                    [min(50, (cat + 1) // 2) for cat in self.cardinality]
                )
            ])
            self.embedding_dim = sum(
                embedding_dim 
                for embedding_dim in (
                    self.embedding_dimension if self.embedding_dimension is not None else
                    [min(50, (cat + 1) // 2) for cat in self.cardinality]
                )
            )
        else:
            self.embedder = None
            self.embedding_dim = 0
            
        # Set up lags_seq
        self.lags_seq = lags_seq or [0]
        
        # --- Create the core TACTiS model ---
        # Construct configuration dictionaries based on passed parameters

        # Calculate d_model for transformers
        marginal_d_model = marginal_num_heads * marginal_embedding_dim_per_head
        copula_d_model = copula_num_heads * copula_embedding_dim_per_head
        decoder_transformer_d_model = decoder_transformer_num_heads * decoder_transformer_embedding_dim_per_head

        # Assume positional encoding uses the larger of the two main embedding dims
        pos_encoding_dim = max(marginal_d_model, copula_d_model)
        positional_encoding_args = {
            "embedding_dim": pos_encoding_dim,
            "dropout": dropout_rate,
            "max_len": 5000,
        }

        flow_encoder_args = {
            "d_model": marginal_d_model,
            "nhead": marginal_num_heads,
            "num_encoder_layers": marginal_num_layers,
            "dim_feedforward": marginal_d_model * 4, # Standard practice
            "dropout": dropout_rate,
        }

        copula_encoder_args = {
            "d_model": copula_d_model,
            "nhead": copula_num_heads,
            "num_encoder_layers": copula_num_layers,
            "dim_feedforward": copula_d_model * 4, # Standard practice
            "dropout": dropout_rate,
        }

        # Note: flow_temporal_encoder and copula_temporal_encoder args are not directly in the table
        # We'll keep their structure but use the main encoder params for now.
        # Ideally, these would also be tunable parameters.
        flow_temporal_encoder_args = flow_encoder_args.copy()
        copula_temporal_encoder_args = copula_encoder_args.copy()

        copula_decoder_args = {
            # Dimensions might need adjustment based on actual encoder outputs
            "flow_input_dim": marginal_d_model,
            "copula_input_dim": copula_d_model,
            "min_u": 0.0,
            "max_u": 1.0,
            "skip_sampling_marginal": False,
            "attentional_copula": {
                "input_dim": copula_d_model, # Use copula encoder output dim
                "attention_heads": decoder_transformer_num_heads,
                "attention_layers": decoder_transformer_num_layers,
                "attention_dim": decoder_transformer_embedding_dim_per_head, # Dim per head
                "mlp_layers": 2, # Default value, will be overridden if provided
                "mlp_dim": decoder_transformer_d_model * 4, # Default value, will be overridden if provided
                "resolution": decoder_num_bins,
                "dropout": dropout_rate, # Use configured dropout
                "attention_mlp_class": "_easy_mlp", # Default value
                "activation_function": "relu", # Default value, will be overridden if provided
            },
            "dsf_marginal": {
                "context_dim": marginal_d_model, # Use flow encoder output dim
                "mlp_layers": decoder_mlp_num_layers,
                "mlp_dim": decoder_mlp_hidden_dim,
                "flow_layers": decoder_dsf_num_layers,
                "flow_hid_dim": decoder_dsf_hidden_dim,
            },
            "skip_copula": False, # Will be controlled by LightningModule stage
        }

        # Apply AttentionalCopula custom parameters if provided
        if attentional_copula_kwargs is not None and isinstance(attentional_copula_kwargs, dict):
            logger.info(f"Applying custom AttentionalCopula parameters from kwargs: {attentional_copula_kwargs}")
            # Update the attentional_copula dictionary within copula_decoder_args
            if "attentional_copula" in copula_decoder_args:
                copula_decoder_args["attentional_copula"].update(attentional_copula_kwargs)
            else:
                logger.warning("attentional_copula key missing in copula_decoder_args, creating it.")
                copula_decoder_args["attentional_copula"] = attentional_copula_kwargs

        # Initialize the TACTiS model with constructed args
        self.tactis = TACTiS(
            num_series=num_series,
            flow_series_embedding_dim=flow_series_embedding_dim, # Base embedding before input encoder
            copula_series_embedding_dim=copula_series_embedding_dim, # Base embedding before input encoder
            flow_input_encoder_layers=flow_input_encoder_layers, # Passed directly
            copula_input_encoder_layers=copula_input_encoder_layers, # Passed directly
            bagging_size=bagging_size,
            input_encoding_normalization=input_encoding_normalization,
            loss_normalization=loss_normalization,
            positional_encoding=positional_encoding_args,
            flow_encoder=flow_encoder_args,
            copula_encoder=copula_encoder_args,
            flow_temporal_encoder=flow_temporal_encoder_args, # Using flow_encoder args for now
            copula_temporal_encoder=copula_temporal_encoder_args, # Using copula_encoder args for now
            copula_decoder=copula_decoder_args,
            encoder_type=encoder_type,
            stage=stage,  # Pass the stage parameter to TACTiS
        )
    
    @property
    def _past_length(self) -> int:
        """
        Return the required length of the input.
        
        Returns
        -------
        The required length of the past input.
        """
        return self.context_length + max(self.lags_seq)
    
    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Create lagged subsequences of a given sequence.
        
        Parameters
        ----------
        sequence
            The input sequence (batch, time, variables).
        subsequences_length
            Length of each subsequence.
        shift
            Shift for the subsequences.
        
        Returns
        -------
        Lagged subsequences of the input.
        """
        batch_size = sequence.shape[0]
        
        # Shift for lags
        indices = [l - shift for l in self.lags_seq]
        
        # Check if we need padding
        pad_length = max(-min(indices) + subsequences_length, 0)
        if pad_length > 0:
            # Add padding
            padding = torch.zeros(
                batch_size, pad_length, sequence.shape[2], device=sequence.device
            )
            sequence = torch.cat([padding, sequence], dim=1)
            indices = [l + pad_length for l in indices]
            
        # Create subsequences
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            
            subsequence = sequence[:, begin_index:end_index, :]
            lagged_values.append(subsequence)
            
        return torch.cat(lagged_values, dim=-1)
    
    def create_network_inputs(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        """
        Create the inputs for the network.
        
        Parameters
        ----------
        feat_static_cat
            Static categorical features (batch, num_features).
        feat_static_real
            Static real features (batch, num_features).
        past_time_feat
            Past time features (batch, time, num_features).
        past_target
            Past target values (batch, time, variables).
        past_observed_values
            Indicator for observed values in the past (batch, time, variables).
        future_time_feat
            Future time features (batch, prediction_length, num_features).
        future_target
            Future target values (batch, prediction_length, variables).
            
        Returns
        -------
        Dict with prepared inputs for the network.
        """
        # Handle static features
        if self.num_feat_static_cat > 0:
            # Get embeddings for categorical features
            embedded_cat = torch.cat(
                [embed(feat_static_cat[:, i]) for i, embed in enumerate(self.embedder)],
                dim=1
            )
        else:
            embedded_cat = None
            
        # Static real features
        if self.num_feat_static_real > 0:
            static_real = feat_static_real
        else:
            static_real = None
            
        # Scale the target - take only observed values into account
        scaled_past_target, loc, scale = self.scaler(
            past_target,
            past_observed_values,
        )
        
        # Create lagged features
        subsequences_length = self.context_length
        lagged_sequence = self.get_lagged_subsequences(
            sequence=scaled_past_target,
            subsequences_length=subsequences_length,
        )
        
        # Time features
        time_feat = past_time_feat[:, -subsequences_length:, ...]
        
        # Prepare input history tensor for TACTiS
        history_values = scaled_past_target[:, -subsequences_length:, ...]  # (batch, context_length, variables)
        
        # Create timestamps for input to TACTiS
        # TACTiS expects integer timesteps, so we create sequential values
        batch_size = history_values.shape[0]
        history_time = torch.arange(
            0, self.context_length, device=history_values.device
        ).unsqueeze(0).repeat(batch_size, 1)  # (batch, context_length)
        
        # Prepare future timestamps
        future_time = torch.arange(
            self.context_length, 
            self.context_length + self.prediction_length, 
            device=history_values.device
        ).unsqueeze(0).repeat(batch_size, 1)  # (batch, prediction_length)
        
        # For training, we need future values too
        if future_target is not None:
            future_values = future_target
            if self.scaling != "none":
                future_values = (future_values - loc) / scale
        else:
            future_values = None
            
        return dict(
            past_target=history_values.transpose(1, 2),  # TACTiS expects (batch, variables, time)
            past_observed_values=past_observed_values.transpose(1, 2),
            hist_time=history_time,
            pred_time=future_time,
            future_target=future_values.transpose(1, 2) if future_values is not None else None,
            scaled_past_target=scaled_past_target,
            loc=loc,
            scale=scale,
            lagged_sequence=lagged_sequence,
            time_feat=time_feat,
            static_feat=static_real,
            embedded_cat=embedded_cat,
        )

    def output_params(self, network_input):
        """
        Compute the parameters of the output distribution or generate samples.
        
        Parameters
        ----------
        network_input : dict
            Output from create_network_inputs containing processed tensors including:
            - hist_time [batch, context_length]
            - past_target [batch, series, context_length]
            - pred_time [batch, prediction_length]
            - future_target [batch, series, prediction_length] (only for training)
            - scale and loc for denormalization
            
        Returns
        -------
        For training mode (when future_target is provided):
            The computed loss from TACTiS model forward pass.
            
        For inference mode (when future_target is None):
            Samples in the format expected by GluonTS SampleForecastGenerator:
            [batch, num_samples, prediction_length, series]
            
        Notes
        -----
        The shape transformation during inference is critical for GluonTS compatibility:
        1. TACTiS.sample returns: [num_samples, batch, series, pred_len]
        2. This method transforms it to: [batch, num_samples, pred_len, series]
        This transformation ensures compatibility with GluonTS forecast generators.
        """
        if network_input["future_target"] is not None:
            # Training mode - compute loss
            loss = self.tactis.forward(
                hist_time=network_input["hist_time"],
                hist_value=network_input["past_target"],
                pred_time=network_input["pred_time"],
                pred_value=network_input["future_target"],
            )
            return loss
        else:
            # Inference mode - sample from the distribution
            samples = self.tactis.sample(
                num_samples=self.num_parallel_samples,
                hist_time=network_input["hist_time"],
                hist_value=network_input["past_target"],
                pred_time=network_input["pred_time"],
            )
            
            # Rescale the samples if needed
            if self.scaling != "none":
                # Reshape scale and loc to [1, batch, series, 1] for broadcasting with samples [num_samples, batch, series, pred_len]
                scale_reshaped = network_input["scale"].squeeze(1).unsqueeze(0).unsqueeze(-1)
                loc_reshaped = network_input["loc"].squeeze(1).unsqueeze(0).unsqueeze(-1)
                # Perform rescaling
                samples = samples * scale_reshaped + loc_reshaped

            # Return samples in the format expected by SampleForecastGenerator: [batch, num_samples, pred_len, series]
            # Current shape: [num_samples, batch, series, pred_len]
            # Target shape:  [batch, num_samples, pred_len, series]
            # Permutation:   (1, 0, 3, 2)
            return samples.permute(1, 0, 3, 2)
            
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
        output_distr_params: Optional[dict] = {}
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
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
        future_target
            Future target values (optional, only used during training).
        num_parallel_samples
            Number of samples to generate in parallel for inference.
        output_distr_params
            Whether to output distribution parameters.
            
        Returns
        -------
        Either the loss (during training) or samples (during inference).
        """
        if num_parallel_samples is not None:
            self.num_parallel_samples = num_parallel_samples
            
        # Create the inputs for the network
        network_input = self.create_network_inputs(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        # Compute the output params (loss during training, samples during inference)
        params = self.output_params(network_input)
        
        if output_distr_params:
            return params
        
        if network_input["future_target"] is not None:
            # In training, return the loss
            return params
        else:
            # In inference, return the samples
            return params
