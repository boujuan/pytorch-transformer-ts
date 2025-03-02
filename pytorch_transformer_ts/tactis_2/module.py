from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler

from .tactis2 import TACTiS

class TACTiS2Model(nn.Module):
    """
    Module class that wraps the TACTiS model for use with the pytorch-transformer-ts framework.
    """
    
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: Optional[List[int]] = None,
        # TACTiS specific parameters
        flow_series_embedding_dim: int = 32,
        copula_series_embedding_dim: int = 32,
        flow_input_encoder_layers: int = 1,
        copula_input_encoder_layers: int = 1,
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        data_normalization: str = "none",
        loss_normalization: str = "series",
        # Other parameters
        distr_output: Optional[DistributionOutput] = None,
        input_size: int = 1,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ):
        super().__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.scaling = scaling
        self.num_parallel_samples = num_parallel_samples
        self.training_stage = 1  # Start with flow training only
        
        if distr_output is None:
            if input_size > 1:
                self.distr_output = MultivariateGaussianOutput(dim=input_size)
            else:
                self.distr_output = StudentTOutput()
        else:
            self.distr_output = distr_output
            
        self.target_shape = (input_size,) if input_size > 1 else ()
        
        # Compute lag indices
        if lags_seq is None:
            lags_seq = get_lags_for_frequency(freq_str=freq)
        self.lags_seq = lags_seq
        
        # Set up scalers
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
            
        # Set up feature embedders
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=embedding_dimension if embedding_dimension is not None else [min(50, (c + 1) // 2) for c in cardinality],
        )
        
        # Compute total number of features
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.num_feat_dynamic_real = num_feat_dynamic_real
        
        # Set up TACTiS model with specific configuration for time series forecasting
        # Using default encoder/decoder configurations adapted for the GluonTS framework
        flow_encoder_config = {
            "attention_layers": 2,
            "attention_heads": 4,
            "attention_dim": 32,
            "attention_feedforward_dim": 128,
            "dropout": 0.1,
        }
        
        copula_encoder_config = {
            "attention_layers": 2,
            "attention_heads": 4,
            "attention_dim": 32,
            "attention_feedforward_dim": 128,
            "dropout": 0.1,
        }
        
        marginal_config = {
            "mlp_layers": 2,
            "mlp_dim": 64,
            "flow_layers": 3, 
            "flow_hid_dim": 16,
        }
        
        copula_config = {
            "resolution": 32,
            "attention_layers": 2,
            "attention_heads": 4, 
            "attention_dim": 32,
            "min_u": 0.0,
            "max_u": 1.0,
        }
        
        flow_encoder_dict = {
            "attention_layers": flow_encoder_config["attention_layers"],
            "attention_heads": flow_encoder_config["attention_heads"], 
            "attention_dim": flow_encoder_config["attention_dim"],
            "attention_feedforward_dim": flow_encoder_config["attention_feedforward_dim"],
            "dropout": flow_encoder_config["dropout"]
        }
        
        copula_encoder_dict = {
            "attention_layers": copula_encoder_config["attention_layers"],
            "attention_heads": copula_encoder_config["attention_heads"],
            "attention_dim": copula_encoder_config["attention_dim"],
            "attention_feedforward_dim": copula_encoder_config["attention_feedforward_dim"],
            "dropout": copula_encoder_config["dropout"]
        }
        
        dsf_marginal_dict = {
            "mlp_layers": marginal_config["mlp_layers"],
            "mlp_dim": marginal_config["mlp_dim"],
            "flow_layers": marginal_config["flow_layers"],
            "flow_hid_dim": marginal_config["flow_hid_dim"]
        }
        
        attentional_copula_dict = {
            "resolution": copula_config["resolution"],
            "attention_layers": copula_config["attention_layers"],
            "attention_heads": copula_config["attention_heads"],
            "attention_dim": copula_config["attention_dim"],
            "min_u": copula_config["min_u"],
            "max_u": copula_config["max_u"]
        }
        
        copula_decoder_dict = {
            "dsf_marginal": dsf_marginal_dict,
            "attentional_copula": attentional_copula_dict
        }
        
        self.tactis = TACTiS(
            num_series=input_size,
            flow_series_embedding_dim=flow_series_embedding_dim,
            copula_series_embedding_dim=copula_series_embedding_dim,
            flow_input_encoder_layers=flow_input_encoder_layers,
            copula_input_encoder_layers=copula_input_encoder_layers,
            bagging_size=bagging_size,
            input_encoding_normalization=input_encoding_normalization,
            data_normalization=data_normalization,
            loss_normalization=loss_normalization,
            positional_encoding={"dropout": 0.1, "max_length": 5000},
            flow_encoder=flow_encoder_dict,
            copula_encoder=copula_encoder_dict,
            copula_decoder=copula_decoder_dict,
            skip_copula=True
        )
        
        # Output projection for compatibility with GluonTS
        hidden_dim = flow_series_embedding_dim * 4  # Use a sufficiently large hidden dim
        self.param_proj = self.distr_output.get_args_proj(hidden_dim)
    
    def create_network_inputs(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Create inputs for the TACTiS network.
        """
        # Static features
        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat([embedded_cat, feat_static_real], dim=1)
        
        # Scale if needed
        if self.scaling:
            past_target_scaled, loc, scale = self.scaler(past_target, past_observed_values)
        else:
            past_target_scaled = past_target
            loc = torch.zeros_like(past_target).mean(dim=1, keepdim=True)
            scale = torch.ones_like(past_target).mean(dim=1, keepdim=True)
        
        # Reshape to TACTiS expected format if needed
        if len(self.target_shape) == 0:
            past_target_scaled = past_target_scaled.unsqueeze(-1)
            if future_target is not None:
                future_target_scaled = (future_target - loc) / scale
                future_target_scaled = future_target_scaled.unsqueeze(-1)
        else:
            if future_target is not None:
                future_target_scaled = (future_target - loc) / scale
        
        # Prepare time features
        past_time_feat_reshaped = past_time_feat
        if future_time_feat is not None:
            future_time_feat_reshaped = future_time_feat
            
            # For TACTiS, we need time indices
            hist_time = torch.arange(past_target.shape[1], device=past_target.device).unsqueeze(0).expand(past_target.shape[0], -1)
            pred_time = torch.arange(past_target.shape[1], past_target.shape[1] + future_time_feat.shape[1], 
                                    device=past_target.device).unsqueeze(0).expand(past_target.shape[0], -1)
        else:
            future_target_scaled = None
            future_time_feat_reshaped = None
            hist_time = torch.arange(past_target.shape[1], device=past_target.device).unsqueeze(0).expand(past_target.shape[0], -1)
            pred_time = None
        
        return (hist_time, past_target_scaled.transpose(1,2), pred_time, future_target_scaled.transpose(1,2) if future_target_scaled is not None else None), loc, scale, static_feat, past_time_feat_reshaped
    
    def output_params(self, inputs, static_feat=None):
        """
        Produce parameters for the output distribution.
        """
        # For compatibility with GluonTS - in actual use we'd use the TACTiS distribution directly
        hist_time, hist_value, pred_time, pred_value = inputs
        
        # Get encoder output
        output = self.tactis.forward(hist_time, hist_value, pred_time, pred_value)[0]
        
        # Project to distribution parameters
        # Convert shape from [batch, pred_len, series] to [batch, pred_len, param_dim]
        batch_size, pred_len, num_series = output.shape
        output = output.reshape(batch_size * pred_len, num_series)
        params = self.param_proj(output)
        
        # Reshape params for distribution
        return params.reshape(batch_size, pred_len, -1)
    
    def output_distribution(self, params, loc=None, scale=None):
        """
        Create output distribution from parameters.
        """
        return self.distr_output.distribution(params, loc=loc, scale=scale)
    
    def output_loss(self, params, target, loc=None, scale=None):
        """
        Compute loss directly - alternative to distribution-based loss.
        """
        distr = self.output_distribution(params, loc=loc, scale=scale)
        return -distr.log_prob(target)
    
    def set_training_stage(self, stage):
        """
        Switch between flow-only (stage 1) and full model (stage 2) training.
        """
        self.training_stage = stage
        self.tactis.set_stage(stage)
    
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
        output_distr_params: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Forward pass for inference.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        inputs, loc, scale, static_feat, _ = self.create_network_inputs(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
        )
        
        # Output distribution parameters if requested
        if output_distr_params:
            params = self.output_params(inputs, static_feat)
            return params
        
        # Generate samples using TACTiS
        hist_time, hist_value, pred_time, _ = inputs
        samples = self.tactis.sample(
            num_samples=num_parallel_samples,
            hist_time=hist_time,
            hist_value=hist_value,
            pred_time=pred_time,
        )
        
        # Scale samples back
        # TACTiS samples are [num_samples, batch, series, pred_len]
        # We need to transpose to [batch, num_samples, pred_len, series]
        samples = samples.permute(1, 0, 3, 2)
        
        # Scale back
        scaled_samples = samples * scale.unsqueeze(1) + loc.unsqueeze(1)
        
        return scaled_samples
