"""
TACTiS2 Implementation for pytorch-transformer-ts.
Based on the original TACTiS implementation.
"""

import math
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
from torch import nn
import copy

# Constants
EPSILON = 1e-6

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        assert embedding_dim % 2 == 0, "PositionalEncoding needs an even embedding dimension"
        self.dropout = nn.Dropout(p=dropout)
        
        pos_encoding = torch.zeros(max_len, embedding_dim)
        possible_pos = torch.arange(0, max_len, dtype=torch.float)[:, None]
        factor = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_dim))
        
        # Alternate between sine and cosine
        pos_encoding[:, 0::2] = torch.sin(possible_pos * factor)
        pos_encoding[:, 1::2] = torch.cos(possible_pos * factor)
        
        self.register_buffer("pos_encoding", pos_encoding)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding based on timesteps"""
        min_t = timesteps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        delta_t = timesteps - min_t
        output = x + self.pos_encoding[delta_t]
        return self.dropout(output)

class SigmoidFlow(nn.Module):
    """
    A single layer of the Deep Sigmoid Flow network.
    """
    def __init__(self, hidden_dim: int, no_logit: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.no_logit = no_logit
    
    def forward(self, params, x, logdet):
        """Transform with derivative computation"""
        a = torch.nn.functional.softplus(params[..., :self.hidden_dim]) + EPSILON
        b = params[..., self.hidden_dim:2*self.hidden_dim]
        w = torch.nn.functional.softmax(params[..., 2*self.hidden_dim:], dim=-1)
        
        pre_sigm = a * x[..., None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = (w * sigm).sum(dim=-1)
        
        logj = (w * sigm * (1 - sigm) * a).sum(dim=-1)
        logdet = logdet + torch.log(logj)
        
        if self.no_logit:
            return x_pre, logdet
        
        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)
        
        logdet = logdet - torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)
        
        return xnew, logdet
    
    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        a = torch.nn.functional.softplus(params[..., :self.hidden_dim]) + EPSILON
        b = params[..., self.hidden_dim:2*self.hidden_dim]
        w = torch.nn.functional.softmax(params[..., 2*self.hidden_dim:], dim=-1)
        
        pre_sigm = a * x[..., None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = (w * sigm).sum(dim=-1)
        
        if self.no_logit:
            return x_pre
        
        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)
        
        return xnew

class DeepSigmoidFlow(nn.Module):
    """
    A Deep Sigmoid Flow network for flexible distribution modeling
    """
    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        self.params_length = 3 * hidden_dim
        
        layers = [SigmoidFlow(hidden_dim) for _ in range(n_layers - 1)]
        layers.append(SigmoidFlow(hidden_dim, no_logit=True))
        self.layers = nn.ModuleList(layers)
    
    @property
    def total_params_length(self):
        return len(self.layers) * self.params_length
    
    def forward(self, params, x):
        """Transform with derivative computation"""
        logdet = torch.zeros(x.shape[0], device=x.device)
        for i, layer in enumerate(self.layers):
            x, logdet = layer(
                params[..., i * self.params_length : (i + 1) * self.params_length],
                x,
                logdet,
            )
        return x, logdet
    
    def forward_no_logdet(self, params, x):
        """Transform without derivative computation"""
        for i, layer in enumerate(self.layers):
            x = layer.forward_no_logdet(
                params[..., i * self.params_length : (i + 1) * self.params_length], x
            )
        return x

class DSFMarginal(nn.Module):
    """
    Deep Sigmoid Flow for marginal distribution modeling
    """
    def __init__(
        self,
        context_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        flow_layers: int,
        flow_hid_dim: int,
    ):
        super().__init__()
        self.marginal_flow = DeepSigmoidFlow(n_layers=flow_layers, hidden_dim=flow_hid_dim)
        
        # Create conditioner network
        layers = [nn.Linear(context_dim, mlp_dim), nn.ReLU()]
        for _ in range(1, mlp_layers):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, self.marginal_flow.total_params_length)]
        self.marginal_conditioner = nn.Sequential(*layers)
    
    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform with derivative computation"""
        marginal_params = self.marginal_conditioner(context)
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]
        return self.marginal_flow.forward(marginal_params, x)
    
    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Transform without derivative computation"""
        marginal_params = self.marginal_conditioner(context)
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]
        return self.marginal_flow.forward_no_logdet(marginal_params, x)
    
    def inverse(self, context: torch.Tensor, u: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
        """Compute inverse CDF using binary search"""
        marginal_params = self.marginal_conditioner(context)
        if marginal_params.dim() == u.dim():
            marginal_params = marginal_params[:, :, None, :]
        
        left = -1000.0 * torch.ones_like(u)
        right = 1000.0 * torch.ones_like(u)
        for _ in range(max_iter):
            mid = (left + right) / 2
            error = self.marginal_flow.forward_no_logdet(marginal_params, mid) - u
            left[error <= 0] = mid[error <= 0]
            right[error >= 0] = mid[error >= 0]
            
            max_error = error.abs().max().item()
            if max_error < 1e-6:
                break
        return mid

class AttentionalCopula(nn.Module):
    """
    Attentional Copula for modeling dependencies between variables
    """
    def __init__(
        self,
        input_dim: int,
        attention_layers: int = 4,
        attention_heads: int = 4,
        attention_dim: int = 32,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        resolution: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.resolution = resolution
        
        # Dimension adjustment
        self.dimension_shifting_layer = nn.Linear(input_dim, attention_heads * attention_dim)
        
        # Create attention components
        self.attention_layer_norms = nn.ModuleList([
            nn.LayerNorm(attention_heads * attention_dim) for _ in range(attention_layers)
        ])
        self.feed_forward_layer_norms = nn.ModuleList([
            nn.LayerNorm(attention_heads * attention_dim) for _ in range(attention_layers)
        ])
        self.attention_dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(attention_layers)
        ])
        
        # Feed-forward layers
        feed_forwards = []
        for _ in range(attention_layers):
            layers = [nn.Linear(attention_heads * attention_dim, mlp_dim), nn.ReLU()]
            for _ in range(1, mlp_layers):
                layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
            layers += [nn.Linear(mlp_dim, attention_heads * attention_dim)]
            feed_forwards.append(nn.Sequential(*layers))
        self.feed_forwards = nn.ModuleList(feed_forwards)
        
        # Create key/value generators for each layer and head
        self.key_creators = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim + 1, attention_dim) for _ in range(attention_heads)
            ]) for _ in range(attention_layers)
        ])
        
        self.value_creators = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim + 1, attention_dim) for _ in range(attention_heads)
            ]) for _ in range(attention_layers)
        ])
        
        # Distribution output layer
        self.dist_extractors = nn.Linear(attention_heads * attention_dim, resolution)
    
    def sample(self, num_samples: int, flow_encoded: torch.Tensor, embedded_dims: int) -> torch.Tensor:
        """Sample from the copula"""
        batch_size, num_vars, embedding_dim = flow_encoded.shape
        device = flow_encoded.device
        
        # Prepare storage for samples
        samples = torch.zeros(batch_size, num_vars, num_samples, device=device)
        
        # Generate keys and values for historical data
        keys_hist = []
        values_hist = []
        for layer in range(self.attention_layers):
            # Process each attention head
            layer_keys = []
            layer_values = []
            for head in range(self.attention_heads):
                # Keys and values for history
                hist_keys = self.key_creators[layer][head](flow_encoded)
                hist_values = self.value_creators[layer][head](flow_encoded)
                
                layer_keys.append(hist_keys)
                layer_values.append(hist_values)
            
            # Stack head outputs
            keys_hist.append(torch.stack(layer_keys, dim=1))  # [batch, heads, vars, dim]
            values_hist.append(torch.stack(layer_values, dim=1))  # [batch, heads, vars, dim]
        
        # Sample each variable autoregressively
        p = torch.randperm(num_vars)
        for i in range(num_vars):
            var_idx = p[i]
            
            # Prepare encoded representation for current variable
            current_encoded = flow_encoded[:, var_idx:var_idx+1].expand(-1, num_samples, -1)
            
            # Generate attention representation
            att_value = self.dimension_shifting_layer(current_encoded)
            
            # Run through attention layers
            for layer in range(self.attention_layers):
                att_value_heads = att_value.reshape(
                    batch_size, num_samples, self.attention_heads, self.attention_dim
                )
                
                # Get keys and values
                keys_hist_layer = keys_hist[layer]
                values_hist_layer = values_hist[layer]
                
                # Calculate attention weights for history
                product_hist = torch.einsum("bnhi,bhvi->bnhv", att_value_heads, keys_hist_layer)
                product_hist = self.attention_dim ** (-0.5) * product_hist
                
                # Softmax over history
                weights_hist = torch.softmax(product_hist, dim=-1)
                
                # Apply attention
                att_hist = torch.einsum("bnhv,bhvj->bnhj", weights_hist, values_hist_layer)
                
                # Merge attention heads
                att_merged = att_hist.reshape(batch_size, num_samples, self.attention_heads * self.attention_dim)
                
                # Apply dropout, residual connection, and layer norm
                att_merged = self.attention_dropouts[layer](att_merged)
                att_value = att_value + att_merged
                att_value = self.attention_layer_norms[layer](att_value)
                
                # Apply feed-forward layers
                att_feed_forward = self.feed_forwards[layer](att_value)
                att_value = att_value + att_feed_forward
                att_value = self.feed_forward_layer_norms[layer](att_value)
            
            # Get distribution parameters
            logits = self.dist_extractors(att_value)
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            indices = torch.multinomial(probs.view(-1, self.resolution), 1)
            indices = indices.view(batch_size, num_samples)
            
            # Convert to uniform samples in [0,1]
            u_samples = (indices.float() + torch.rand_like(indices.float())) / self.resolution
            
            # Store samples
            samples[:, var_idx, :] = u_samples
        
        return samples

class TACTiS(nn.Module):
    """
    The top-level module for TACTiS-2.
    """
    def __init__(
        self,
        num_series: int,
        flow_series_embedding_dim: int,
        copula_series_embedding_dim: int,
        flow_input_encoder_layers: int,
        copula_input_encoder_layers: int,
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        data_normalization: str = "none",
        loss_normalization: str = "series",
        positional_encoding: Optional[Dict[str, Any]] = None,
        flow_encoder: Optional[Dict[str, Any]] = None,
        copula_encoder: Optional[Dict[str, Any]] = None,
        flow_temporal_encoder: Optional[Dict[str, Any]] = None,
        copula_temporal_encoder: Optional[Dict[str, Any]] = None,
        copula_decoder: Optional[Dict[str, Any]] = None,
        skip_copula: bool = True,
        experiment_mode: str = "forecasting",
    ):
        super().__init__()
        
        self.num_series = num_series
        self.flow_series_embedding_dim = flow_series_embedding_dim
        self.copula_series_embedding_dim = copula_series_embedding_dim
        self.flow_input_encoder_layers = flow_input_encoder_layers
        self.copula_input_encoder_layers = copula_input_encoder_layers
        self.bagging_size = bagging_size
        self.input_encoding_normalization = input_encoding_normalization
        self.data_normalization = data_normalization
        self.loss_normalization = loss_normalization
        self.skip_copula = skip_copula
        self.experiment_mode = experiment_mode
        
        # Stage tracking
        self.stage = 1
        
        # Series embeddings
        self.flow_series_encoder = nn.Embedding(num_series, flow_series_embedding_dim)
        
        # Initialize model components
        self._initialize_flow_components(flow_encoder, positional_encoding)
        
        # Marginals
        self.marginal = DSFMarginal(
            context_dim=flow_series_embedding_dim*4,
            mlp_layers=2,
            mlp_dim=128,
            flow_layers=3,
            flow_hid_dim=32
        )
        
        # Initialize copula components if not skipping
        if not skip_copula:
            self._initialize_copula_components(copula_encoder, positional_encoding, copula_decoder)
        
        # For tracking loss components
        self.marginal_logdet = None
        self.copula_loss = None
    
    def _initialize_flow_components(self, flow_encoder_args, pos_encoding_args):
        """Initialize flow-related components"""
        # Input encoder
        encoder_dim = 256  # Default dimension
        self.flow_input_encoder = nn.Sequential(
            nn.Linear(self.flow_series_embedding_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, encoder_dim)
        )
        
        # Positional encoding
        self.flow_time_encoding = PositionalEncoding(encoder_dim)
        
        # Flow encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.flow_encoder = nn.TransformerEncoder(encoder_layer, self.flow_input_encoder_layers)
    
    def _initialize_copula_components(self, copula_encoder_args, pos_encoding_args, copula_decoder_args):
        """Initialize copula-related components (called in second stage)"""
        self.copula_series_encoder = nn.Embedding(self.num_series, self.copula_series_embedding_dim)
        
        # Input encoder
        encoder_dim = 256  # Default dimension
        self.copula_input_encoder = nn.Sequential(
            nn.Linear(self.copula_series_embedding_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, encoder_dim)
        )
        
        # Positional encoding
        self.copula_time_encoding = PositionalEncoding(encoder_dim)
        
        # Copula encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.copula_encoder = nn.TransformerEncoder(encoder_layer, self.copula_input_encoder_layers)
        
        # Copula decoder
        self.copula = AttentionalCopula(
            input_dim=encoder_dim,
            attention_layers=4,
            attention_heads=4,
            attention_dim=32,
            mlp_layers=2,
            mlp_dim=128,
            resolution=128
        )
    
    def forward(
        self, 
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training
        
        Parameters:
        -----------
        hist_time: Time values for historical data
        hist_value: Values for historical data
        pred_time: Time values for prediction period
        pred_value: Target values for prediction (training only)
        
        Returns:
        --------
        Tuple of output values and loss
        """
        if pred_value is None:
            return self.sample(1, hist_time, hist_value, pred_time), None
        
        # Training forward pass
        batch_size, num_series, hist_len = hist_value.shape
        device = hist_value.device
        
        # Prepare inputs
        time_steps = torch.cat([hist_time, pred_time], dim=1)
        value = torch.cat([hist_value, pred_value], dim=2)
        mask = torch.ones_like(value, dtype=torch.bool)
        
        # Create embeddings for series
        series_indices = torch.arange(num_series, device=device)
        flow_series_emb = self.flow_series_encoder(series_indices)
        
        # Encode inputs for flow
        flow_encoded = self._encode_flow(time_steps, value, mask, series_indices)
        
        # Process with marginal model
        normalized_data = value
        u_vals, marginal_logdet = self.marginal.forward_logdet(flow_encoded, normalized_data)
        self.marginal_logdet = marginal_logdet.mean()
        
        # Process with copula if needed
        if not self.skip_copula and self.stage >= 2:
            # Encode inputs for copula
            copula_encoded = self._encode_copula(time_steps, value, mask, series_indices)
            
            # Process with copula
            self.copula_loss = -self.copula.log_prob(u_vals, copula_encoded)
        else:
            self.copula_loss = torch.zeros(1, device=device).mean()
        
        # Return output
        output = pred_value  # For training, return target as "output"
        loss = self.copula_loss - self.marginal_logdet  # Negative log-likelihood
        
        return output, loss
    
    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples from the model
        
        Parameters:
        -----------
        num_samples: Number of samples to generate
        hist_time: Time values for historical data
        hist_value: Values for historical data
        pred_time: Time values for prediction period
        
        Returns:
        --------
        Samples of shape [num_samples, batch, series, pred_len]
        """
        batch_size, num_series, hist_len = hist_value.shape
        pred_len = pred_time.shape[1]
        device = hist_value.device
        
        # Prepare inputs
        time_steps = torch.cat([hist_time, pred_time], dim=1)
        series_indices = torch.arange(num_series, device=device)
        
        # Create placeholder for full data (with zeros for pred)
        value = torch.cat([
            hist_value, 
            torch.zeros(batch_size, num_series, pred_len, device=device)
        ], dim=2)
        mask = torch.cat([
            torch.ones(batch_size, num_series, hist_len, dtype=torch.bool, device=device),
            torch.zeros(batch_size, num_series, pred_len, dtype=torch.bool, device=device)
        ], dim=2)
        
        # Encode inputs for flow
        flow_encoded = self._encode_flow(time_steps, value, mask, series_indices)
        
        # Generate samples using the copula (or uniform if skipping)
        if not self.skip_copula and self.stage >= 2:
            # Encode inputs for copula
            copula_encoded = self._encode_copula(time_steps, value, mask, series_indices)
            
            # Sample from copula
            u_samples = self.copula.sample(
                num_samples, 
                flow_encoded[:, :, hist_len:],
                flow_encoded.shape[-1]
            )
        else:
            # Use uniform samples
            u_samples = torch.rand(
                num_samples, batch_size, num_series, pred_len,
                device=device
            )
        
        # Transform samples using marginal inverse CDF
        samples = torch.zeros_like(u_samples)
        for i in range(batch_size):
            for j in range(num_series):
                samples[:, i, j] = self.marginal.inverse(
                    flow_encoded[i, j, hist_len:].unsqueeze(0).expand(num_samples, -1, -1),
                    u_samples[:, i, j]
                )
        
        return samples
    
    def _encode_flow(self, time_steps, value, mask, series_indices):
        """Encode data for flow component"""
        batch_size, num_timesteps = time_steps.shape
        num_series = series_indices.size(0)
        device = time_steps.device
        
        # Get series embeddings
        series_emb = self.flow_series_encoder(series_indices)  # [series, dim]
        
        # Prepare input tensor with values, series embedding, and mask
        flow_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),  # [batch, time, series, 1]
            series_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, num_timesteps, -1, -1),  # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1)
        
        # Reshape for input encoder
        flow_input = flow_input.reshape(batch_size * num_timesteps * num_series, -1)
        
        # Apply input encoder
        encoded = self.flow_input_encoder(flow_input)
        
        # Reshape back
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        
        # Apply positional encoding
        time_tensor = time_steps.unsqueeze(-1).expand(-1, -1, num_series).unsqueeze(-1)
        encoded = self.flow_time_encoding(encoded, time_tensor)
        
        # Apply flow encoder
        encoded = encoded.reshape(batch_size, num_timesteps * num_series, -1)
        encoded = self.flow_encoder(encoded)
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        
        # Arrange dimensions to [batch, series, time, dim]
        encoded = encoded.permute(0, 2, 1, 3)
        
        return encoded
    
    def _encode_copula(self, time_steps, value, mask, series_indices):
        """Encode data for copula component"""
        batch_size, num_timesteps = time_steps.shape
        num_series = series_indices.size(0)
        device = time_steps.device
        
        # Get series embeddings
        series_emb = self.copula_series_encoder(series_indices)  # [series, dim]
        
        # Prepare input tensor with values, series embedding, and mask
        copula_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),  # [batch, time, series, 1]
            series_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, num_timesteps, -1, -1),  # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1)
        
        # Reshape for input encoder
        copula_input = copula_input.reshape(batch_size * num_timesteps * num_series, -1)
        
        # Apply input encoder
        encoded = self.copula_input_encoder(copula_input)
        
        # Reshape back
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        
        # Apply positional encoding
        time_tensor = time_steps.unsqueeze(-1).expand(-1, -1, num_series).unsqueeze(-1)
        encoded = self.copula_time_encoding(encoded, time_tensor)
        
        # Apply copula encoder
        encoded = encoded.reshape(batch_size, num_timesteps * num_series, -1)
        encoded = self.copula_encoder(encoded)
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        
        # Arrange dimensions to [batch, series, time, dim]
        encoded = encoded.permute(0, 2, 1, 3)
        
        return encoded
    
    def set_stage(self, stage: int):
        """Set the training stage"""
        assert stage in [1, 2], "Stage must be 1 or 2"
        self.stage = stage
        
        if stage == 2 and self.skip_copula:
            self.skip_copula = False
            self._initialize_copula_components(None, None, None)
    
    def set_experiment_mode(self, experiment_mode: str):
        """Set the experiment mode (forecasting or interpolation)"""
        assert experiment_mode in ["forecasting", "interpolation"]
        self.experiment_mode = experiment_mode
