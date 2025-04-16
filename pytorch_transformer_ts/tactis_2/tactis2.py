"""
TACTiS2 Implementation for pytorch-transformer-ts.
Based on the original TACTiS implementation.
"""

import math
import logging
from typing import Optional, Dict, Any, Tuple
import torch
from torch import nn

# Constants
EPSILON = 1e-6

# Set up logging
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        assert embedding_dim % 2 == 0, "PositionalEncoding needs an even embedding dimension"
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        
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
        delta_t = torch.clamp(delta_t, 0, self.pos_encoding.shape[0] - 1)
        
        try:
            # Get the positional encodings
            pos_enc = self.pos_encoding[delta_t.long()]
            
            # Special case for handling [batch, time, series, features] tensors
            if len(x.shape) == 4 and len(pos_enc.shape) == 4:
                # Check if the specific mismatch is in the series dimension (dim 2)
                if pos_enc.shape[2] == 1 and x.shape[2] > 1:
                    # Explicitly expand the series dimension
                    pos_enc = pos_enc.expand(-1, -1, x.shape[2], -1)
                
            # General handling for any remaining shape mismatches
            if pos_enc.shape != x.shape:
                # Try to automatically adapt the shape
                try:
                    # For tensors with same number of dimensions
                    if len(pos_enc.shape) == len(x.shape):
                        # Use PyTorch's expand method which is more efficient
                        expand_shape = list(x.shape)
                        pos_enc = pos_enc.expand(*expand_shape)
                    # If pos_enc has fewer dimensions
                    elif len(pos_enc.shape) < len(x.shape):
                        # Add missing dimensions
                        for _ in range(len(x.shape) - len(pos_enc.shape)):
                            pos_enc = pos_enc.unsqueeze(-1)
                        # Then expand
                        pos_enc = pos_enc.expand(*x.shape)
                    # If pos_enc has more dimensions (unusual)
                    else:
                        # Just force reshape to match x's shape
                        pos_enc = pos_enc.view(*x.shape)
                except Exception as inner_e:
                    logger.warning(f"Error during positional encoding shape adjustment: {inner_e}")
                    return self.dropout(x)  # Skip positional encoding rather than crash
            
            # Add the position encoding to the input
            output = x + pos_enc
            
        except Exception as e:
            logger.error(f"Error in positional encoding: {e}")
            # Fallback - just return the input unchanged
            output = x
            
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
        # Apply softplus with numerical stability
        a = torch.nn.functional.softplus(params[..., :self.hidden_dim]) + EPSILON
        b = params[..., self.hidden_dim:2*self.hidden_dim]
        w = torch.nn.functional.softmax(params[..., 2*self.hidden_dim:], dim=-1)
        
        # Compute pre-sigmoid with better numerical stability
        # Clamp x to avoid extreme values that could cause instability
        x_clamped = torch.clamp(x, min=-100.0, max=100.0)
        pre_sigm = a * x_clamped[..., None] + b
        
        # Apply sigmoid with numerical stability
        # Use torch.sigmoid which is more numerically stable than manual computation
        sigm = torch.sigmoid(pre_sigm)
        
        # Add small epsilon to prevent zeros
        sigm = torch.clamp(sigm, min=EPSILON, max=1.0-EPSILON)
        
        # Compute weighted sum
        x_pre = (w * sigm).sum(dim=-1)
        
        # Compute log jacobian with numerical stability
        # (1-sigm) can be close to 0 for large pre_sigm values
        logj = (w * sigm * (1 - sigm) * a).sum(dim=-1)
        # Add epsilon to prevent log of zero
        logj = torch.clamp(logj, min=EPSILON)
        
        # Fix for dimension mismatch: expand logdet to match logj's dimensions
        if logdet.shape != logj.shape:
            # Expand logdet to match logj's dimensions
            for _ in range(len(logj.shape) - len(logdet.shape)):
                logdet = logdet.unsqueeze(-1)
            # Now broadcast to the full shape
            logdet = logdet.expand_as(logj)
            
        # Add log jacobian with checks for NaN
        log_logj = torch.log(logj)
        if torch.isnan(log_logj).any():
            logger.warning("NaN detected in log_logj, replacing with log(EPSILON)")
            # Replace NaNs with a large negative number (log of EPSILON)
            log_logj = torch.nan_to_num(log_logj, nan=math.log(EPSILON))
        
        logdet = logdet + log_logj
        
        # Check for NaNs in logdet
        if torch.isnan(logdet).any():
            logger.warning("NaN detected in logdet after addition")
            # Replace NaNs with zeros to allow training to continue
            logdet = torch.nan_to_num(logdet, nan=0.0)
        
        if self.no_logit:
            return x_pre, logdet
        
        # Apply logit transform with numerical stability
        x_pre_clipped = torch.clamp(x_pre, min=EPSILON, max=1.0-EPSILON)
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)
        
        # Update logdet to account for logit transform
        logdet = logdet - torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)
        
        # Final NaN check
        if torch.isnan(xnew).any() or torch.isnan(logdet).any():
            logger.warning(f"NaNs in final SigmoidFlow output, replacing with zeros")
            xnew = torch.nan_to_num(xnew)
            logdet = torch.nan_to_num(logdet)
        
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
        
        # Initialize logdet with the correct dimensions
        # We need to match x's full shape for proper accumulation of log determinant
        # This would be [batch_size, num_series, time_steps] for our model
        if x.dim() > 2:
            # For multi-dimensional tensors like [batch, series, timesteps]
            batch_shape = x.shape[:-1]  # All but the last dimension
        else:
            # For simpler cases
            batch_shape = x.shape
            logdet = torch.zeros(batch_shape, device=x.device)
        
        # Track original x for debugging
        x_orig = x.clone()
        
        for i, layer in enumerate(self.layers):
            # Extract parameters for this layer
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]
            
            # Check for NaNs in inputs
            if torch.isnan(x).any():
                logger.warning(f"NaN in x before DeepSigmoidFlow layer {i}")
                # Replace NaNs with zeros to allow continued processing
                x = torch.nan_to_num(x)
                
            if torch.isnan(logdet).any():
                logger.warning(f"NaN in logdet before DeepSigmoidFlow layer {i}")
                logdet = torch.nan_to_num(logdet)
                
            # Process through the layer
            x, logdet = layer(
                layer_params,
                x,
                logdet,
            )
            
        # Final check for NaNs
        if torch.isnan(x).any() or torch.isnan(logdet).any():
            logger.warning("NaNs in final output of DeepSigmoidFlow, replacing with zeros")
            # Replace NaNs as a last resort
            x = torch.nan_to_num(x)
            logdet = torch.nan_to_num(logdet)
            
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
        
        # Add a dimension reduction layer if the expected context_dim doesn't match what we'll get from the encoder
        self.input_dim = None  # Will be set on first call if needed
        self.dim_reducer = None  # Will be initialized on first call if needed
        
        # Create conditioner network
        layers = [nn.Linear(context_dim, mlp_dim), nn.ReLU()]
        for _ in range(1, mlp_layers):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        layers += [nn.Linear(mlp_dim, self.marginal_flow.total_params_length)]
        self.marginal_conditioner = nn.Sequential(*layers)
        
        # Store the expected context_dim for later checks
        self.expected_context_dim = context_dim
    
    def _ensure_dimensions_match(self, input_tensor):
        """Ensure input dimensions match what's expected by adding a dimension reducer if needed"""
        # Get the dimension of the input tensor
        input_dim = input_tensor.size(-1)
        
        # If dimensions don't match and we haven't initialized a reducer yet
        if input_dim != self.expected_context_dim and self.dim_reducer is None:
            logger.debug(f"Creating dimension reducer from {input_dim} to {self.expected_context_dim}")
            self.input_dim = input_dim
            self.dim_reducer = nn.Linear(input_dim, self.expected_context_dim)
            # Move to the same device as the input tensor
            self.dim_reducer = self.dim_reducer.type_as(input_tensor) # CHANGE to(input_tensor.device)
        
        # Apply dimension reduction if needed
        if self.dim_reducer is not None:
            return self.dim_reducer(input_tensor)
        else:
            return input_tensor
    
    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform with derivative computation"""
        # Check for NaNs in input
        if torch.isnan(x).any():
            logger.warning("NaN values detected in input x, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
            if torch.isnan(context).any():
                logger.warning("NaN values detected in context tensor, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)
        # Clamp x to a valid range to prevent numerical issues
        # The flow expects values that can be reasonably transformed without producing extremes
        x_min, x_max = x.min().item(), x.max().item()
        
        # Only clamp if values are extreme
        if x_min < -100 or x_max > 100:
            logger.info("Clamping extreme values in x")
            x = torch.clamp(x, min=-100.0, max=100.0)
        
        # Reshape context for linear layer
        # context comes in as [batch, series, time, dim] but the linear layer expects [batch, dim]
        # or [N, dim] where N = batch * series * time
        batch_size, num_series, num_timesteps, feature_dim = context.shape
        
        # Reshape to [batch*series*time, feature_dim]
        reshaped_context = context.reshape(-1, feature_dim)
        
        # Ensure dimensions match before applying conditioner
        reshaped_context = self._ensure_dimensions_match(reshaped_context)
        
        # Check for NaNs after reshaping
        if torch.isnan(reshaped_context).any():
            logger.warning("NaN values detected after reshaping context, replacing with zeros")
            reshaped_context = torch.nan_to_num(reshaped_context, nan=0.0)
        
        # Apply conditioner
        try:
            marginal_params = self.marginal_conditioner(reshaped_context)
            
            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                logger.warning("NaN values in marginal parameters, replacing with zeros")
                marginal_params = torch.nan_to_num(marginal_params, nan=0.0)
                
            # Reshape back to match x
            marginal_params = marginal_params.reshape(batch_size, num_series, num_timesteps, -1)
            
            # Make sure dimensions match
            if marginal_params.dim() == x.dim():
                # Check if reshaping is needed
                if marginal_params.shape != x.shape:
                    marginal_params = marginal_params[:, :, :, :x.size(-1)]
            
            # Forward through the flow
            transformed_x, logdet = self.marginal_flow.forward(marginal_params, x)
            
            # Check for NaNs in the output
            if torch.isnan(transformed_x).any() or torch.isnan(logdet).any():
                logger.warning("NaN values in DSFMarginal output, replacing with zeros")
                transformed_x = torch.nan_to_num(transformed_x, nan=0.0)
                logdet = torch.nan_to_num(logdet, nan=0.0)
            
            return transformed_x, logdet
            
        except Exception as e:
            logger.error(f"Error in DSFMarginal forward_logdet: {e}")
            # Return zeros as a fallback to avoid crashing
            zeros_x = torch.zeros_like(x)
            zeros_logdet = torch.zeros(batch_size, num_series, num_timesteps, device=x.device)
            return zeros_x, zeros_logdet
    
    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Transform without derivative computation"""
        # Check for NaNs in input
        if torch.isnan(x).any():
            logger.warning("NaN values detected in input x for forward_no_logdet, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
            
        if torch.isnan(context).any():
            logger.warning("NaN values detected in context tensor for forward_no_logdet, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)
        
        # Clamp x to a valid range to prevent numerical issues
        x_min, x_max = x.min().item(), x.max().item()
        if x_min < -100 or x_max > 100:
            logger.info("Clamping extreme values in x for forward_no_logdet")
            x = torch.clamp(x, min=-100.0, max=100.0)
        
        # Reshape context for linear layer
        batch_size, num_series, num_timesteps, feature_dim = context.shape
        
        # Reshape to [batch*series*time, feature_dim]
        reshaped_context = context.reshape(-1, feature_dim)
        
        # Ensure dimensions match before applying conditioner
        reshaped_context = self._ensure_dimensions_match(reshaped_context)
        
        # Check for NaNs after reshaping
        if torch.isnan(reshaped_context).any():
            logger.warning("NaN values after reshaping context in forward_no_logdet, replacing with zeros")
            reshaped_context = torch.nan_to_num(reshaped_context, nan=0.0)
        
        try:
            # Apply conditioner
            marginal_params = self.marginal_conditioner(reshaped_context)
            
            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                logger.warning("NaN values in marginal parameters for forward_no_logdet, replacing with zeros")
                marginal_params = torch.nan_to_num(marginal_params, nan=0.0)
            
            # Reshape back to match x
            marginal_params = marginal_params.reshape(batch_size, num_series, num_timesteps, -1)
            
            # Make sure dimensions match
            if marginal_params.dim() == x.dim():
                # Check if reshaping is needed
                if marginal_params.shape != x.shape:
                    marginal_params = marginal_params[:, :, :, :x.size(-1)]
            
            # Forward through the flow
            transformed_x = self.marginal_flow.forward_no_logdet(marginal_params, x)
            
            # Check for NaNs in the output
            if torch.isnan(transformed_x).any():
                logger.warning("NaN values in DSFMarginal forward_no_logdet output, replacing with zeros")
                transformed_x = torch.nan_to_num(transformed_x, nan=0.0)
            
            return transformed_x
            
        except Exception as e:
            logger.error(f"Error in DSFMarginal forward_no_logdet: {e}")
            # Return zeros as a fallback to avoid crashing
            return torch.zeros_like(x)
    
    def inverse(self, context: torch.Tensor, u: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
        """Compute inverse CDF using binary search"""
        # Reshape context for linear layer
        if context.dim() == 4:  # [batch, series, time, dim]
            batch_size, num_series, num_timesteps, feature_dim = context.shape
            
            # Reshape to [batch*series*time, feature_dim]
            reshaped_context = context.reshape(-1, feature_dim)
            
            # Ensure dimensions match before applying conditioner
            reshaped_context = self._ensure_dimensions_match(reshaped_context)
            
            # Apply conditioner
            marginal_params = self.marginal_conditioner(reshaped_context)
            
            # Reshape back
            marginal_params = marginal_params.reshape(batch_size, num_series, num_timesteps, -1)
            
            # Make sure dimensions match
            if marginal_params.dim() == u.dim():
                # Check if reshaping is needed
                if marginal_params.shape != u.shape:
                    marginal_params = marginal_params[:, :, :, :u.size(-1)]
        else:
            # Fall back to original method for other cases
            reshaped_context = self._ensure_dimensions_match(context)
            marginal_params = self.marginal_conditioner(reshaped_context)
            
            # Make sure dimensions match
            if marginal_params.dim() == u.dim():
                if marginal_params.shape != u.shape:
                    # If the parameter is missing a dimension that u has
                    if marginal_params.dim() < u.dim():
                        for _ in range(u.dim() - marginal_params.dim()):
                            marginal_params = marginal_params.unsqueeze(-2)
                    # Make sure the last dimension matches
                    marginal_params = marginal_params[..., :u.size(-1)]
        
        left = -1000.0 * torch.ones_like(u)
        right = 1000.0 * torch.ones_like(u)
        
        for _ in range(max_iter):
            mid = (left + right) / 2
            
            # Evaluate CDF at midpoint
            cdf_mid = self.marginal_flow.forward_no_logdet(marginal_params, mid)
            
            # Update bounds
            left = torch.where(cdf_mid < u, mid, left)
            right = torch.where(cdf_mid >= u, mid, right)
            
        return (left + right) / 2

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
        dropout: float = 0.1,
        attention_mlp_class: str = "_easy_mlp",
        activation_function: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.resolution = resolution
        self.dropout = dropout
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
            nn.Dropout(self.dropout) for _ in range(attention_layers)
        ])
        
        # Feed-forward layers
        feed_forwards = []
        for _ in range(attention_layers):
            layers = [nn.Linear(attention_heads * attention_dim, mlp_dim), nn.ReLU()]
            for _ in range(1, mlp_layers):
                layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
            layers += [nn.Linear(mlp_dim, attention_heads * attention_dim), nn.Dropout(self.dropout)]
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


class TemporalEncoder(nn.Module):
    """
    The encoder for TACTiS, based on the Temporal Transformer architecture.
    This encoder alternate between doing self-attention between different series of the same time steps,
    and doing self-attention between different time steps of the same series.
    This greatly reduces the memory footprint compared to TACTiSEncoder.

    The encoder receives an input which contains for each variable and time step:
    * The series value at the time step, masked to zero if part of the values to be forecasted
    * The mask
    * The embedding for the series
    * The embedding for the time step
    And has already been through any input encoder.

    The decoder returns an output containing an embedding for each series and time step.
    """

    def __init__(
        self,
        d_model: int, # Renamed from attention_dim * attention_heads
        nhead: int, # Renamed from attention_heads
        num_encoder_layers: int, # Renamed from attention_layers (represents pairs)
        dim_feedforward: int, # Renamed from attention_feedforward_dim
        dropout: float = 0.1,
        # attention_dim: int, # Removed, use d_model / nhead if needed internally
    ):
        """
        Parameters:
        -----------
        d_model: int
            The total dimension of the model.
        nhead: int
            How many independant heads the attention layer will have.
        num_encoder_layers: int
            How many successive attention pairs of layers this will use.
            Note that the total number of layers is going to be the double of this number.
            Each pair will consist of a layer with attention done over time steps,
            followed by a layer with attention done over series.
        dim_feedforward: int
            The dimension of the hidden layer in the feed forward step.
        dropout: float, default to 0.1
            Dropout parameter for the attention.
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.total_attention_time = 0.0 # Keep for potential timing analysis

        # Create pairs of layers: one for time attention, one for series attention
        self.layer_timesteps = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    batch_first=True # Important: Assume batch_first for easier handling
                )
                for _ in range(self.num_encoder_layers)
            ]
        )

        self.layer_series = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    batch_first=True # Important: Assume batch_first for easier handling
                )
                for _ in range(self.num_encoder_layers)
            ]
        )

    @property
    def embedding_dim(self) -> int:
        """
        Returns:
        --------
        dim: int
            The expected dimensionality of the input embedding, and the dimensionality of the output embedding
        """
        return self.d_model

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Compute the embedding for each series and time step.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, input embedding dimension]
            A tensor containing an embedding for each series and time step.
            This embedding is expected to only contain local information, with no interaction between series or time steps.

        Returns:
        --------
        output: torch.Tensor [batch, series, time steps, output embedding dimension]
            The transformed embedding for each series and time step.
        """
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_timesteps = encoded.shape[2]

        data = encoded

        # attention_start_time = time.time() # Removed timing for clarity
        for i in range(self.num_encoder_layers):
            # --- Time Attention ---
            # Treat the various series as a batch dimension
            mod_timesteps = self.layer_timesteps[i]
            # [batch * series, time steps, embedding]
            data_time = data.reshape(num_batches * num_series, num_timesteps, self.embedding_dim)
            # Perform attention (batch_first=True)
            data_time = mod_timesteps(data_time)
            # [batch, series, time steps, embedding]
            data = data_time.reshape(num_batches, num_series, num_timesteps, self.embedding_dim)

            # --- Series Attention ---
            # Treat the various time steps as a batch dimension
            mod_series = self.layer_series[i]
            # Transpose to [batch, timesteps, series, embedding]
            data_series = data.transpose(1, 2)
            # [batch * time steps, series, embedding]
            data_series = data_series.reshape(num_batches * num_timesteps, num_series, self.embedding_dim)
            # Perform attention (batch_first=True)
            data_series = mod_series(data_series)
            # [batch, time steps, series, embedding]
            data_series = data_series.reshape(num_batches, num_timesteps, num_series, self.embedding_dim)
            # Transpose back to [batch, series, time steps, embedding]
            data = data_series.transpose(1, 2)

        # attention_end_time = time.time()
        # self.total_attention_time = attention_end_time - attention_start_time
        # The resulting tensor may not be contiguous, which can cause problems further down the line.
        output = data.contiguous()

        return output

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
        loss_normalization: str = "series",
        positional_encoding: Optional[Dict[str, Any]] = None,
        flow_encoder: Optional[Dict[str, Any]] = None,
        copula_encoder: Optional[Dict[str, Any]] = None,
        flow_temporal_encoder: Optional[Dict[str, Any]] = None,
        copula_temporal_encoder: Optional[Dict[str, Any]] = None,
        copula_decoder: Optional[Dict[str, Any]] = None,
        skip_copula: bool = True,
        encoder_type: str = "standard",
    ):
        """
        Initialize TACTiS model components.
        
        Parameters
        ----------
        num_series: int
            Number of time series to model
        flow_series_embedding_dim: int
            Dimension of series embedding for flow component
        copula_series_embedding_dim: int
            Dimension of series embedding for copula component
        flow_input_encoder_layers: int
            Number of layers in flow input encoder
        copula_input_encoder_layers: int
            Number of layers in copula input encoder
        bagging_size: Optional[int]
            Size of bagging ensemble (None for no bagging)
        input_encoding_normalization: bool
            Whether to normalize input encodings
        loss_normalization: str
            Type of normalization to apply to loss
        positional_encoding: Optional[Dict]
            Arguments for positional encoding
        flow_encoder: Optional[Dict]
            Arguments for flow encoder
        copula_encoder: Optional[Dict]
            Arguments for copula encoder
        flow_temporal_encoder: Optional[Dict]
            Arguments for flow temporal encoder
        copula_temporal_encoder: Optional[Dict]
            Arguments for copula temporal encoder
        copula_decoder: Optional[Dict]
            Arguments for copula decoder
        skip_copula: bool
            Whether to skip copula component (use only flow for first stage)
        encoder_type: str
            Type of encoder to use ("standard" or "temporal").
        """
        super().__init__()
        
        # Store parameters
        self.num_series = num_series
        self.flow_series_embedding_dim = flow_series_embedding_dim
        self.copula_series_embedding_dim = copula_series_embedding_dim
        self.flow_input_encoder_layers = flow_input_encoder_layers
        self.copula_input_encoder_layers = copula_input_encoder_layers
        self.input_encoding_normalization = input_encoding_normalization
        self.loss_normalization = loss_normalization
        self.skip_copula = skip_copula
        self.bagging_size = bagging_size
        self.encoder_type = encoder_type
        
        # Stage tracks whether to use flow only (1) or flow+copula (2)
        self.stage = 1
        self.marginal_logdet = None
        self.copula_loss = None
        
        # Create embeddings for series
        self.flow_series_encoder = nn.Embedding(num_series, flow_series_embedding_dim)
        
        # Store config dictionaries
        self.positional_encoding_args = positional_encoding
        self.flow_encoder_args = flow_encoder
        self.copula_encoder_args = copula_encoder
        self.flow_temporal_encoder_args = flow_temporal_encoder
        self.copula_temporal_encoder_args = copula_temporal_encoder
        self.copula_decoder_args = copula_decoder

        # Initialize model components using stored args
        self._initialize_flow_components()

        # Initialize copula components if not skipping initially and args are provided
        if not skip_copula and self.copula_decoder_args is not None:
             self._initialize_copula_components()
        elif not skip_copula and self.copula_decoder_args is None:
             logger.warning("skip_copula is False, but copula_decoder args were not provided. Copula components not initialized.")
    
    def _initialize_flow_components(self):
        """Initialize flow-related components using stored args"""
        # Input encoder - Use flow_input_encoder_layers (passed directly)
        # The input dim is value (1) + mask (1) + series_embedding
        # The output dim should match the flow_encoder's d_model
        flow_d_model = self.flow_encoder_args["d_model"]
        flow_input_dim = self.flow_series_embedding_dim + 2
        flow_encoder_layers = [nn.Linear(flow_input_dim, flow_d_model), nn.ReLU()] # First layer
        for _ in range(1, self.flow_input_encoder_layers):
             # Intermediate layers could have different dims, but let's keep it simple
             flow_encoder_layers += [nn.Linear(flow_d_model, flow_d_model), nn.ReLU()]
        # Last layer ensures output matches d_model without ReLU
        if self.flow_input_encoder_layers > 1:
             flow_encoder_layers[-2] = nn.Linear(flow_d_model, flow_d_model) # Replace last Linear
             flow_encoder_layers.pop() # Remove last ReLU

        self.flow_input_encoder = nn.Sequential(*flow_encoder_layers)
        logger.debug(f"Initialized flow_input_encoder with {self.flow_input_encoder_layers} layers, input_dim={flow_input_dim}, output_dim={flow_d_model}")
        
        # Positional encoding - Use dimension from args
        pos_encoding_dim = self.positional_encoding_args["embedding_dim"]
        self.flow_time_encoding = PositionalEncoding(
             embedding_dim=pos_encoding_dim,
             dropout=self.positional_encoding_args.get("dropout", 0.1),
             max_len=self.positional_encoding_args.get("max_len", 5000)
        )
        logger.debug(f"Initialized flow_time_encoding with embedding_dim={pos_encoding_dim}")
        
        # Dimension adjustment layer if positional encoding dim doesn't match flow d_model
        if pos_encoding_dim != flow_d_model:
             self.flow_pos_adjust = nn.Linear(pos_encoding_dim, flow_d_model)
             logger.debug(f"Added flow_pos_adjust layer: {pos_encoding_dim} -> {flow_d_model}")
        else:
             self.flow_pos_adjust = nn.Identity()
        
        # Flow encoder (Transformer) - Use parameters from flow_encoder_args
        # Ensure d_model matches the output of input_encoder + pos_encoding
        if self.flow_encoder_args["d_model"] != flow_d_model:
              logger.warning(f"flow_encoder_args d_model ({self.flow_encoder_args['d_model']}) differs from calculated flow_d_model ({flow_d_model}). Using calculated value.")
        
        if self.encoder_type == "standard":
            logger.debug(f"Initializing standard flow_encoder (Transformer) with {self.flow_encoder_args['num_encoder_layers']} layers, d_model={flow_d_model}, nhead={self.flow_encoder_args['nhead']}")
            self.flow_encoder = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=flow_d_model, # Use consistent d_model
                nhead=self.flow_encoder_args["nhead"],
                dim_feedforward=self.flow_encoder_args.get("dim_feedforward", flow_d_model * 4), # Default if missing
                dropout=self.flow_encoder_args.get("dropout", 0.1),
                batch_first=True
            ),
                num_layers=self.flow_encoder_args["num_encoder_layers"]
            )
        elif self.encoder_type == "temporal":
            logger.debug(f"Initializing temporal flow_encoder with {self.flow_encoder_args['num_encoder_layers']} layers, d_model={flow_d_model}, nhead={self.flow_encoder_args['nhead']}")
            self.flow_encoder = TemporalEncoder(
                d_model=flow_d_model,
                nhead=self.flow_encoder_args["nhead"],
                num_encoder_layers=self.flow_encoder_args["num_encoder_layers"],
                dim_feedforward=self.flow_encoder_args.get("dim_feedforward", flow_d_model * 4),
                dropout=self.flow_encoder_args.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown encoder_type for flow: {self.encoder_type}")
        
        # Marginals (DSF) - Use parameters from copula_decoder_args['dsf_marginal']
        dsf_args = self.copula_decoder_args["dsf_marginal"]
        # Ensure context_dim matches the output of the flow_encoder
        if dsf_args["context_dim"] != flow_d_model:
              logger.warning(f"dsf_marginal context_dim ({dsf_args['context_dim']}) differs from flow_encoder output dim ({flow_d_model}). Using encoder output dim.")
              dsf_args["context_dim"] = flow_d_model

        self.marginal = DSFMarginal(
            context_dim=dsf_args["context_dim"],
            mlp_layers=dsf_args["mlp_layers"],
            mlp_dim=dsf_args["mlp_dim"],
            flow_layers=dsf_args["flow_layers"],
            flow_hid_dim=dsf_args["flow_hid_dim"]
        )
        logger.debug(f"Initialized marginal (DSF) with context_dim={dsf_args['context_dim']}")
    
    def _initialize_copula_components(self):
        """Initialize copula-related components using stored args"""
        # Series encoder
        self.copula_series_encoder = nn.Embedding(self.num_series, self.copula_series_embedding_dim)
        
        # Input encoder - Use copula_input_encoder_layers (passed directly)
        copula_d_model = self.copula_encoder_args["d_model"]
        copula_input_dim = self.copula_series_embedding_dim + 2
        copula_encoder_layers_list = [nn.Linear(copula_input_dim, copula_d_model), nn.ReLU()] # First layer
        for _ in range(1, self.copula_input_encoder_layers):
             copula_encoder_layers_list += [nn.Linear(copula_d_model, copula_d_model), nn.ReLU()]
        # Last layer ensures output matches d_model without ReLU
        if self.copula_input_encoder_layers > 1:
             copula_encoder_layers_list[-2] = nn.Linear(copula_d_model, copula_d_model) # Replace last Linear
             copula_encoder_layers_list.pop() # Remove last ReLU

        self.copula_input_encoder = nn.Sequential(*copula_encoder_layers_list)
        logger.debug(f"Initialized copula_input_encoder with {self.copula_input_encoder_layers} layers, input_dim={copula_input_dim}, output_dim={copula_d_model}")
        
        # Positional encoding - Use dimension from args
        pos_encoding_dim = self.positional_encoding_args["embedding_dim"]
        self.copula_time_encoding = PositionalEncoding(
             embedding_dim=pos_encoding_dim,
             dropout=self.positional_encoding_args.get("dropout", 0.1),
             max_len=self.positional_encoding_args.get("max_len", 5000)
        )
        logger.debug(f"Initialized copula_time_encoding with embedding_dim={pos_encoding_dim}")
        
        # Dimension adjustment layer if positional encoding dim doesn't match copula d_model
        if pos_encoding_dim != copula_d_model:
             self.copula_pos_adjust = nn.Linear(pos_encoding_dim, copula_d_model)
             logger.debug(f"Added copula_pos_adjust layer: {pos_encoding_dim} -> {copula_d_model}")
        else:
             self.copula_pos_adjust = nn.Identity()
        
        # Copula encoder (Transformer) - Use parameters from copula_encoder_args
        if self.copula_encoder_args["d_model"] != copula_d_model:
              logger.warning(f"copula_encoder_args d_model ({self.copula_encoder_args['d_model']}) differs from calculated copula_d_model ({copula_d_model}). Using calculated value.")
        
        if self.encoder_type == "standard":
            logger.debug(f"Initializing standard copula_encoder (Transformer) with {self.copula_encoder_args['num_encoder_layers']} layers, d_model={copula_d_model}, nhead={self.copula_encoder_args['nhead']}")
            self.copula_encoder = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=copula_d_model, # Use consistent d_model
                nhead=self.copula_encoder_args["nhead"],
                dim_feedforward=self.copula_encoder_args.get("dim_feedforward", copula_d_model * 4),
                dropout=self.copula_encoder_args.get("dropout", 0.1),
                batch_first=True
            ),
                num_layers=self.copula_encoder_args["num_encoder_layers"]
            )
        elif self.encoder_type == "temporal":
            logger.debug(f"Initializing temporal copula_encoder with {self.copula_encoder_args['num_encoder_layers']} layers, d_model={copula_d_model}, nhead={self.copula_encoder_args['nhead']}")
            self.copula_encoder = TemporalEncoder(
                d_model=copula_d_model,
                nhead=self.copula_encoder_args["nhead"],
                num_encoder_layers=self.copula_encoder_args["num_encoder_layers"],
                dim_feedforward=self.copula_encoder_args.get("dim_feedforward", copula_d_model * 4),
                dropout=self.copula_encoder_args.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown encoder_type for copula: {self.encoder_type}")
            
        # Attentional copula - Use parameters from copula_decoder_args['attentional_copula']
        if self.copula_decoder_args is None or "attentional_copula" not in self.copula_decoder_args:
             raise ValueError("Attentional Copula arguments ('attentional_copula') missing in copula_decoder config.")
        cop_args = self.copula_decoder_args["attentional_copula"]

        # Ensure input_dim matches the output of the copula_encoder
        if cop_args.get("input_dim", copula_d_model) != copula_d_model: # Use get with default
              logger.warning(f"attentional_copula input_dim ({cop_args['input_dim']}) differs from copula_encoder output dim ({copula_d_model}). Using encoder output dim.")
              cop_args["input_dim"] = copula_d_model

        self.copula = AttentionalCopula(
            input_dim=cop_args["input_dim"],
            attention_layers=cop_args["attention_layers"],
            attention_heads=cop_args["attention_heads"],
            attention_dim=cop_args["attention_dim"], # This is dim_per_head
            mlp_layers=cop_args["mlp_layers"],
            mlp_dim=cop_args["mlp_dim"],
            resolution=cop_args["resolution"],
            dropout=cop_args.get("dropout", 0.1),
            attention_mlp_class=cop_args.get("attention_mlp_class", "_easy_mlp"),
            activation_function=cop_args.get("activation_function", "relu")
        )
        logger.debug(f"Initialized AttentionalCopula with input_dim={cop_args['input_dim']}, resolution={cop_args['resolution']}")

    @staticmethod
    def _apply_bagging(
        bagging_size,
        time_steps: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        true_value: torch.Tensor,
        flow_series_emb=None,
        copula_series_emb=None,
    ):
        """
        Only keep a small number of series for each of the input tensors.
        Which series will be kept is randomly selected for each batch. The order is not preserved.

        Parameters:
        -----------
        bagging_size: int
            How many series to keep
        time_steps: Tensor [batch, time steps]
            Combined historical and prediction time steps.
        value: Tensor [batch, series, time steps]
            Combined historical and prediction values (normalized).
        mask: Tensor [batch, series, time steps]
            Boolean mask indicating observed (True) vs predicted (False).
        true_value: Tensor [batch, series, time steps]
            Same as value, used for loss calculation.
        flow_series_emb: Tensor [batch, series, flow embedding size]
            An embedding for each series for the marginals, expanded over the batches.
        copula_series_emb: Tensor [batch, series, copula embedding size]
            An embedding for each series for the copula, expanded over the batches.

        Returns:
        --------
        Tuple containing bagged versions of:
        time_steps, value, mask, true_value, flow_series_emb, copula_series_emb
        """
        num_batches = value.shape[0]
        num_series = value.shape[1]

        # Make sure to have the exact same bag for all series within a batch
        bags = [torch.randperm(num_series, device=value.device)[0:bagging_size] for _ in range(num_batches)]

        # Bag the tensors that have a series dimension
        value = torch.stack([value[i, bags[i], :] for i in range(num_batches)], dim=0)
        mask = torch.stack([mask[i, bags[i], :] for i in range(num_batches)], dim=0)
        true_value = torch.stack([true_value[i, bags[i], :] for i in range(num_batches)], dim=0)
        flow_series_emb = torch.stack([flow_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)
        if copula_series_emb is not None:
            copula_series_emb = torch.stack([copula_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)

        # time_steps usually doesn't have a series dimension, so it's returned as is
        return (
            time_steps,
            value,
            mask,
            true_value,
            flow_series_emb,
            copula_series_emb,
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
             # Inference mode: call sample method
             # Assumes hist_value is already normalized externally
             # Denormalization is handled by TACTiS2Model wrapper
             norm_samples = self.sample(1, hist_time, hist_value, pred_time)
             return norm_samples, None # Loss is None during inference

        # --- Training forward pass ---
        batch_size, num_series, hist_len = hist_value.shape
        pred_len = pred_value.shape[2]
        device = hist_value.device

        # Prepare inputs (assuming external normalization)
        time_steps = torch.cat([hist_time, pred_time], dim=1)
        value = torch.cat([hist_value, pred_value], dim=2) # Combined history + target
        mask = torch.cat([ # Mask True = observed, False = predicted
            torch.ones(batch_size, num_series, hist_len, dtype=torch.bool, device=device),
            torch.zeros(batch_size, num_series, pred_len, dtype=torch.bool, device=device)
        ], dim=2)
        true_value = value # Use combined value for encoding & loss calculation
        
        # Create embeddings for series
        series_indices = torch.arange(num_series, device=device)
        # Expand over batches for potential bagging
        flow_series_emb = self.flow_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1)
        copula_series_emb = None
        if not self.skip_copula and self.stage >= 2:
             # Ensure copula encoder exists before creating embedding
             if hasattr(self, 'copula_series_encoder'):
                 copula_series_emb = self.copula_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1)
             else:
                 logger.warning("Attempted to use copula embedding in forward pass, but copula_series_encoder not initialized.")

        # Apply bagging if configured and in training mode
        if self.bagging_size is not None and pred_value is not None:
             logger.debug(f"Applying bagging with size {self.bagging_size}")
             (
                 time_steps, # time_steps is not bagged as it has no series dim
                 value,
                 mask,
                 true_value,
                 flow_series_emb,
                 copula_series_emb, # Will be None if not initialized
             ) = self._apply_bagging(
                 self.bagging_size,
                 time_steps,
                 value,
                 mask,
                 true_value,
                 flow_series_emb=flow_series_emb,
                 copula_series_emb=copula_series_emb,
             )
             # Update num_series after bagging
             num_series = self.bagging_size
             # Update series_indices for encoding after bagging
             # Note: The actual series identity is lost after bagging,
             # but we need indices matching the new bagged dimension for encoding.
             series_indices = torch.arange(num_series, device=device)


        # Encode inputs for flow using potentially bagged tensors
        # Pass the potentially bagged flow_series_emb directly to _encode_flow
        flow_encoded = self._encode_flow(time_steps, value, mask, series_indices, flow_series_emb)
        
        # Process with marginal model
        normalized_data = value
        u_vals, marginal_logdet = self.marginal.forward_logdet(flow_encoded, normalized_data)
        self.marginal_logdet = marginal_logdet.mean()
        
        # Process with copula if needed
        if not self.skip_copula and self.stage >= 2:
            # Encode inputs for copula using potentially bagged tensors
            # Pass the potentially bagged copula_series_emb directly
            if copula_series_emb is not None: # Check if copula embeddings exist (i.e., stage 2 and initialized)
                 copula_encoded = self._encode_copula(time_steps, value, mask, series_indices, copula_series_emb)
            else:
                 copula_encoded = None # Ensure copula_encoded is None if embeddings weren't created

            # Process with copula if encoded representation exists
            if copula_encoded is not None:
                self.copula_loss = -self.copula.log_prob(u_vals, copula_encoded)
            else:
                # If copula_encoded is None (e.g., stage 1 or bagging happened before init), set loss to zero
                self.copula_loss = torch.zeros(1, device=device).mean()
        else: # Corresponds to 'if not self.skip_copula and self.stage >= 2:'
            self.copula_loss = torch.zeros(1, device=device).mean()

        # Loss normalization based on potentially bagged num_series
        if self.loss_normalization in {"series", "both"}:
             # Use the potentially updated num_series after bagging
             self.copula_loss = self.copula_loss / num_series
             self.marginal_logdet = self.marginal_logdet / num_series
        if self.loss_normalization in {"timesteps", "both"}:
             self.copula_loss = self.copula_loss / pred_len # Use original pred_len
             self.marginal_logdet = self.marginal_logdet / pred_len # Use original pred_len

        # Return output
        output = pred_value # For training, return target as "output"
        loss = self.copula_loss - self.marginal_logdet # Negative log-likelihood

        return output, loss

    @staticmethod
    def _apply_bagging(
        bagging_size,
        time_steps: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        true_value: torch.Tensor,
        flow_series_emb=None,
        copula_series_emb=None,
    ):
        """
        Only keep a small number of series for each of the input tensors.
        Which series will be kept is randomly selected for each batch. The order is not preserved.

        Parameters:
        -----------
        bagging_size: int
            How many series to keep
        time_steps: Tensor [batch, time steps]
            Combined historical and prediction time steps.
        value: Tensor [batch, series, time steps]
            Combined historical and prediction values (normalized).
        mask: Tensor [batch, series, time steps]
            Boolean mask indicating observed (True) vs predicted (False).
        true_value: Tensor [batch, series, time steps]
            Same as value, used for loss calculation.
        flow_series_emb: Tensor [batch, series, flow embedding size]
            An embedding for each series for the marginals, expanded over the batches.
        copula_series_emb: Tensor [batch, series, copula embedding size]
            An embedding for each series for the copula, expanded over the batches.

        Returns:
        --------
        Tuple containing bagged versions of:
        time_steps, value, mask, true_value, flow_series_emb, copula_series_emb
        """
        num_batches = value.shape[0]
        num_series = value.shape[1]

        # Make sure to have the exact same bag for all series within a batch
        bags = [torch.randperm(num_series, device=value.device)[0:bagging_size] for _ in range(num_batches)]

        # Bag the tensors that have a series dimension
        value = torch.stack([value[i, bags[i], :] for i in range(num_batches)], dim=0)
        mask = torch.stack([mask[i, bags[i], :] for i in range(num_batches)], dim=0)
        true_value = torch.stack([true_value[i, bags[i], :] for i in range(num_batches)], dim=0)
        flow_series_emb = torch.stack([flow_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)
        if copula_series_emb is not None:
            copula_series_emb = torch.stack([copula_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)

        # time_steps usually doesn't have a series dimension, so it's returned as is
        return (
            time_steps,
            value,
            mask,
            true_value,
            flow_series_emb,
            copula_series_emb,
        )

    # Sample method simplified assuming forecasting mode and external normalization
    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor, # Expected to be normalized externally
        pred_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        """
        batch_size, num_series, hist_len = hist_value.shape
        pred_len = pred_time.shape[1]
        device = hist_value.device

        # Prepare inputs for encoding (history + zeros for prediction period)
        time_steps = torch.cat([hist_time, pred_time], dim=1)
        series_indices = torch.arange(num_series, device=device)
        value_placeholder = torch.cat([
            hist_value,
            torch.zeros(batch_size, num_series, pred_len, device=device)
        ], dim=2)
        mask = torch.cat([
            torch.ones(batch_size, num_series, hist_len, dtype=torch.bool, device=device),
            torch.zeros(batch_size, num_series, pred_len, dtype=torch.bool, device=device)
        ], dim=2)

        # Create series embeddings (needed for encoding)
        # Expand over batches as encoding expects batch dimension
        flow_series_emb = self.flow_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1)
        copula_series_emb = None
        if not self.skip_copula and self.stage >= 2 and hasattr(self, 'copula_series_encoder'):
             copula_series_emb = self.copula_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1)

        # Encode inputs
        # Pass the created series embeddings directly
        flow_encoded = self._encode_flow(time_steps, value_placeholder, mask, series_indices, flow_series_emb)
        copula_encoded = None
        if copula_series_emb is not None: # Check if copula embeddings exist
             copula_encoded = self._encode_copula(time_steps, value_placeholder, mask, series_indices, copula_series_emb)

        # Generate uniform samples or copula samples
        if copula_encoded is None: # Stage 1 or copula skipped
            u_samples = torch.rand(num_samples, batch_size, num_series, pred_len, device=device)
        else:
            # Sample from copula using encoded representation of prediction steps
            # Ensure copula exists before sampling
            if not hasattr(self, 'copula'):
                 logger.error("Attempting to sample from copula, but it's not initialized.")
                 # Fallback: generate uniform samples
                 u_samples = torch.rand(num_samples, batch_size, num_series, pred_len, device=device)
            else:
                 # Select the copula encoding corresponding to prediction steps
                 copula_encoded_pred = copula_encoded[:, :, hist_len:] # Shape [batch, series, pred_len, dim]
                 # Reshape for copula sample method if needed by its implementation
                 # Assuming copula.sample expects [batch, series * pred_len, dim]
                 copula_encoded_pred_flat = copula_encoded_pred.reshape(batch_size, num_series * pred_len, -1)

                 u_samples_flat = self.copula.sample(
                     num_samples,
                     copula_encoded_pred_flat, # Pass flattened copula encoding for prediction steps
                     copula_encoded_pred_flat.shape[-1] # Pass embedding dim
                 ) # Expected Shape: [batch, series*pred_len, num_samples] ? Check AttentionalCopula.sample output shape
                 # Reshape back to [num_samples, batch, series, pred_len]
                 try:
                      u_samples = u_samples_flat.reshape(batch_size, num_series, pred_len, num_samples).permute(3, 0, 1, 2)
                 except RuntimeError as e:
                      logger.error(f"Error reshaping copula samples: {e}. Input shape: {u_samples_flat.shape}")
                      # Fallback: generate uniform samples
                      u_samples = torch.rand(num_samples, batch_size, num_series, pred_len, device=device)


        # Transform samples using marginal inverse CDF
        # Reshape inputs for marginal.inverse
        # Select flow encoding for prediction steps
        flow_encoded_pred = flow_encoded[:, :, hist_len:] # Shape [batch, series, pred_len, dim]
        # Reshape u_samples and flow_encoded_pred for marginal.inverse
        # u_samples: [num_samples, batch, series, pred_len] -> [batch, series, pred_len, num_samples]
        u_samples_permuted = u_samples.permute(1, 2, 3, 0)

        # marginal.inverse expects context [batch*N, dim] and u [batch*N, num_samples]
        # Reshape context: [batch, series, pred_len, dim] -> [batch * series * pred_len, dim]
        flow_encoded_pred_flat = flow_encoded_pred.reshape(-1, flow_encoded_pred.shape[-1])
        # Reshape u: [batch, series, pred_len, num_samples] -> [batch * series * pred_len, num_samples]
        u_samples_flat = u_samples_permuted.reshape(-1, num_samples)

        samples_normalized_flat = self.marginal.inverse(flow_encoded_pred_flat, u_samples_flat)
        # Reshape back: [batch * series * pred_len, num_samples] -> [batch, series, pred_len, num_samples]
        samples_normalized = samples_normalized_flat.reshape(batch_size, num_series, pred_len, num_samples)

        # Permute to final shape: [num_samples, batch, series, pred_len]
        samples = samples_normalized.permute(3, 0, 1, 2)


        # Return normalized samples; denormalization is external
        return samples

    def _encode_flow(self, time_steps, value, mask, series_indices, flow_series_emb):
        """Encode data for flow component, accepting pre-computed embeddings"""
        batch_size, num_series, num_timesteps = value.shape # Get dimensions from value tensor
        device = time_steps.device

        # flow_series_emb is now passed in, shape [batch, series, dim]
        # Expand series embedding to match time dimension
        series_emb_expanded = flow_series_emb.unsqueeze(2).expand(-1, -1, num_timesteps, -1) # [batch, series, time, dim]

        # Prepare input tensor - permute value and mask to [batch, time, series, 1] first
        flow_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),  # [batch, time, series, 1]
            series_emb_expanded.permute(0, 2, 1, 3), # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1)

        # Reshape for input encoder [batch * time * series, features]
        flow_input = flow_input.reshape(-1, flow_input.shape[-1])

        # Apply input encoder
        encoded = self.flow_input_encoder(flow_input)
        # Reshape back [batch, time, series, embed_dim]
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)

        # Apply positional encoding
        # time_steps shape [batch, time] -> unsqueeze for pos encoding [batch, time, 1]
        time_tensor = time_steps.unsqueeze(-1)
        # encoded shape [batch, time, series, embed_dim]
        # PositionalEncoding handles broadcasting/expanding time encoding to match series dim
        pos_encoded = self.flow_time_encoding(encoded, time_tensor)
        encoded = self.flow_pos_adjust(pos_encoded) # Apply adjustment if needed

        # Apply flow encoder (handles standard vs temporal internally now)
        encoded = self.flow_encoder(encoded) # Assumes encoder handles input shape correctly

        # Final reshape and permute to [batch, series, time, dim]
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        encoded = encoded.permute(0, 2, 1, 3)

        return encoded

    def _encode_copula(self, time_steps, value, mask, series_indices, copula_series_emb):
        """Encode data for copula component, accepting pre-computed embeddings"""
        batch_size, num_series, num_timesteps = value.shape
        device = time_steps.device

        # copula_series_emb is passed in, shape [batch, series, dim]
        # Expand series embedding to match time dimension
        series_emb_expanded = copula_series_emb.unsqueeze(2).expand(-1, -1, num_timesteps, -1) # [batch, series, time, dim]

        # Prepare input tensor - permute value and mask to [batch, time, series, 1] first
        copula_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),  # [batch, time, series, 1]
            series_emb_expanded.permute(0, 2, 1, 3), # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1)

        # Reshape for input encoder [batch * time * series, features]
        copula_input = copula_input.reshape(-1, copula_input.shape[-1])
        # Apply input encoder
        encoded = self.copula_input_encoder(copula_input)

        # Reshape back [batch, time, series, embed_dim]
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)

        # Apply positional encoding
        time_tensor = time_steps.unsqueeze(-1)
        pos_encoded = self.copula_time_encoding(encoded, time_tensor)
        encoded = self.copula_pos_adjust(pos_encoded)

        # Apply copula encoder
        encoded = self.copula_encoder(encoded) # Assumes encoder handles input shape correctly

        # Final reshape and permute to [batch, series, time, dim]
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        encoded = encoded.permute(0, 2, 1, 3)

        return encoded

    def set_stage(self, stage: int):
        """Set the training stage"""
        assert stage in [1, 2], "Stage must be 1 or 2"
        self.stage = stage

        # Re-initialize copula components if switching to stage 2 and they weren't initialized before
        if stage == 2 and self.skip_copula:
             # Check if args are available before initializing
             # Check specifically for copula_encoder_args as it's needed for d_model
             if self.copula_decoder_args is not None and self.copula_encoder_args is not None:
                 logger.info("Initializing copula components for stage 2...")
                 self.skip_copula = False
                 # Pass the stored config dictionaries
                 self._initialize_copula_components()
                 # Ensure the initialized components are moved to the correct device
                 # This assumes the TACTiS module itself is already on the correct device
                 device = next(self.parameters()).device
                 if hasattr(self, 'copula_series_encoder'):
                      self.copula_series_encoder.to(device)
                 if hasattr(self, 'copula_input_encoder'):
                      self.copula_input_encoder.to(device)
                 if hasattr(self, 'copula_time_encoding'):
                      self.copula_time_encoding.to(device)
                 if hasattr(self, 'copula_pos_adjust'):
                      self.copula_pos_adjust.to(device)
                 if hasattr(self, 'copula_encoder'):
                      self.copula_encoder.to(device)
                 if hasattr(self, 'copula'):
                      self.copula.to(device)

                 else:
                    logger.warning("Cannot initialize copula components for stage 2 - config args missing.")
                    self.copula_loss = -self.copula.log_prob(u_vals, copula_encoded)
             else:
                 # If copula_encoded is None (e.g., stage 1 or bagging happened before init), set loss to zero
                 self.copula_loss = torch.zeros(1, device=device).mean()
        else: # Corresponds to 'if not self.skip_copula and self.stage >= 2:'
            self.copula_loss = torch.zeros(1, device=device).mean()

        # Loss normalization based on potentially bagged num_series
        if self.loss_normalization in {"series", "both"}:
             # Use the potentially updated num_series after bagging
             self.copula_loss = self.copula_loss / num_series
             self.marginal_logdet = self.marginal_logdet / num_series
        if self.loss_normalization in {"timesteps", "both"}:
             self.copula_loss = self.copula_loss / pred_len # Use original pred_len
             self.marginal_logdet = self.marginal_logdet / pred_len # Use original pred_len

        # Return output
        output = pred_value  # For training, return target as "output"
        loss = self.copula_loss - self.marginal_logdet  # Negative log-likelihood
        
        return output, loss
    
    # Sample method simplified assuming forecasting mode and external normalization
    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor, # Expected to be normalized externally
        pred_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        """
        batch_size, num_series, hist_len = hist_value.shape
        pred_len = pred_time.shape[1]
        device = hist_value.device

        # Prepare inputs for encoding (history + zeros for prediction period)
        time_steps = torch.cat([hist_time, pred_time], dim=1)
        series_indices = torch.arange(num_series, device=device)
        value_placeholder = torch.cat([
            hist_value,
            torch.zeros(batch_size, num_series, pred_len, device=device)
        ], dim=2)
        mask = torch.cat([
            torch.ones(batch_size, num_series, hist_len, dtype=torch.bool, device=device),
            torch.zeros(batch_size, num_series, pred_len, dtype=torch.bool, device=device)
        ], dim=2)

        # Encode inputs
        flow_encoded = self._encode_flow(time_steps, value_placeholder, mask, series_indices)
        copula_encoded = None
        if not self.skip_copula and self.stage >= 2:
            copula_encoded = self._encode_copula(time_steps, value_placeholder, mask, series_indices)

        # Generate uniform samples or copula samples
        if copula_encoded is None: # Stage 1 or copula skipped
            u_samples = torch.rand(num_samples, batch_size, num_series, pred_len, device=device)
        else:
            # Sample from copula using encoded representation of prediction steps
            u_samples_flat = self.copula.sample(
                num_samples,
                copula_encoded[:, :, hist_len:], # Pass copula encoding for prediction steps
                copula_encoded.shape[-1] # Pass embedding dim
            ) # Shape: [batch, series*pred_len, num_samples]
            u_samples = u_samples_flat.reshape(batch_size, num_series, pred_len, num_samples).permute(3, 0, 1, 2)

        # Transform samples using marginal inverse CDF
        # Reshape inputs for marginal.inverse
        u_samples_reshaped = u_samples.permute(1, 2, 3, 0).reshape(batch_size, num_series * pred_len, num_samples)
        flow_encoded_pred = flow_encoded[:, :, hist_len:].reshape(batch_size, num_series * pred_len, -1)

        samples_normalized = self.marginal.inverse(flow_encoded_pred, u_samples_reshaped)
        samples = samples_normalized.reshape(batch_size, num_series, pred_len, num_samples).permute(3, 0, 1, 2)

        # Return normalized samples; denormalization is external
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
        
        # Apply positional encoding - Fix for time tensor
        # Create time tensor with the correct shape for positional encoding
        # This is where the error was occurring
        time_tensor = time_steps.unsqueeze(-1)
        # We don't need to expand to match series dimension, as the positional encoding
        # will handle this mismatch with our improved implementation
        # Apply positional encoding and adjust dimension if needed
        pos_encoded = self.flow_time_encoding(encoded, time_tensor)
        encoded = self.flow_pos_adjust(pos_encoded)
        
        # Apply flow encoder (already initialized correctly based on self.encoder_type)
        # Standard Transformer expects [batch, time * series, embed_dim] if batch_first=True
        # TemporalEncoder expects [batch, time, series, embed_dim] and handles internal reshaping
        if isinstance(self.flow_encoder, nn.TransformerEncoder):
            encoded_reshaped = encoded.reshape(batch_size, num_timesteps * num_series, -1)
            encoded_out = self.flow_encoder(encoded_reshaped)
            encoded = encoded_out.reshape(batch_size, num_timesteps, num_series, -1) # Reshape back
        elif isinstance(self.flow_encoder, TemporalEncoder):
            # TemporalEncoder expects [batch, time, series, embed_dim]
            encoded = self.flow_encoder(encoded) # Pass directly
        else:
            # Should not happen if initialization is correct
            raise TypeError(f"Unexpected flow_encoder type: {type(self.flow_encoder)}")
            
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
        
        # Apply positional encoding - Fix for time tensor
        # Create time tensor with the correct shape for positional encoding
        time_tensor = time_steps.unsqueeze(-1)  # simplified time tensor
        # Apply positional encoding and adjust dimension if needed
        pos_encoded = self.copula_time_encoding(encoded, time_tensor)
        encoded = self.copula_pos_adjust(pos_encoded)
        
        # Apply copula encoder (already initialized correctly based on self.encoder_type)
        if isinstance(self.copula_encoder, nn.TransformerEncoder):
            encoded_reshaped = encoded.reshape(batch_size, num_timesteps * num_series, -1)
            encoded_out = self.copula_encoder(encoded_reshaped)
            encoded = encoded_out.reshape(batch_size, num_timesteps, num_series, -1) # Reshape back
        elif isinstance(self.copula_encoder, TemporalEncoder):
            # TemporalEncoder expects [batch, time, series, embed_dim]
            encoded = self.copula_encoder(encoded) # Pass directly
        else:
            # Should not happen if initialization is correct
            raise TypeError(f"Unexpected copula_encoder type: {type(self.copula_encoder)}")
            
        # Arrange dimensions to [batch, series, time, dim]
        encoded = encoded.permute(0, 2, 1, 3)
        
        return encoded
    
    def set_stage(self, stage: int):
        """Set the training stage"""
        assert stage in [1, 2], "Stage must be 1 or 2"
        self.stage = stage
        
        # Re-initialize copula components if switching to stage 2 and they weren't initialized before
        # Re-initialize copula components if switching to stage 2 and they weren't initialized before
        if stage == 2 and self.skip_copula:
             # Check if args are available before initializing
             if self.copula_decoder_args is not None and self.copula_encoder_args is not None:
                 logger.info("Initializing copula components for stage 2...")
                 self.skip_copula = False
                 # Pass the stored config dictionaries
                 self._initialize_copula_components()
             else:
                 logger.warning("Cannot initialize copula components for stage 2 - config args missing.")
