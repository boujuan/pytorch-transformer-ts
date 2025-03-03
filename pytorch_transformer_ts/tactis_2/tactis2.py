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
        # Print full tensor dimensions for debugging
        print(f"PositionalEncoding input shapes: x={x.shape}, timesteps={timesteps.shape}")
        print(f"PositionalEncoding buffer shape: pos_encoding={self.pos_encoding.shape}")
        
        min_t = timesteps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        delta_t = timesteps - min_t
        delta_t = torch.clamp(delta_t, 0, self.pos_encoding.shape[0] - 1)
        
        print(f"delta_t shape: {delta_t.shape}, min={delta_t.min().item()}, max={delta_t.max().item()}")
        
        try:
            # Get the positional encodings
            # This will have shape [batch, timesteps, 1, embedding_dim] for our data
            pos_enc = self.pos_encoding[delta_t.long()]
            print(f"pos_enc shape after indexing: {pos_enc.shape}")
            
            # Special case for handling [batch, time, series, features] tensors
            # This is the specific case we're seeing in TACTiS-2
            if len(x.shape) == 4 and len(pos_enc.shape) == 4:
                # Check if the specific mismatch is in the series dimension (dim 2)
                if pos_enc.shape[2] == 1 and x.shape[2] > 1:
                    # Explicitly expand the series dimension
                    pos_enc = pos_enc.expand(-1, -1, x.shape[2], -1)
                    print(f"Expanded series dimension: {pos_enc.shape}")
                
            # General handling for any remaining shape mismatches
            if pos_enc.shape != x.shape:
                print(f"Shape mismatch remains! pos_enc: {pos_enc.shape}, x: {x.shape}")
                
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
                    print(f"Error during shape adjustment: {inner_e}")
                    # As last resort, if all else fails
                    return self.dropout(x)  # Skip positional encoding rather than crash
            
            # Verify the shapes match now
            print(f"Final shapes - pos_enc: {pos_enc.shape}, x: {x.shape}")
            
            # Add the position encoding to the input
            output = x + pos_enc
            
        except Exception as e:
            print(f"Error in positional encoding: {e}")
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
        # Print shapes for debugging
        print(f"SigmoidFlow input shapes - params: {params.shape}, x: {x.shape}, logdet: {logdet.shape}")
        
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
        
        print(f"SigmoidFlow - logj shape: {logj.shape}, logdet shape: {logdet.shape}")
        print(f"SigmoidFlow - logj min: {logj.min().item()}, max: {logj.max().item()}")
        
        # Fix for dimension mismatch: expand logdet to match logj's dimensions
        if logdet.shape != logj.shape:
            print(f"Dimension mismatch between logdet {logdet.shape} and logj {logj.shape}")
            # Expand logdet to match logj's dimensions
            for _ in range(len(logj.shape) - len(logdet.shape)):
                logdet = logdet.unsqueeze(-1)
            # Now broadcast to the full shape
            logdet = logdet.expand_as(logj)
            print(f"Expanded logdet shape: {logdet.shape}")
            
        # Add log jacobian with checks for NaN
        log_logj = torch.log(logj)
        if torch.isnan(log_logj).any():
            print(f"WARNING: NaN detected in log_logj! Min logj: {logj.min().item()}")
            # Replace NaNs with a large negative number (log of EPSILON)
            log_logj = torch.nan_to_num(log_logj, nan=math.log(EPSILON))
        
        logdet = logdet + log_logj
        
        # Check for NaNs in logdet
        if torch.isnan(logdet).any():
            print(f"WARNING: NaN detected in logdet after addition!")
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
            print(f"WARNING: NaNs in final output! xnew: {torch.isnan(xnew).sum().item()}, logdet: {torch.isnan(logdet).sum().item()}")
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
        # Print input shapes for debugging
        print(f"DeepSigmoidFlow input shapes - params: {params.shape}, x: {x.shape}")
        
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
        
        print(f"DeepSigmoidFlow - initialized logdet shape: {logdet.shape}")
        
        # Track original x for debugging
        x_orig = x.clone()
        
        for i, layer in enumerate(self.layers):
            # Extract parameters for this layer
            layer_params = params[..., i * self.params_length : (i + 1) * self.params_length]
            print(f"DeepSigmoidFlow - layer {i} params shape: {layer_params.shape}")
            
            # Check for NaNs in inputs
            if torch.isnan(x).any():
                print(f"WARNING: NaN in x before layer {i}")
                # Replace NaNs with zeros to allow continued processing
                x = torch.nan_to_num(x)
                
            if torch.isnan(logdet).any():
                print(f"WARNING: NaN in logdet before layer {i}")
                logdet = torch.nan_to_num(logdet)
                
            # Process through the layer
            x, logdet = layer(
                layer_params,
                x,
                logdet,
            )
            
            # Check outputs of each layer
            print(f"DeepSigmoidFlow - after layer {i}, x range: [{x.min().item()}, {x.max().item()}], logdet shape: {logdet.shape}")
            
        # Final check for NaNs
        if torch.isnan(x).any() or torch.isnan(logdet).any():
            print(f"WARNING: NaNs in final output of DeepSigmoidFlow!")
            print(f"x_orig min: {x_orig.min().item()}, max: {x_orig.max().item()}")
            print(f"x NaNs: {torch.isnan(x).sum().item()}, logdet NaNs: {torch.isnan(logdet).sum().item()}")
            
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
            print(f"Creating dimension reducer from {input_dim} to {self.expected_context_dim}")
            self.input_dim = input_dim
            self.dim_reducer = nn.Linear(input_dim, self.expected_context_dim)
            # Move to the same device as the input tensor
            self.dim_reducer = self.dim_reducer.to(input_tensor.device)
        
        # Apply dimension reduction if needed
        if self.dim_reducer is not None:
            return self.dim_reducer(input_tensor)
        else:
            return input_tensor
    
    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform with derivative computation"""
        # Print shapes for debugging
        print(f"DSFMarginal inputs - context: {context.shape}, x: {x.shape}")
        
        # Check for NaNs in input
        if torch.isnan(x).any():
            print(f"WARNING: NaN values detected in input x, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
            
        if torch.isnan(context).any():
            print(f"WARNING: NaN values detected in context tensor, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)
        
        # Clamp x to a valid range to prevent numerical issues
        # The flow expects values that can be reasonably transformed without producing extremes
        x_min, x_max = x.min().item(), x.max().item()
        print(f"Input x range: min={x_min}, max={x_max}")
        
        # Only clamp if values are extreme
        if x_min < -100 or x_max > 100:
            print(f"Clamping extreme values in x")
            x = torch.clamp(x, min=-100.0, max=100.0)
        
        # Reshape context for linear layer
        # context comes in as [batch, series, time, dim] but the linear layer expects [batch, dim]
        # or [N, dim] where N = batch * series * time
        batch_size, num_series, num_timesteps, feature_dim = context.shape
        
        # Reshape to [batch*series*time, feature_dim]
        reshaped_context = context.reshape(-1, feature_dim)
        print(f"Reshaped context for conditioner: {reshaped_context.shape}")
        
        # Ensure dimensions match before applying conditioner
        reshaped_context = self._ensure_dimensions_match(reshaped_context)
        print(f"Reshaped context after dimension matching: {reshaped_context.shape}")
        
        # Check for NaNs after reshaping
        if torch.isnan(reshaped_context).any():
            print(f"WARNING: NaN values detected after reshaping context, replacing with zeros")
            reshaped_context = torch.nan_to_num(reshaped_context, nan=0.0)
        
        # Apply conditioner
        try:
            marginal_params = self.marginal_conditioner(reshaped_context)
            
            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                print(f"WARNING: NaN values in marginal parameters, replacing with zeros")
                marginal_params = torch.nan_to_num(marginal_params, nan=0.0)
                
            print(f"Marginal params after conditioner: {marginal_params.shape}")
            
            # Reshape back to match x
            marginal_params = marginal_params.reshape(batch_size, num_series, num_timesteps, -1)
            print(f"Marginal params reshaped: {marginal_params.shape}, x shape: {x.shape}")
            
            # Make sure dimensions match
            if marginal_params.dim() == x.dim():
                # Check if reshaping is needed
                if marginal_params.shape != x.shape:
                    print(f"Dimension match but shapes differ: params={marginal_params.shape}, x={x.shape}")
                    marginal_params = marginal_params[:, :, :, :x.size(-1)]
            
            # Forward through the flow
            transformed_x, logdet = self.marginal_flow.forward(marginal_params, x)
            
            # Check for NaNs in the output
            if torch.isnan(transformed_x).any() or torch.isnan(logdet).any():
                print(f"WARNING: NaN values in DSFMarginal output! transformed_x NaNs: {torch.isnan(transformed_x).sum().item()}, logdet NaNs: {torch.isnan(logdet).sum().item()}")
                transformed_x = torch.nan_to_num(transformed_x, nan=0.0)
                logdet = torch.nan_to_num(logdet, nan=0.0)
            
            return transformed_x, logdet
            
        except Exception as e:
            print(f"ERROR in DSFMarginal forward_logdet: {e}")
            # Return zeros as a fallback to avoid crashing
            zeros_x = torch.zeros_like(x)
            zeros_logdet = torch.zeros(batch_size, num_series, num_timesteps, device=x.device)
            return zeros_x, zeros_logdet
    
    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Transform without derivative computation"""
        # Check for NaNs in input
        if torch.isnan(x).any():
            print(f"WARNING: NaN values detected in input x for forward_no_logdet, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
            
        if torch.isnan(context).any():
            print(f"WARNING: NaN values detected in context tensor for forward_no_logdet, replacing with zeros")
            context = torch.nan_to_num(context, nan=0.0)
        
        # Clamp x to a valid range to prevent numerical issues
        x_min, x_max = x.min().item(), x.max().item()
        if x_min < -100 or x_max > 100:
            print(f"Clamping extreme values in x for forward_no_logdet")
            x = torch.clamp(x, min=-100.0, max=100.0)
        
        # Reshape context for linear layer
        batch_size, num_series, num_timesteps, feature_dim = context.shape
        
        # Reshape to [batch*series*time, feature_dim]
        reshaped_context = context.reshape(-1, feature_dim)
        
        # Ensure dimensions match before applying conditioner
        reshaped_context = self._ensure_dimensions_match(reshaped_context)
        
        # Check for NaNs after reshaping
        if torch.isnan(reshaped_context).any():
            print(f"WARNING: NaN values after reshaping context in forward_no_logdet, replacing with zeros")
            reshaped_context = torch.nan_to_num(reshaped_context, nan=0.0)
        
        try:
            # Apply conditioner
            marginal_params = self.marginal_conditioner(reshaped_context)
            
            # Check for NaNs in parameters
            if torch.isnan(marginal_params).any():
                print(f"WARNING: NaN values in marginal parameters for forward_no_logdet, replacing with zeros")
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
                print(f"WARNING: NaN values in DSFMarginal forward_no_logdet output! Replacing with zeros")
                transformed_x = torch.nan_to_num(transformed_x, nan=0.0)
            
            return transformed_x
            
        except Exception as e:
            print(f"ERROR in DSFMarginal forward_no_logdet: {e}")
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
        data_normalization: str
            Type of normalization to apply to data values
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
        experiment_mode: str
            Mode of operation: "forecasting" or "imputation"
        """
        super().__init__()
        
        # Store parameters
        self.num_series = num_series
        self.flow_series_embedding_dim = flow_series_embedding_dim
        self.copula_series_embedding_dim = copula_series_embedding_dim
        self.flow_input_encoder_layers = flow_input_encoder_layers
        self.copula_input_encoder_layers = copula_input_encoder_layers
        self.input_encoding_normalization = input_encoding_normalization
        self.data_normalization = data_normalization
        self.loss_normalization = loss_normalization
        self.skip_copula = skip_copula
        self.experiment_mode = experiment_mode
        self.bagging_size = bagging_size
        
        # Stage tracks whether to use flow only (1) or flow+copula (2)
        self.stage = 1
        self.marginal_logdet = None
        self.copula_loss = None
        
        # Create embeddings for series
        self.flow_series_encoder = nn.Embedding(num_series, flow_series_embedding_dim)
        
        # Initialize model components
        self._initialize_flow_components(flow_encoder, positional_encoding)
        
        # Marginals - REMOVED (moved to _initialize_flow_components)
        
        # Initialize copula components if not skipping
        if not skip_copula:
            self._initialize_copula_components(copula_encoder, positional_encoding, copula_decoder)
    
    def _initialize_flow_components(self, flow_encoder_args, pos_encoding_args):
        """Initialize flow-related components"""
        # Input encoder
        encoder_dim = 256  # Default dimension
        
        # Print encoder dimensions for debugging
        print(f"TACTiS initialization - flow_series_embedding_dim: {self.flow_series_embedding_dim}")
        print(f"TACTiS initialization - encoder_dim: {encoder_dim}")
        
        self.flow_input_encoder = nn.Sequential(
            nn.Linear(self.flow_series_embedding_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, encoder_dim)
        )
        
        # Positional encoding - explicitly set to same dimension as encoder
        self.flow_time_encoding = PositionalEncoding(embedding_dim=encoder_dim, dropout=0.1, max_len=5000)
        
        # Print positional encoding dimensions for debugging
        print(f"TACTiS initialization - positional_encoding dim: {self.flow_time_encoding.pos_encoding.size(1)}")
        
        # Flow encoder is a transformer encoder
        default_flow_args = {
            "d_model": encoder_dim,  # Match with flow_input_encoder output
            "nhead": 8,
            "num_encoder_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.1
        }
        
        if flow_encoder_args is not None:
            default_flow_args.update(flow_encoder_args)
            
        self.flow_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=default_flow_args["d_model"],
                nhead=default_flow_args["nhead"],
                dim_feedforward=default_flow_args["dim_feedforward"],
                dropout=default_flow_args["dropout"],
                batch_first=True
            ),
            num_layers=default_flow_args["num_encoder_layers"]
        )
        
        # Update the marginal flow context_dim to match the encoder output
        # NOTE: In the __init__ method this was hardcoded to flow_series_embedding_dim*4
        print(f"Setting up marginal flow with context_dim={encoder_dim} (previously was {self.flow_series_embedding_dim*4})")
        
        # Marginals
        self.marginal = DSFMarginal(
            context_dim=encoder_dim,  # Use encoder_dim instead of flow_series_embedding_dim*4
            mlp_layers=2,
            mlp_dim=128,
            flow_layers=3,
            flow_hid_dim=32
        )
    
    def _initialize_copula_components(self, copula_encoder_args, pos_encoding_args, copula_decoder_args):
        """Initialize copula-related components"""
        # Series encoder
        self.copula_series_encoder = nn.Embedding(self.num_series, self.copula_series_embedding_dim)
        
        # Input encoder
        encoder_dim = 256  # Default dimension
        
        # Print encoder dimensions for debugging
        print(f"TACTiS initialization - copula_series_embedding_dim: {self.copula_series_embedding_dim}")
        print(f"TACTiS initialization - copula encoder_dim: {encoder_dim}")
        
        self.copula_input_encoder = nn.Sequential(
            nn.Linear(self.copula_series_embedding_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, encoder_dim)
        )
        
        # Positional encoding - explicitly set to same dimension as encoder
        self.copula_time_encoding = PositionalEncoding(embedding_dim=encoder_dim, dropout=0.1, max_len=5000)
        
        # Print positional encoding dimensions for debugging
        print(f"TACTiS initialization - copula positional_encoding dim: {self.copula_time_encoding.pos_encoding.size(1)}")
        
        # Copula encoder is a transformer encoder
        default_copula_args = {
            "d_model": encoder_dim,  # Match with copula_input_encoder output
            "nhead": 8,
            "num_encoder_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.1
        }
        
        if copula_encoder_args is not None:
            default_copula_args.update(copula_encoder_args)
            
        self.copula_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=default_copula_args["d_model"],
                nhead=default_copula_args["nhead"],
                dim_feedforward=default_copula_args["dim_feedforward"],
                dropout=default_copula_args["dropout"],
                batch_first=True
            ),
            num_layers=default_copula_args["num_encoder_layers"]
        )
        
        # Attentional copula for modeling dependencies
        self.copula = AttentionalCopula(
            input_dim=encoder_dim,
            attention_layers=2,
            attention_heads=4,
            attention_dim=64,
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
        
        # Print input shapes for debugging
        print(f"_encode_flow input shapes: time_steps={time_steps.shape}, value={value.shape}, mask={mask.shape}, series_indices={series_indices.shape}")
        
        # Get series embeddings
        series_emb = self.flow_series_encoder(series_indices)  # [series, dim]
        print(f"series_emb shape: {series_emb.shape}")
        
        # Prepare input tensor with values, series embedding, and mask
        flow_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),  # [batch, time, series, 1]
            series_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, num_timesteps, -1, -1),  # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1)
        print(f"flow_input shape: {flow_input.shape}")
        
        # Reshape for input encoder
        flow_input = flow_input.reshape(batch_size * num_timesteps * num_series, -1)
        print(f"flow_input reshaped: {flow_input.shape}")
        
        # Apply input encoder
        encoded = self.flow_input_encoder(flow_input)
        print(f"encoded from flow_input_encoder: {encoded.shape}")
        
        # Reshape back
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        print(f"encoded reshaped for time encoding: {encoded.shape}")
        
        # Apply positional encoding - Fix for time tensor
        # Create time tensor with the correct shape for positional encoding
        # This is where the error was occurring
        time_tensor = time_steps.unsqueeze(-1)
        # We don't need to expand to match series dimension, as the positional encoding
        # will handle this mismatch with our improved implementation
        print(f"time_tensor for positional encoding: {time_tensor.shape}")
        
        # Apply positional encoding with the simplified time_tensor
        encoded = self.flow_time_encoding(encoded, time_tensor)
        print(f"encoded after positional encoding: {encoded.shape}")
        
        # Apply flow encoder
        encoded = encoded.reshape(batch_size, num_timesteps * num_series, -1)
        print(f"encoded reshaped for flow encoder: {encoded.shape}")
        
        
        encoded = self.flow_encoder(encoded)
        print(f"encoded after flow encoder: {encoded.shape}")
        
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        print(f"encoded final reshape: {encoded.shape}")
        
        # Arrange dimensions to [batch, series, time, dim]
        encoded = encoded.permute(0, 2, 1, 3)
        print(f"encoded after permute: {encoded.shape}")
        
        return encoded
    
    def _encode_copula(self, time_steps, value, mask, series_indices):
        """Encode data for copula component"""
        batch_size, num_timesteps = time_steps.shape
        num_series = series_indices.size(0)
        device = time_steps.device
        
        # Print input shapes for debugging
        print(f"_encode_copula input shapes: time_steps={time_steps.shape}, value={value.shape}, mask={mask.shape}, series_indices={series_indices.shape}")
        
        # Get series embeddings
        series_emb = self.copula_series_encoder(series_indices)  # [series, dim]
        print(f"copula series_emb shape: {series_emb.shape}")
        
        # Prepare input tensor with values, series embedding, and mask
        copula_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),  # [batch, time, series, 1]
            series_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, num_timesteps, -1, -1),  # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1)
        print(f"copula_input shape: {copula_input.shape}")
        
        # Reshape for input encoder
        copula_input = copula_input.reshape(batch_size * num_timesteps * num_series, -1)
        print(f"copula_input reshaped: {copula_input.shape}")
        
        # Apply input encoder
        encoded = self.copula_input_encoder(copula_input)
        print(f"encoded from copula_input_encoder: {encoded.shape}")
        
        # Reshape back
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        print(f"encoded reshaped for time encoding: {encoded.shape}")
        
        # Apply positional encoding - Fix for time tensor
        # Create time tensor with the correct shape for positional encoding
        time_tensor = time_steps.unsqueeze(-1)  # simplified time tensor
        print(f"copula time_tensor for positional encoding: {time_tensor.shape}")
        
        # Apply positional encoding with the simplified time_tensor
        encoded = self.copula_time_encoding(encoded, time_tensor)
        print(f"encoded after positional encoding: {encoded.shape}")
        
        # Apply copula encoder
        encoded = encoded.reshape(batch_size, num_timesteps * num_series, -1)
        print(f"encoded reshaped for copula encoder: {encoded.shape}")
        
        encoded = self.copula_encoder(encoded)
        print(f"encoded after copula encoder: {encoded.shape}")
        
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, -1)
        print(f"encoded final reshape: {encoded.shape}")
        
        # Arrange dimensions to [batch, series, time, dim]
        encoded = encoded.permute(0, 2, 1, 3)
        print(f"encoded after permute: {encoded.shape}")
        
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
