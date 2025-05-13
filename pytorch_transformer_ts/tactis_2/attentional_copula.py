import torch
from torch import nn
import math
from typing import Dict, Type # Added for type hinting
import torch.utils.checkpoint # For gradient checkpointing

# Helper dictionary to map activation function names to nn modules
_ACTIVATION_MAP: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    # Add other activations as needed
}


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
        activation_function: str = "ReLU",
        use_gradient_checkpointing: bool = False, # New parameter
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Validate activation function
        if activation_function.lower() not in _ACTIVATION_MAP:
            raise ValueError(
                f"Unsupported activation function: {activation_function}. "
                f"Supported functions are: {list(_ACTIVATION_MAP.keys())}"
            )
        self.activation_cls = _ACTIVATION_MAP[activation_function.lower()]

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
            # Use the selected activation class
            layers = [nn.Linear(attention_heads * attention_dim, mlp_dim), self.activation_cls()]
            for _ in range(1, mlp_layers):
                layers += [nn.Linear(mlp_dim, mlp_dim), self.activation_cls()]
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

    def _attention_block(self, att_value: torch.Tensor, layer_idx: int, keys_current_layer: torch.Tensor, values_current_layer: torch.Tensor, product_mask: torch.Tensor) -> torch.Tensor:
        """Helper function for one attention block, to be checkpointed if enabled."""
        # Split the hidden layer into its various heads (Query)
        # Shape: [bsz, num_variables, num_attention_heads, attention_dim]
        att_value_heads = att_value.reshape(
            att_value.shape[0],
            att_value.shape[1],
            self.attention_heads,
            self.attention_dim,
        )

        # Perform Attention
        # product_base shape: [bsz, num_attention_heads, num_variables, num_history+num_variables]
        product_base = torch.einsum("bvhi,bhwi->bhvw", att_value_heads, keys_current_layer)
        product = product_base - product_mask
        product = self.attention_dim ** (-0.5) * product
        weights = nn.functional.softmax(product, dim=-1)

        # att shape: [bsz, num_variables, num_attention_heads, attention_dim]
        att = torch.einsum("bhvw,bhwj->bvhj", weights, values_current_layer)

        # Merge heads: [bsz, num_variables, num_attention_heads * attention_dim]
        att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])
        
        # Dropout, Residual, LayerNorm
        att_merged_heads = self.attention_dropouts[layer_idx](att_merged_heads)
        att_value = att_value + att_merged_heads
        att_value = self.attention_layer_norms[layer_idx](att_value)

        # Feed-forward, Residual, LayerNorm
        att_feed_forward = self.feed_forwards[layer_idx](att_value)
        att_value = att_value + att_feed_forward
        att_value = self.feed_forward_layer_norms[layer_idx](att_value)
        
        return att_value

    def loss(
        self,
        hist_encoded: torch.Tensor,
        hist_true_u: torch.Tensor,
        pred_encoded: torch.Tensor,
        pred_true_u: torch.Tensor,
        num_series: int,
        num_timesteps: int,
    ) -> torch.Tensor:
        """
        Compute the loss function of the copula portion of the decoder.

        Parameters:
        -----------
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
        hist_true_u: Tensor [batch, series * time steps]
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
        pred_true_u: Tensor [batch, series * time steps]

        Returns:
        --------
        loss: torch.Tensor [batch]
        """
        num_batches = pred_encoded.shape[0]
        num_variables = pred_encoded.shape[1]
        num_history = hist_encoded.shape[1]
        device = pred_encoded.device

        assert num_variables == num_series * num_timesteps, (
            f"num_variables: {num_variables} but num_series: {num_series} and num_timesteps: {num_timesteps}"
        )

        permutation = torch.arange(0, num_variables, device=device).long() # Ensure permutation is on the same device

        pred_encoded = pred_encoded[:, permutation, :]
        pred_true_u = pred_true_u[:, permutation]

        att_value = self.dimension_shifting_layer(pred_encoded)

        # Precompute all keys and values outside the loop
        all_keys = []
        all_values = []
        for layer_idx_for_kv in range(self.attention_layers):
            key_input_hist_kv = torch.cat([hist_encoded, hist_true_u[:, :, None]], dim=2)
            key_input_pred_kv = torch.cat([pred_encoded, pred_true_u[:, :, None]], dim=2)
            key_input_kv = torch.cat([key_input_hist_kv, key_input_pred_kv], dim=1)

            value_input_hist_kv = torch.cat([hist_encoded, hist_true_u[:, :, None]], dim=2)
            value_input_pred_kv = torch.cat([pred_encoded, pred_true_u[:, :, None]], dim=2)
            value_input_kv = torch.cat([value_input_hist_kv, value_input_pred_kv], dim=1)
            
            # Ensure mlp output is correctly shaped before cat for heads
            # mlp(key_input_kv) should be [bsz, num_hist+num_vars, attention_dim]
            # Then unsqueeze for head dim before cat
            all_keys.append(
                torch.stack( # stack creates new head dimension
                    [mlp(key_input_kv) for mlp in self.key_creators[layer_idx_for_kv]],
                    dim=1, # New dimension for heads: [bsz, num_heads, num_hist+num_vars, attention_dim]
                )
            )
            all_values.append(
                torch.stack(
                    [mlp(value_input_kv) for mlp in self.value_creators[layer_idx_for_kv]],
                    dim=1, # New dimension for heads: [bsz, num_heads, num_hist+num_vars, attention_dim]
                )
            )
        
        for layer_idx in range(self.attention_layers):
            keys_current_layer = all_keys[layer_idx]
            values_current_layer = all_values[layer_idx]

            product_mask = torch.ones(
                num_batches,
                self.attention_heads,
                num_variables,
                num_variables + num_history,
                device=device,
            )
            product_mask = torch.tril(float("inf") * product_mask).flip((2, 3))

            if self.use_gradient_checkpointing and self.training:
                # Ensure all non-tensor arguments are passed correctly after tensor arguments
                att_value = torch.utils.checkpoint.checkpoint(
                    self._attention_block, att_value, layer_idx, keys_current_layer, values_current_layer, product_mask,
                    use_reentrant=False # Recommended for newer PyTorch versions
                )
            else:
                att_value = self._attention_block(att_value, layer_idx, keys_current_layer, values_current_layer, product_mask)

        # Assign each observed U(0,1) value to a bin.
        # shape: [b, variables*timesteps - 1]
        target = torch.clip(
            torch.floor(pred_true_u[:, 1:] * self.resolution).long(),
            min=0,
            max=self.resolution - 1,
        )

        # Final shape of att_value would be: [bsz, num_variables, num_attention_heads*attention_dim]
        # Compute the (un-normalized) logarithm likelihood of the conditional distribution.
        # Note: This section could instead call a specialized module to allow for easier customization.
        # Get conditional distributions over bins for all variables but the first one.
        # The first one is considered to always be U(0,1), which has a constant logarithm likelihood of 0.
        logits = self.dist_extractors(att_value)[:, 1:, :]  # shape: [b, variables*timesteps - 1, self.resolution]

        # We multiply the probability by self.resolution to get the PDF of the continuous-by-part distribution.
        # prob = self.resolution * softmax(logits, dim=2)
        # logprob = log(self.resolution) + log_softmax(logits, dim=2)
        # Basically softmax of the logits, but the softmax is scaled according to the resolution instead of being in [0,1]
        logprob = math.log(self.resolution) + nn.functional.log_softmax(logits, dim=2)
        # For each batch + variable pair, we want the value of the logits associated with its true value (target):
        # logprob[batch,variable] = logits[batch,variable,target[batch,variable]]
        # Since gather wants the same number of dimensions for both tensors, add and remove a dummy third dimension.
        logprob = torch.gather(logprob, dim=2, index=target[:, :, None])[:, :, 0]

        return -logprob.sum(axis=1)  # Only keep the batch dimension

    def sample(
        self,
        num_samples: int,
        hist_encoded: torch.Tensor,
        hist_true_u: torch.Tensor,
        pred_encoded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted copula.
        (Based on original implementation in tactis/model/decoder.py)

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_encoded: Tensor [batch, series * hist_steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * hist_steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * pred_steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.

        Returns:
        --------
        samples: torch.Tensor [batch, series * pred_steps, samples]
            Samples drawn from the forecasted copula, thus in the [0, 1] range.
            The series and time steps dimensions are merged.
        """
        num_batches = pred_encoded.shape[0]
        num_variables = pred_encoded.shape[1] # num_variables = series * pred_steps
        num_history = hist_encoded.shape[1] # num_history = series * hist_steps
        device = pred_encoded.device

        # Use a fixed permutation for sampling consistency if needed, or keep random
        permutation = torch.arange(0, num_variables).long()
        # If multiple samples require different permutations (more complex):
        # permutations = torch.stack([torch.randperm(num_variables, device=device) for _ in range(num_samples)])
        permutations = torch.stack([permutation for _ in range(num_samples)]) # Fixed permutation across samples

        # Precompute keys and values for the history part
        key_value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
        keys_hist = [
            torch.cat(
                [mlp(key_value_input_hist)[:, None, :, :] for mlp in self.key_creators[layer]],
                axis=1,
            )
            for layer in range(self.attention_layers)
        ] # List (layers) of [bsz, heads, hist_len, attn_dim]
        values_hist = [
            torch.cat(
                [mlp(key_value_input_hist)[:, None, :, :] for mlp in self.value_creators[layer]],
                axis=1,
            )
            for layer in range(self.attention_layers)
        ] # List (layers) of [bsz, heads, hist_len, attn_dim]

        # Store keys and values for already sampled variables autoregressively
        samples = torch.zeros(num_batches, num_variables, num_samples, device=device)
        # Need shape [layers, bsz, num_samples, heads, num_variables, attn_dim] for easier indexing
        keys_samples = torch.zeros(
                self.attention_layers,
                num_batches,
                num_samples,
                self.attention_heads,
                num_variables,
                self.attention_dim,
                device=device,
            )
        values_samples = torch.zeros(
                self.attention_layers,
                num_batches,
                num_samples,
                self.attention_heads,
                num_variables,
                self.attention_dim,
                device=device,
            )

        # Autoregressive sampling loop
        for i in range(num_variables):
            # p: indices of variables being sampled at step i across all samples
            # For fixed permutation, p is just a scalar index repeated num_samples times
            p = permutations[:, i] # Shape [num_samples], contains the variable index for step i

            # Get the encoded representation for the current variable(s) across batches
            # Shape: [bsz, num_variables, input_dim] -> select p -> [bsz, input_dim]
            # Expand for num_samples: [bsz, 1, input_dim] -> [bsz, num_samples, input_dim]
            current_pred_encoded = pred_encoded[:, p[0], :].unsqueeze(1).expand(-1, num_samples, -1)

            if i == 0:
                # First variable is sampled from Uniform(0,1)
                current_samples = torch.rand(num_batches, num_samples, device=device)
            else:
                # Apply dimension shifting layer
                # Input: [bsz, num_samples, input_dim] -> Output: [bsz, num_samples, heads*attn_dim]
                att_value = self.dimension_shifting_layer(current_pred_encoded)

                # Apply attention layers
                for layer in range(self.attention_layers):
                    # Reshape query for multi-head attention
                    # Shape: [bsz, num_samples, heads, attn_dim]
                    att_value_heads = att_value.reshape(
                        num_batches,
                        num_samples,
                        self.attention_heads,
                        self.attention_dim,
                    )

                    # Prepare keys and values from history and previous samples
                    keys_hist_current_layer = keys_hist[layer] # [bsz, heads, hist_len, attn_dim]
                    # Select keys for variables sampled so far (indices 0 to i-1)
                    # Shape: [bsz, num_samples, heads, i, attn_dim]
                    keys_samples_current_layer = keys_samples[layer][:, :, :, :i, :]
                    values_hist_current_layer = values_hist[layer] # [bsz, heads, hist_len, attn_dim]
                    # Shape: [bsz, num_samples, heads, i, attn_dim]
                    values_samples_current_layer = values_samples[layer][:, :, :, :i, :]

                    # Calculate attention scores (Query * Key)
                    # Query: [bsz, num_samples, heads, attn_dim]
                    # Key_hist: [bsz, heads, hist_len, attn_dim] -> Need broadcasting/einsum
                    # Key_samples: [bsz, num_samples, heads, i, attn_dim] -> Need broadcasting/einsum

                    # Attention with history keys
                    # einsum: bnhd, bhwd -> bnhw (b=batch, n=sample, h=head, d=attn_dim, w=hist_len)
                    product_hist = torch.einsum("bnhd,bhwd->bnhw", att_value_heads, keys_hist_current_layer)

                    # Attention with previously sampled keys
                    # einsum: bnhd, bnhid -> bnhi (b=batch, n=sample, h=head, d=attn_dim, i=prev_samples)
                    product_samples = torch.einsum("bnhd,bnhid->bnhi", att_value_heads, keys_samples_current_layer)

                    # Concatenate scores: [b, n, h, hist_len + i]
                    product = torch.cat([product_hist, product_samples], dim=3)
                    product = self.attention_dim ** (-0.5) * product

                    # Apply softmax to get weights
                    weights = nn.functional.softmax(product, dim=3)
                    weights_hist = weights[:, :, :, :num_history]    # [b, n, h, hist_len]
                    weights_samples = weights[:, :, :, num_history:] # [b, n, h, i]

                    # Calculate weighted sum of values (Attention output)
                    # Value_hist: [bsz, heads, hist_len, attn_dim]
                    # Value_samples: [bsz, num_samples, heads, i, attn_dim]

                    # einsum: bnhw, bhwd -> bnhd (b=batch, n=sample, h=head, w=hist_len, d=attn_dim)
                    att_hist = torch.einsum("bnhw,bhwd->bnhd", weights_hist, values_hist_current_layer)
                    # einsum: bnhi, bnhid -> bnhd (b=batch, n=sample, h=head, i=prev_samples, d=attn_dim)
                    att_samples = torch.einsum("bnhi,bnhid->bnhd", weights_samples, values_samples_current_layer)

                    # Combine attention outputs: [bsz, num_samples, heads, attn_dim]
                    att = att_hist + att_samples

                    # Merge heads: [bsz, num_samples, heads*attn_dim]
                    att_merged_heads = att.reshape(num_batches, num_samples, self.attention_heads * self.attention_dim)

                    # Apply dropout, residual connection, layer norm, feed-forward
                    att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
                    att_value = att_value + att_merged_heads
                    att_value = self.attention_layer_norms[layer](att_value)
                    att_feed_forward = self.feed_forwards[layer](att_value)
                    att_value = att_value + att_feed_forward
                    att_value = self.feed_forward_layer_norms[layer](att_value)

                # Get logits for the categorical distribution over bins
                # Input: [bsz, num_samples, heads*attn_dim] -> Output: [bsz, num_samples, resolution]
                logits = self.dist_extractors(att_value)

                # Sample a bin index for each sample in the batch
                # Reshape logits for multinomial: [bsz * num_samples, resolution]
                probs = torch.softmax(logits.reshape(-1, self.resolution), dim=1)
                # Sample bin indices: [bsz * num_samples, 1]
                sampled_bins = torch.multinomial(probs, num_samples=1)
                # Reshape back: [bsz, num_samples]
                sampled_bins = sampled_bins.reshape(num_batches, num_samples)

                # Convert bin index to a value in [0, 1)
                current_samples = (sampled_bins.float() + torch.rand_like(sampled_bins.float())) / self.resolution

            # Store the sampled value for the current variable index p[0] (since fixed permutation)
            # samples shape: [bsz, num_variables, num_samples]
            samples[:, p[0], :] = current_samples # Store [bsz, num_samples] into slice

            # Compute and store keys/values for the *newly sampled* variable for subsequent steps
            # Input: [bsz, num_samples, input_dim] (current_pred_encoded)
            #        [bsz, num_samples] (current_samples)
            # -> [bsz, num_samples, input_dim + 1]
            key_value_input = torch.cat([current_pred_encoded, current_samples.unsqueeze(-1)], dim=-1)
            for layer in range(self.attention_layers):
                # new_keys/new_values shape: [bsz, num_samples, heads, attn_dim]
                new_keys = torch.stack(
                    [k(key_value_input) for k in self.key_creators[layer]],
                    dim=2, # Stack along head dimension
                )
                new_values = torch.stack(
                    [v(key_value_input) for v in self.value_creators[layer]],
                    dim=2, # Stack along head dimension
                )
                # Store in keys_samples/values_samples at index p[0] for the current variable
                # Target shape: [layers, bsz, num_samples, heads, num_variables, attn_dim]
                keys_samples[layer][:, :, :, p[0], :] = new_keys
                values_samples[layer][:, :, :, p[0], :] = new_values

        # Samples shape: [bsz, num_variables, num_samples]
        return samples