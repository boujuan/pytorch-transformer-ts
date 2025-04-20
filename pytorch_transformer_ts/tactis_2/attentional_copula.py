import torch
from torch import nn
import math

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
            A tensor containing an embedding for each series and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.
        pred_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.

        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the copula.
        """
        num_batches = pred_encoded.shape[0]
        num_variables = pred_encoded.shape[1]
        num_history = hist_encoded.shape[1]
        device = pred_encoded.device

        assert num_variables == num_series * num_timesteps, (
            "num_variables:"
            + str(num_variables)
            + " but num_series:"
            + str(num_series)
            + " and num_timesteps:"
            + str(num_timesteps)
        )

        permutation = torch.arange(0, num_variables).long()

        # Permute the variables according the random permutation
        pred_encoded = pred_encoded[:, permutation, :]
        pred_true_u = pred_true_u[:, permutation]

        # At the beginning of the attention, we start with the input embedding.
        # Since it does not necessarily have the same dimensions as the hidden layers, apply a linear layer to scale it up.
        att_value = self.dimension_shifting_layer(pred_encoded)

        # # The MLP which generates the keys and values used the encoded embedding + transformed true values.
        # key_value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
        # key_value_input_pred = torch.cat([pred_encoded, pred_true_u[:, :, None]], axis=2)
        # # key_value_input shape: [bsz, num_history+num_variables, embedding dimension+1]
        # key_value_input = torch.cat([key_value_input_hist, key_value_input_pred], axis=1)

        keys = []
        values = []
        for layer in range(self.attention_layers):
            key_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
            key_input_pred = torch.cat([pred_encoded, pred_true_u[:, :, None]], axis=2)
            key_input = torch.cat([key_input_hist, key_input_pred], axis=1)

            value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
            value_input_pred = torch.cat([pred_encoded, pred_true_u[:, :, None]], axis=2)
            value_input = torch.cat([value_input_hist, value_input_pred], axis=1)

            # Keys shape in every layer: [bsz, num_attention_heads, num_history+num_variables, attention_dim]
            keys.append(
                torch.cat(
                    [mlp(key_input)[:, None, :, :] for mlp in self.key_creators[layer]],
                    axis=1,
                )
            )
            # Values shape in every layer: [bsz, num_attention_heads, num_history+num_variables, attention_dim]
            values.append(
                torch.cat(
                    [mlp(value_input)[:, None, :, :] for mlp in self.value_creators[layer]],
                    axis=1,
                )
            )

        for layer in range(self.attention_layers):
            # Split the hidden layer into its various heads
            # Basically the Query in the attention
            # Shape: [bsz, num_variables, num_attention_heads, attention_dim]
            att_value_heads = att_value.reshape(
                att_value.shape[0],
                att_value.shape[1],
                self.attention_heads,
                self.attention_dim,
            )

            # Attention layer, for each batch and head:
            # A_vi' = sum_w(softmax_w(sum_i(Q_vi * K_wi) / sqrt(d)) * V_wi')

            # During attention, we will add -float("inf") to pairs of indices where the variable to be forecasted (query)
            # is after the variable that gives information (key), after the random permutation.
            # Doing this prevent information from flowing from later in the permutation to before in the permutation,
            # which cannot happen during inference.
            # tril fill the diagonal and values that are below it, flip rotates it by 180 degrees,
            # leaving only the pairs of indices which represent not yet sampled values.
            # Note float("inf") * 0 is unsafe, so do the multiplication inside the torch.tril()
            # pred/hist_encoded dimensions: number of batches, number of variables, size of embedding per variable
            product_mask = torch.ones(
                num_batches,
                self.attention_heads,
                num_variables,
                num_variables + num_history,
                device=device,
            )
            product_mask = torch.tril(float("inf") * product_mask).flip((2, 3))

            # Perform Attention
            # Einstein sum indices:
            # b: batch number
            # h: attention head number (Note the change in order for att_value_heads)
            # v: variable we want to predict
            # w: variable we want to get information from (history or prediction)
            # i: embedding dimension of the keys and queries (self.attention_dim)
            # Output shape: [bsz, num_attention_heads, num_variables, num_history+num_variables]
            product_base = torch.einsum("bvhi,bhwi->bhvw", att_value_heads, keys[layer])

            # Adding -inf shunts the attention to zero, for any variable that has not "yet" been predicted,
            # aka: are in the future according to the permutation.
            product = product_base - product_mask
            product = self.attention_dim ** (-0.5) * product
            weights = nn.functional.softmax(product, dim=-1)

            # Einstein sum indices:
            # b: batch number
            # h: attention head number (Note the change in order for the result)
            # v: variable we want to predict
            # w: variable we want to get information from (history or prediction)
            # j: embedding dimension of the values (self.attention_dim)
            # Output shape: [bsz, num_variables, num_attention_heads, attention_dim]
            att = torch.einsum("bhvw,bhwj->bvhj", weights, values[layer])

            # Merge back the various heads to allow the feed forwards module to share information between heads
            # Shape: [b, v, h*j]
            att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])
            # print("Q:", att_value_heads.shape, "K:", keys[layer].shape, "V:", values[layer].shape, "Mask:", product_mask.shape)
            # print("Attn:", att_merged_heads.shape, "Attn Scores", weights.shape)

            # Compute and add dropout
            att_merged_heads = self.attention_dropouts[layer](att_merged_heads)

            # att_value = att_value + att_merged_heads
            # Layernorm
            att_value = att_value + att_merged_heads
            att_value = self.attention_layer_norms[layer](att_value)

            # Add the contribution of the feed-forward layer, mixing up the heads
            att_feed_forward = self.feed_forwards[layer](att_value)
            att_value = att_value + att_feed_forward
            att_value = self.feed_forward_layer_norms[layer](att_value)

        # Assign each observed U(0,1) value to a bin. The clip is to avoid issues with numerical inaccuracies.
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