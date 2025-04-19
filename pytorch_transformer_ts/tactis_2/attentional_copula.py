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