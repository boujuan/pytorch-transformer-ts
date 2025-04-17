import torch
from torch import nn

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