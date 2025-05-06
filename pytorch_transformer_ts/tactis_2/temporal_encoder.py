import torch
from torch import nn

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
        activation: nn.Module = nn.ReLU,
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
        self.activation = activation
        self.total_attention_time = 0.0 # Keep for potential timing analysis

        # Create pairs of layers: one for time attention, one for series attention
        self.layer_timesteps = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation(),  # Use the provided activation function
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
                    activation=self.activation(),  # Use the provided activation function
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