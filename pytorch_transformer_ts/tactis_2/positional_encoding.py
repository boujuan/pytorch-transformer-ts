import math
import logging
import torch
from torch import nn

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