import logging
from typing import Optional, Dict, Any, Tuple
import torch
import math
from torch import nn

from .positional_encoding import PositionalEncoding
from .temporal_encoder import TemporalEncoder
from .dsf_marginal import DSFMarginal
from .attentional_copula import AttentionalCopula

# Set up logging
logger = logging.getLogger(__name__)

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
        flow_temporal_encoder: Optional[Dict[str, Any]] = None, # Kept for potential future use
        copula_temporal_encoder: Optional[Dict[str, Any]] = None, # Kept for potential future use
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
            Arguments for flow encoder (standard Transformer or TemporalEncoder)
        copula_encoder: Optional[Dict]
            Arguments for copula encoder (standard Transformer or TemporalEncoder)
        flow_temporal_encoder: Optional[Dict]
            Arguments for flow temporal encoder (currently unused if encoder_type='temporal')
        copula_temporal_encoder: Optional[Dict]
            Arguments for copula temporal encoder (currently unused if encoder_type='temporal')
        copula_decoder: Optional[Dict]
            Arguments for copula decoder (contains DSF and AttentionalCopula args)
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
        # Store temporal encoder args separately, even if unused by standard encoder
        self.flow_temporal_encoder_args = flow_temporal_encoder if flow_temporal_encoder else flow_encoder
        self.copula_temporal_encoder_args = copula_temporal_encoder if copula_temporal_encoder else copula_encoder
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
        flow_input_dim = self.flow_series_embedding_dim + 2 # value + mask + embedding
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

        # Positional encoding - Use flow_d_model
        self.flow_time_encoding = PositionalEncoding(
             embedding_dim=flow_d_model, # Use flow_d_model directly
             dropout=self.positional_encoding_args.get("dropout", 0.1),
             max_len=self.positional_encoding_args.get("max_len", 5000)
        )
        logger.debug(f"Initialized flow_time_encoding with embedding_dim={flow_d_model}")

        # Remove flow_pos_adjust as dimensions should now match
        self.flow_pos_adjust = nn.Identity()

        # Flow encoder (Transformer or TemporalEncoder) - Use parameters from flow_encoder_args
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
            logger.debug(f"Initializing temporal flow_encoder with {self.flow_temporal_encoder_args['num_encoder_layers']} layers, d_model={flow_d_model}, nhead={self.flow_temporal_encoder_args['nhead']}")
            # Use flow_temporal_encoder_args here
            self.flow_encoder = TemporalEncoder(
                d_model=flow_d_model,
                nhead=self.flow_temporal_encoder_args["nhead"],
                num_encoder_layers=self.flow_temporal_encoder_args["num_encoder_layers"],
                dim_feedforward=self.flow_temporal_encoder_args.get("dim_feedforward", flow_d_model * 4),
                dropout=self.flow_temporal_encoder_args.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown encoder_type for flow: {self.encoder_type}")

        # Marginals (DSF) - Use parameters from copula_decoder_args['dsf_marginal']
        if self.copula_decoder_args is None or "dsf_marginal" not in self.copula_decoder_args:
             raise ValueError("DSF Marginal arguments ('dsf_marginal') missing in copula_decoder config.")
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
        copula_input_dim = self.copula_series_embedding_dim + 2 # value + mask + embedding
        copula_encoder_layers_list = [nn.Linear(copula_input_dim, copula_d_model), nn.ReLU()] # First layer
        for _ in range(1, self.copula_input_encoder_layers):
             copula_encoder_layers_list += [nn.Linear(copula_d_model, copula_d_model), nn.ReLU()]
        # Last layer ensures output matches d_model without ReLU
        if self.copula_input_encoder_layers > 1:
             copula_encoder_layers_list[-2] = nn.Linear(copula_d_model, copula_d_model) # Replace last Linear
             copula_encoder_layers_list.pop() # Remove last ReLU

        self.copula_input_encoder = nn.Sequential(*copula_encoder_layers_list)
        logger.debug(f"Initialized copula_input_encoder with {self.copula_input_encoder_layers} layers, input_dim={copula_input_dim}, output_dim={copula_d_model}")

        # Positional encoding - Use copula_d_model
        self.copula_time_encoding = PositionalEncoding(
             embedding_dim=copula_d_model, # Use copula_d_model directly
             dropout=self.positional_encoding_args.get("dropout", 0.1),
             max_len=self.positional_encoding_args.get("max_len", 5000)
        )
        logger.debug(f"Initialized copula_time_encoding with embedding_dim={copula_d_model}")

        # Remove copula_pos_adjust as dimensions should now match
        self.copula_pos_adjust = nn.Identity()

        # Copula encoder (Transformer or TemporalEncoder) - Use parameters from copula_encoder_args
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
            logger.debug(f"Initializing temporal copula_encoder with {self.copula_temporal_encoder_args['num_encoder_layers']} layers, d_model={copula_d_model}, nhead={self.copula_temporal_encoder_args['nhead']}")
            # Use copula_temporal_encoder_args here
            self.copula_encoder = TemporalEncoder(
                d_model=copula_d_model,
                nhead=self.copula_temporal_encoder_args["nhead"],
                num_encoder_layers=self.copula_temporal_encoder_args["num_encoder_layers"],
                dim_feedforward=self.copula_temporal_encoder_args.get("dim_feedforward", copula_d_model * 4),
                dropout=self.copula_temporal_encoder_args.get("dropout", 0.1),
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
        Forward pass for training or inference.

        Parameters:
        -----------
        hist_time: Time values for historical data [batch, hist_len]
        hist_value: Values for historical data [batch, series, hist_len] (normalized externally)
        pred_time: Time values for prediction period [batch, pred_len]
        pred_value: Target values for prediction [batch, series, pred_len] (training only, normalized externally)

        Returns:
        --------
        Tuple of (output, loss)
        - output: Predictions [batch, series, pred_len] (normalized) during inference, or pred_value during training.
        - loss: Computed loss tensor during training, None during inference.
        """
        if pred_value is None:
             # Inference mode: call sample method
             # Assumes hist_value is already normalized externally
             # Denormalization is handled by TACTiS2Model wrapper
             norm_samples = self.sample(1, hist_time, hist_value, pred_time) # Generate 1 sample path for typical inference
             # Squeeze the sample dimension for standard predictor output
             return norm_samples.squeeze(0), None # Loss is None during inference

        # --- Training forward pass ---
        batch_size, num_series, hist_len = hist_value.shape
        pred_len = pred_value.shape[2]
        device = hist_value.device

        # Prepare inputs (assuming external normalization)
        time_steps = torch.cat([hist_time, pred_time], dim=1) # [batch, hist_len + pred_len]
        value = torch.cat([hist_value, pred_value], dim=2) # [batch, series, hist_len + pred_len]
        mask = torch.cat([ # Mask True = observed, False = predicted
            torch.ones(batch_size, num_series, hist_len, dtype=torch.bool, device=device),
            torch.zeros(batch_size, num_series, pred_len, dtype=torch.bool, device=device)
        ], dim=2) # [batch, series, hist_len + pred_len]
        true_value = value # Use combined value for encoding & loss calculation

        # Create embeddings for series
        series_indices = torch.arange(num_series, device=device)
        # Expand over batches for potential bagging
        flow_series_emb = self.flow_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1) # [batch, series, flow_dim]
        copula_series_emb = None
        if not self.skip_copula and self.stage >= 2:
             # Ensure copula encoder exists before creating embedding
             if hasattr(self, 'copula_series_encoder'):
                 copula_series_emb = self.copula_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1) # [batch, series, copula_dim]
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
        # flow_encoded shape: [batch, series, time, dim]

        # Process with marginal model
        normalized_data = value # Shape [B, S, T]
        # DSFMarginal now expects context [B, N, D] and x [B, N] where N = S*T
        B, S, T_total, D_flow = flow_encoded.shape
        N = S * T_total
        flow_encoded_merged = flow_encoded.reshape(B, N, D_flow)
        normalized_data_merged = normalized_data.reshape(B, N)

        # Pass merged context and data to marginal flow
        u_vals_merged, marginal_logdet_batch = self.marginal.forward_logdet(flow_encoded_merged, normalized_data_merged)
        # u_vals_merged shape: [B, N]
        # marginal_logdet_batch shape: [B] (Log determinant summed over non-batch dims)

        # Marginal NLL per batch item is -marginal_logdet_batch
        # We store the mean over the batch for logging
        self.marginal_logdet = marginal_logdet_batch.mean()

        # Process with copula if needed
        # Initialize copula_loss_batch with correct shape [B] and device
        copula_loss_batch = torch.zeros(B, device=device)
        copula_loss = torch.zeros(1, device=device).mean()

        if not self.skip_copula and self.stage >= 2:
            # Encode inputs for copula using potentially bagged tensors
            # Pass the potentially bagged copula_series_emb directly
            if copula_series_emb is not None: # Check if copula embeddings exist (i.e., stage 2 and initialized)
                 copula_encoded = self._encode_copula(time_steps, value, mask, series_indices, copula_series_emb)
                 # copula_encoded shape: [batch, series, time, dim]
            else:
                 copula_encoded = None # Ensure copula_encoded is None if embeddings weren't created

            # Process with copula if encoded representation exists
            if copula_encoded is not None and hasattr(self, 'copula'):
                # AttentionalCopula log_prob expects u_vals [batch, series, time]
                # and context [batch, series, time, dim]
                # We only care about the prediction steps for the loss
                # Reshape u_vals to [B, N] for copula input if needed, or use u_vals_merged directly
                # AttentionalCopula expects u_vals [batch, series, time] -> [B, S, T_pred]
                # and context [batch, series, time, dim] -> [B, S, T_pred, D_copula]
                u_vals_pred = u_vals_merged.reshape(B, S, T_total)[:, :, hist_len:] # Shape [B, S, T_pred]
                copula_encoded_pred = copula_encoded.reshape(B, S, T_total, -1)[:, :, hist_len:] # Shape [B, S, T_pred, D_copula]

                # Calculate copula NLL (negative log probability)
                # The log_prob method should return the log probability of the copula density
                copula_log_prob = self.copula.log_prob(u_vals_pred, copula_encoded_pred)
                # Sum log probability over series and time dimensions to get per-batch copula NLL
                copula_loss_batch = -copula_log_prob.sum(dim=(1, 2)) # Shape [B]
                copula_loss = copula_loss_batch.mean() # Scalar loss for logging/reporting
            else:
                # If copula_encoded is None or copula not initialized, set loss to zero
                logger.debug("Skipping copula loss calculation (not initialized or stage 1).")
                copula_loss = torch.zeros(1, device=device).mean()
        else:
             logger.debug("Skipping copula loss calculation (skip_copula=True or stage 1).")
             copula_loss = torch.zeros(1, device=device).mean()

        # Store mean copula loss for logging/debugging
        self.copula_loss = copula_loss_batch.mean() if not self.skip_copula else torch.tensor(0.0, device=device)

        # Loss normalization is handled by original TACTiS decoder logic (summing logdets/logprobs)
        # and potential division in the final mean calculation. Remove explicit normalization here.

        # Total loss per batch item = Copula NLL - Marginal LogDet
        # Note: marginal_logdet_batch is log |det(dF/dx)|, so NLL = -log p(z) - log |det(dF/dx)|
        # Assuming p(z) is standard normal, -log p(z) is handled implicitly if copula_loss includes it,
        # or needs to be added if copula_loss is purely the copula term.
        # Original TACTiS decoder loss = copula_loss - marginal_logdet. Let's follow that.
        loss_batch = copula_loss_batch - marginal_logdet_batch # Shape [B]

        # Final scalar loss is the mean over the batch
        loss = loss_batch.mean()

        # Return output (pred_value for training) and loss
        output = pred_value # For training, return target as "output"

        return output, loss

    # Sample method simplified assuming forecasting mode and external normalization
    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor, # Expected to be normalized externally [batch, series, hist_len]
        pred_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        Returns normalized samples [num_samples, batch, series, pred_len].
        """
        batch_size, num_series, hist_len = hist_value.shape
        pred_len = pred_time.shape[1]
        device = hist_value.device

        # Prepare inputs for encoding (history + zeros for prediction period)
        time_steps = torch.cat([hist_time, pred_time], dim=1) # [batch, hist_len + pred_len]
        series_indices = torch.arange(num_series, device=device)
        value_placeholder = torch.cat([
            hist_value,
            torch.zeros(batch_size, num_series, pred_len, device=device) # Zeros for prediction period
        ], dim=2) # [batch, series, hist_len + pred_len]
        mask = torch.cat([
            torch.ones(batch_size, num_series, hist_len, dtype=torch.bool, device=device),
            torch.zeros(batch_size, num_series, pred_len, dtype=torch.bool, device=device)
        ], dim=2) # [batch, series, hist_len + pred_len]

        # Create series embeddings (needed for encoding)
        # Expand over batches as encoding expects batch dimension
        flow_series_emb = self.flow_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1) # [batch, series, flow_dim]
        copula_series_emb = None
        if not self.skip_copula and self.stage >= 2 and hasattr(self, 'copula_series_encoder'):
             copula_series_emb = self.copula_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1) # [batch, series, copula_dim]

        # Encode inputs
        # Pass the created series embeddings directly
        flow_encoded = self._encode_flow(time_steps, value_placeholder, mask, series_indices, flow_series_emb) # [batch, series, time, flow_dim]
        copula_encoded = None
        if copula_series_emb is not None: # Check if copula embeddings exist
             copula_encoded = self._encode_copula(time_steps, value_placeholder, mask, series_indices, copula_series_emb) # [batch, series, time, copula_dim]

        # Generate uniform samples or copula samples
        if copula_encoded is None or not hasattr(self, 'copula'): # Stage 1 or copula skipped/not initialized
            logger.debug("Sampling: Generating uniform samples (Stage 1 or Copula skipped/uninitialized).")
            # Generate uniform samples in shape [num_samples, batch, series, pred_len]
            u_samples = torch.rand(num_samples, batch_size, num_series, pred_len, device=device)
        else:
            logger.debug("Sampling: Generating samples from Copula (Stage 2).")
            # Sample from copula using encoded representation of prediction steps
            # Select the copula encoding corresponding to prediction steps
            copula_encoded_pred = copula_encoded[:, :, hist_len:] # Shape [batch, series, pred_len, dim]

            # AttentionalCopula.sample expects flow_encoded [batch, series * pred_len, dim]
            # Reshape context for copula sampling
            copula_encoded_pred_flat = copula_encoded_pred.reshape(batch_size, num_series * pred_len, -1)

            # Sample from the copula
            # Output shape: [batch, series*pred_len, num_samples]
            u_samples_flat = self.copula.sample(
                num_samples,
                copula_encoded_pred_flat, # Pass flattened copula encoding for prediction steps
                copula_encoded_pred_flat.shape[-1] # Pass embedding dim
            )
            # Reshape back to [num_samples, batch, series, pred_len]
            try:
                 # Reshape: [batch, series*pred_len, num_samples] -> [batch, series, pred_len, num_samples]
                 u_samples_reshaped = u_samples_flat.reshape(batch_size, num_series, pred_len, num_samples)
                 # Permute: [batch, series, pred_len, num_samples] -> [num_samples, batch, series, pred_len]
                 u_samples = u_samples_reshaped.permute(3, 0, 1, 2)
            except RuntimeError as e:
                 logger.error(f"Error reshaping copula samples: {e}. Input shape: {u_samples_flat.shape}. Falling back to uniform samples.")
                 # Fallback: generate uniform samples
                 u_samples = torch.rand(num_samples, batch_size, num_series, pred_len, device=device)

        # Transform samples using marginal inverse CDF
        # Select flow encoding for prediction steps
        flow_encoded_pred = flow_encoded[:, :, hist_len:] # Shape [batch, series, pred_len, dim]

        # Reshape u_samples and flow_encoded_pred for marginal.inverse
        # u_samples: [num_samples, batch, series, pred_len] -> [batch, series, pred_len, num_samples]
        u_samples_permuted = u_samples.permute(1, 2, 3, 0) # Shape [B, S, pred_len, num_samples]

        # marginal.inverse expects context [batch, N, dim] and u [batch, N, num_samples]
        # where N = series * pred_len
        N_pred = num_series * pred_len
        # Reshape context: [batch, series, pred_len, dim] -> [batch, N_pred, dim]
        flow_encoded_pred_merged = flow_encoded_pred.reshape(batch_size, N_pred, -1)
        # Reshape u: [batch, series, pred_len, num_samples] -> [batch, N_pred, num_samples]
        u_samples_merged = u_samples_permuted.reshape(batch_size, N_pred, num_samples)

        # Apply inverse transform (inverse CDF)
        # Output shape: [batch, N_pred, num_samples]
        samples_normalized_merged = self.marginal.inverse(flow_encoded_pred_merged, u_samples_merged)

        # Reshape flat output for consistency before final reshape
        # [batch, N_pred, num_samples] -> [batch * N_pred, num_samples]
        samples_normalized_flat = samples_normalized_merged.reshape(-1, num_samples)

        # Reshape back: [batch * series * pred_len, num_samples] -> [batch, series, pred_len, num_samples]
        samples_normalized = samples_normalized_flat.reshape(batch_size, num_series, pred_len, num_samples)

        # Permute to final shape: [num_samples, batch, series, pred_len]
        samples = samples_normalized.permute(3, 0, 1, 2)

        # Return normalized samples; denormalization is external
        return samples

    def _encode_flow(self, time_steps, value, mask, series_indices, flow_series_emb):
        """
        Encode data for flow component, accepting pre-computed embeddings.
        Input shapes:
            time_steps: [batch, time]
            value: [batch, series, time]
            mask: [batch, series, time]
            series_indices: [series] (used only if flow_series_emb is None)
            flow_series_emb: [batch, series, dim]
        Output shape: [batch, series, time, embed_dim]
        """
        batch_size, num_series, num_timesteps = value.shape # Get dimensions from value tensor
        device = time_steps.device

        # flow_series_emb is now passed in, shape [batch, series, dim]
        # Expand series embedding to match time dimension
        series_emb_expanded = flow_series_emb.unsqueeze(2).expand(-1, -1, num_timesteps, -1) # [batch, series, time, dim]

        # Prepare input tensor - permute value and mask to [batch, time, series, 1] first
        # Concatenate: value, series_embedding, mask
        flow_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),          # [batch, time, series, 1]
            series_emb_expanded.permute(0, 2, 1, 3),       # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1) # Shape: [batch, time, series, 1 + dim + 1]

        # Reshape for input encoder [batch * time * series, features]
        input_encoder_input_dim = flow_input.shape[-1]
        flow_input = flow_input.reshape(-1, input_encoder_input_dim)

        # Apply input encoder
        encoded = self.flow_input_encoder(flow_input) # Shape: [batch * time * series, embed_dim]
        embed_dim = encoded.shape[-1]
        # Reshape back [batch, time, series, embed_dim]
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, embed_dim)

        # Apply positional encoding
        # time_steps shape [batch, time] -> unsqueeze for pos encoding [batch, time, 1]
        time_tensor = time_steps.unsqueeze(-1)
        # encoded shape [batch, time, series, embed_dim]
        # PositionalEncoding handles broadcasting/expanding time encoding to match series dim
        pos_encoded = self.flow_time_encoding(encoded, time_tensor) # Shape: [batch, time, series, embed_dim]
        # encoded = self.flow_pos_adjust(pos_encoded) # Adjustment layer removed
        encoded = pos_encoded # Use pos_encoded directly

        # Apply flow encoder (handles standard vs temporal internally now)
        # Standard Transformer expects [batch, seq_len, embed_dim] where seq_len = time * series
        # TemporalEncoder expects [batch, series, time, embed_dim]
        if isinstance(self.flow_encoder, nn.TransformerEncoder):
            # Reshape for standard Transformer: [batch, time, series, embed_dim] -> [batch, time * series, embed_dim]
            encoded_reshaped = encoded.reshape(batch_size, num_timesteps * num_series, embed_dim)
            encoded_out = self.flow_encoder(encoded_reshaped)
            # Reshape back: [batch, time * series, embed_dim] -> [batch, time, series, embed_dim]
            encoded = encoded_out.reshape(batch_size, num_timesteps, num_series, embed_dim)
        elif isinstance(self.flow_encoder, TemporalEncoder):
            # TemporalEncoder expects [batch, series, time, embed_dim]
            # Permute current shape [batch, time, series, embed_dim] -> [batch, series, time, embed_dim]
            encoded_permuted = encoded.permute(0, 2, 1, 3)
            encoded = self.flow_encoder(encoded_permuted) # Output shape: [batch, series, time, embed_dim]
            # Permute back to [batch, time, series, embed_dim] for consistency before final permute
            encoded = encoded.permute(0, 2, 1, 3)
        else:
            raise TypeError(f"Unexpected flow_encoder type: {type(self.flow_encoder)}")

        # Final reshape and permute to [batch, series, time, dim]
        encoded = encoded.permute(0, 2, 1, 3)

        return encoded.contiguous()

    def _encode_copula(self, time_steps, value, mask, series_indices, copula_series_emb):
        """
        Encode data for copula component, accepting pre-computed embeddings.
        Input shapes:
            time_steps: [batch, time]
            value: [batch, series, time]
            mask: [batch, series, time]
            series_indices: [series] (used only if copula_series_emb is None)
            copula_series_emb: [batch, series, dim]
        Output shape: [batch, series, time, embed_dim]
        """
        batch_size, num_series, num_timesteps = value.shape
        device = time_steps.device

        # copula_series_emb is passed in, shape [batch, series, dim]
        # Expand series embedding to match time dimension
        series_emb_expanded = copula_series_emb.unsqueeze(2).expand(-1, -1, num_timesteps, -1) # [batch, series, time, dim]

        # Prepare input tensor - permute value and mask to [batch, time, series, 1] first
        copula_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),          # [batch, time, series, 1]
            series_emb_expanded.permute(0, 2, 1, 3),       # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),  # [batch, time, series, 1]
        ], dim=-1) # Shape: [batch, time, series, 1 + dim + 1]

        # Reshape for input encoder [batch * time * series, features]
        input_encoder_input_dim = copula_input.shape[-1]
        copula_input = copula_input.reshape(-1, input_encoder_input_dim)
        # Apply input encoder
        encoded = self.copula_input_encoder(copula_input) # Shape: [batch * time * series, embed_dim]
        embed_dim = encoded.shape[-1]

        # Reshape back [batch, time, series, embed_dim]
        encoded = encoded.reshape(batch_size, num_timesteps, num_series, embed_dim)

        # Apply positional encoding
        time_tensor = time_steps.unsqueeze(-1) # [batch, time, 1]
        pos_encoded = self.copula_time_encoding(encoded, time_tensor) # Shape: [batch, time, series, embed_dim]
        # encoded = self.copula_pos_adjust(pos_encoded) # Adjustment layer removed
        encoded = pos_encoded # Use pos_encoded directly

        # Apply copula encoder (handles standard vs temporal internally now)
        if isinstance(self.copula_encoder, nn.TransformerEncoder):
            # Reshape for standard Transformer: [batch, time, series, embed_dim] -> [batch, time * series, embed_dim]
            encoded_reshaped = encoded.reshape(batch_size, num_timesteps * num_series, embed_dim)
            encoded_out = self.copula_encoder(encoded_reshaped)
            # Reshape back: [batch, time * series, embed_dim] -> [batch, time, series, embed_dim]
            encoded = encoded_out.reshape(batch_size, num_timesteps, num_series, embed_dim)
        elif isinstance(self.copula_encoder, TemporalEncoder):
            # TemporalEncoder expects [batch, series, time, embed_dim]
            # Permute current shape [batch, time, series, embed_dim] -> [batch, series, time, embed_dim]
            encoded_permuted = encoded.permute(0, 2, 1, 3)
            encoded = self.copula_encoder(encoded_permuted) # Output shape: [batch, series, time, embed_dim]
            # Permute back to [batch, time, series, embed_dim] for consistency before final permute
            encoded = encoded.permute(0, 2, 1, 3)
        else:
            raise TypeError(f"Unexpected copula_encoder type: {type(self.copula_encoder)}")

        # Final reshape and permute to [batch, series, time, dim]
        encoded = encoded.permute(0, 2, 1, 3)

        return encoded.contiguous()

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