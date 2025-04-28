import logging
from typing import Optional, Dict, Any, Tuple
import torch
import math
from torch import nn

from .positional_encoding import PositionalEncoding
from .temporal_encoder import TemporalEncoder
# Import the new decoder which contains marginal and copula
from .decoder import CopulaDecoder

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
        stage: int = 1,
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
        self.stage = stage
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

        # Initialize copula components if:
        # 1. Not skipping (skip_copula is False)
        # 2. In stage 2
        # 3. Args are provided
        if (not skip_copula or stage == 2) and self.copula_decoder_args is not None:
             self._initialize_copula_components()
        elif (not skip_copula or stage == 2) and self.copula_decoder_args is None:
             logger.warning("Copula components requested, but copula_decoder args were not provided. Copula components not initialized.")

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

        # --- Initialize Decoder (which contains Marginal and Copula) ---
        if self.copula_decoder_args is None:
            raise ValueError("copula_decoder configuration is required.")

        # Determine initial copula_input_dim based on skip_copula flag
        copula_input_dim_init = None
        if not self.skip_copula and self.copula_encoder_args:
             copula_input_dim_init = self.copula_encoder_args["d_model"]
        elif not self.skip_copula and self.copula_temporal_encoder_args:
             copula_input_dim_init = self.copula_temporal_encoder_args["d_model"]


        self.decoder = CopulaDecoder(
             flow_input_dim=flow_d_model,
             copula_input_dim=copula_input_dim_init, # Pass initial dim or None
             dsf_marginal=self.copula_decoder_args["dsf_marginal"],
             attentional_copula=self.copula_decoder_args.get("attentional_copula"), # Pass None if not present
             min_u=self.copula_decoder_args.get("min_u", 0.0),
             max_u=self.copula_decoder_args.get("max_u", 1.0),
             skip_sampling_marginal=self.copula_decoder_args.get("skip_sampling_marginal", False),
             skip_copula=self.skip_copula # Pass initial skip_copula state
        )
        logger.debug(f"Initialized CopulaDecoder.")

    def _initialize_copula_components(self):
        """Initialize copula-related components using stored args. Called during __init__ or set_stage."""
        # Check if already initialized (e.g., if skip_copula was False initially)
        if hasattr(self, 'copula_series_encoder') and self.copula_series_encoder is not None:
            logger.debug("Copula components appear to be already initialized.")
            return

        # Ensure required args are present
        if self.copula_encoder_args is None:
             logger.error("Cannot initialize copula components: copula_encoder_args missing.")
             return
        if self.copula_decoder_args is None or "attentional_copula" not in self.copula_decoder_args:
             logger.error("Cannot initialize copula components: attentional_copula args missing.")
             return

        # Determine device from an existing parameter
        device = self.flow_series_encoder.weight.device
        logger.debug(f"Initializing copula components on device: {device}")

        # Series encoder
        copula_series_encoder_instance = nn.Embedding(self.num_series, self.copula_series_embedding_dim)
        self.copula_series_encoder = copula_series_encoder_instance.to(device)

        # Input encoder
        copula_d_model = self.copula_encoder_args["d_model"]
        copula_input_dim = self.copula_series_embedding_dim + 2 # value + mask + embedding
        copula_encoder_layers_list = [nn.Linear(copula_input_dim, copula_d_model), nn.ReLU()]
        for _ in range(1, self.copula_input_encoder_layers):
             copula_encoder_layers_list += [nn.Linear(copula_d_model, copula_d_model), nn.ReLU()]
        if self.copula_input_encoder_layers > 1:
             copula_encoder_layers_list[-2] = nn.Linear(copula_d_model, copula_d_model)
             copula_encoder_layers_list.pop()
        copula_input_encoder_instance = nn.Sequential(*copula_encoder_layers_list)
        self.copula_input_encoder = copula_input_encoder_instance.to(device)
        logger.debug(f"Initialized copula_input_encoder with {self.copula_input_encoder_layers} layers, input_dim={copula_input_dim}, output_dim={copula_d_model}")

        # Positional encoding
        copula_time_encoding_instance = PositionalEncoding(
             embedding_dim=copula_d_model,
             dropout=self.positional_encoding_args.get("dropout", 0.1),
             max_len=self.positional_encoding_args.get("max_len", 5000)
        )
        self.copula_time_encoding = copula_time_encoding_instance.to(device)
        logger.debug(f"Initialized copula_time_encoding with embedding_dim={copula_d_model}")
        self.copula_pos_adjust = nn.Identity() # Keep identity for consistency

        # Copula encoder
        if self.copula_encoder_args["d_model"] != copula_d_model:
              logger.warning(f"copula_encoder_args d_model mismatch. Using calculated: {copula_d_model}")

        if self.encoder_type == "standard":
            logger.debug(f"Initializing standard copula_encoder (Transformer)")
            copula_encoder_instance = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=copula_d_model,
                    nhead=self.copula_encoder_args["nhead"],
                    dim_feedforward=self.copula_encoder_args.get("dim_feedforward", copula_d_model * 4),
                    dropout=self.copula_encoder_args.get("dropout", 0.1),
                    batch_first=True
                ),
                num_layers=self.copula_encoder_args["num_encoder_layers"]
            )
            self.copula_encoder = copula_encoder_instance.to(device)
        elif self.encoder_type == "temporal":
            logger.debug(f"Initializing temporal copula_encoder")
            copula_encoder_instance = TemporalEncoder(
                d_model=copula_d_model,
                nhead=self.copula_temporal_encoder_args["nhead"],
                num_encoder_layers=self.copula_temporal_encoder_args["num_encoder_layers"],
                dim_feedforward=self.copula_temporal_encoder_args.get("dim_feedforward", copula_d_model * 4),
                dropout=self.copula_temporal_encoder_args.get("dropout", 0.1),
            )
            self.copula_encoder = copula_encoder_instance.to(device)
        else:
            raise ValueError(f"Unknown encoder_type for copula: {self.encoder_type}")

        # Initialize Attentional Copula within the Decoder *after* other components are on device
        # The decoder's __init__ handles creating the AttentionalCopula instance
        # We just need to ensure the decoder itself is updated if needed
        if hasattr(self.decoder, 'create_attentional_copula'):
             # Update decoder's internal state and potentially create the copula instance
             self.decoder.copula_input_dim = copula_d_model # Ensure decoder knows the dim
             self.decoder.attentional_copula_args = self.copula_decoder_args["attentional_copula"]
             self.decoder.create_attentional_copula() # This method should handle device placement internally if needed, or rely on the decoder being on the correct device.
             logger.debug("Called decoder.create_attentional_copula()")
        else:
             logger.error("Decoder object does not have create_attentional_copula method.")

        # Add debugging log to check parameter registration
        logger.debug("Checking parameters within TACTiS after copula initialization:")
        found_copula_params = False
        # Use self.named_parameters() which should include parameters from all submodules
        for name, param in self.named_parameters():
            # Check prefixes for copula components directly under self or within the decoder
            if name.startswith("copula_") or name.startswith("decoder.copula"):
                logger.debug(f"  Found registered copula parameter: {name} on device {param.device}")
                found_copula_params = True
        if not found_copula_params:
            logger.warning("  No parameters starting with 'copula_' or 'decoder.copula' found registered within TACTiS module after initialization!")

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

        # Encode for copula if needed
        copula_encoded = None
        if not self.skip_copula and self.stage >= 2:
            if copula_series_emb is not None:
                 copula_encoded = self._encode_copula(time_steps, value, mask, series_indices, copula_series_emb)
            else:
                 logger.warning("Stage >= 2 but copula components not initialized. Skipping copula encoding.")

        # --- Loss Calculation via Decoder ---
        # Pass encoded representations, mask, and true values to the decoder
        marginal_logdet_batch, copula_loss_batch = self.decoder.loss(
            flow_encoded=flow_encoded,
            copula_encoded=copula_encoded, # Will be None if skipped
            mask=mask,
            true_value=true_value,
        ) # Returns loss components per batch item [B]

        # Store mean values for logging
        self.marginal_logdet = marginal_logdet_batch.mean()
        self.copula_loss = copula_loss_batch.mean() # Will be 0 if copula was skipped

        # --- Apply Loss Normalization ---
        # Based on original TACTiS logic
        if self.loss_normalization in {"series", "both"}:
            marginal_logdet_batch = marginal_logdet_batch / num_series
            copula_loss_batch = copula_loss_batch / num_series
        if self.loss_normalization in {"timesteps", "both"}:
            marginal_logdet_batch = marginal_logdet_batch / pred_len
            copula_loss_batch = copula_loss_batch / pred_len

        # --- Combine Normalized Losses ---
        # Total loss per batch item = Copula NLL - Marginal LogDet
        loss_batch = copula_loss_batch - marginal_logdet_batch # Shape [B]

        # Final scalar loss is the mean over the batch
        loss = loss_batch.mean()

        # Return output (pred_value for training) and loss
        output = pred_value # For training, return target as "output"

        return output, loss

    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor, # Expected to be normalized externally [batch, series, hist_len]
        pred_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples from the model using the CopulaDecoder.
        
        Parameters:
        -----------
        num_samples: int
            Number of sample paths to generate
        hist_time: torch.Tensor [batch, hist_len]
            Time values for historical data
        hist_value: torch.Tensor [batch, series, hist_len]
            Historical values (normalized externally)
        pred_time: torch.Tensor [batch, pred_len]
            Time values for prediction period
            
        Returns:
        --------
        torch.Tensor [num_samples, batch, series, pred_len]
            Normalized samples for the prediction period.
            Each sample represents one possible future trajectory.
            This shape is compatible with module.py which will permute it to
            [batch, num_samples, pred_len, series] for GluonTS compatibility.
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

        # Create series embeddings
        flow_series_emb = self.flow_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1)
        copula_series_emb = None
        if not self.skip_copula and self.stage >= 2 and hasattr(self, 'copula_series_encoder'):
             copula_series_emb = self.copula_series_encoder(series_indices).unsqueeze(0).expand(batch_size, -1, -1)

        # Encode inputs
        flow_encoded = self._encode_flow(time_steps, value_placeholder, mask, series_indices, flow_series_emb)
        copula_encoded = None
        if copula_series_emb is not None:
             copula_encoded = self._encode_copula(time_steps, value_placeholder, mask, series_indices, copula_series_emb)

        # Delegate sampling to the decoder
        # Decoder expects full time range for encoded, mask, and true_value (with history)
        samples_normalized = self.decoder.sample(
            num_samples=num_samples,
            flow_encoded=flow_encoded,       # [B, S, T_total, D_flow]
            copula_encoded=copula_encoded,   # [B, S, T_total, D_copula] or None
            mask=mask,                       # [B, S, T_total]
            true_value=value_placeholder,    # [B, S, T_total] (contains history)
        ) # Output shape: [num_samples, B, S, T_total]

        # Extract only the prediction part
        samples_pred = samples_normalized[..., :, :, hist_len:] # Shape [num_samples, B, S, pred_len]

        # Return normalized samples with shape [num_samples, batch, series, pred_len]
        # Module.py will handle permuting this to [batch, num_samples, pred_len, series] for GluonTS
        return samples_pred

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
        # TODO JUAN dim of value is (n_turbines, 2, 30), of time_steps is (n_tubines, 50)
        batch_size, num_series, num_timesteps = value.shape # Get dimensions from value tensor
        # device = time_steps.device

        # flow_series_emb is now passed in, shape [batch, series, dim]
        # Expand series embedding to match time dimension
        series_emb_expanded = flow_series_emb.unsqueeze(2).expand(-1, -1, num_timesteps, -1) # [batch, series, time, dim]

        # Prepare input tensor - permute value and mask to [batch, time, series, 1] first
        # Concatenate: value, series_embedding, mask
        flow_input = torch.cat([
            value.permute(0, 2, 1).unsqueeze(-1),          # [batch, time, series, 1]
            series_emb_expanded.permute(0, 2, 1, 3),       # [batch, time, series, dim]
            mask.permute(0, 2, 1).unsqueeze(-1).float(),   # [batch, time, series, 1]
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
        pos_encoded = self.flow_time_encoding(encoded, time_tensor) # Shape: [batch, time, series, embed_dim]
        encoded = pos_encoded # Use pos_encoded directly

        # Apply flow encoder (handles standard vs temporal internally now)
        if isinstance(self.flow_encoder, nn.TransformerEncoder):
            # For standard Transformer: reshape [batch, time, series, embed_dim] -> [batch, time*series, embed_dim]
            encoded_reshaped = encoded.reshape(batch_size, num_timesteps * num_series, embed_dim)
            encoded_out = self.flow_encoder(encoded_reshaped)
            # Reshape back: [batch, time*series, embed_dim] -> [batch, time, series, embed_dim]
            encoded = encoded_out.reshape(batch_size, num_timesteps, num_series, embed_dim)
        elif isinstance(self.flow_encoder, TemporalEncoder):
            # For TemporalEncoder: permute [batch, time, series, embed_dim] -> [batch, series, time, embed_dim]
            encoded = encoded.permute(0, 2, 1, 3)
            encoded = self.flow_encoder(encoded) # Output shape: [batch, series, time, embed_dim]
            # No need to permute back as we'll do final permute below
        else:
            raise TypeError(f"Unexpected flow_encoder type: {type(self.flow_encoder)}")

        # Ensure final shape is [batch, series, time, dim]
        if isinstance(self.flow_encoder, nn.TransformerEncoder):
            # For standard Transformer: current shape is [batch, time, series, embed_dim]
            encoded = encoded.permute(0, 2, 1, 3)
        # For TemporalEncoder: already in [batch, series, time, embed_dim]

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
            mask.permute(0, 2, 1).unsqueeze(-1).float(),   # [batch, time, series, 1]
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
        encoded = pos_encoded # Use pos_encoded directly

        # Apply copula encoder (handles standard vs temporal internally now)
        if isinstance(self.copula_encoder, nn.TransformerEncoder):
            # For standard Transformer: reshape [batch, time, series, embed_dim] -> [batch, time*series, embed_dim]
            encoded_reshaped = encoded.reshape(batch_size, num_timesteps * num_series, embed_dim)
            encoded_out = self.copula_encoder(encoded_reshaped)
            # Reshape back: [batch, time*series, embed_dim] -> [batch, time, series, embed_dim]
            encoded = encoded_out.reshape(batch_size, num_timesteps, num_series, embed_dim)
        elif isinstance(self.copula_encoder, TemporalEncoder):
            # For TemporalEncoder: permute [batch, time, series, embed_dim] -> [batch, series, time, embed_dim]
            encoded = encoded.permute(0, 2, 1, 3)
            encoded = self.copula_encoder(encoded) # Output shape: [batch, series, time, embed_dim]
            # No need to permute back as we'll do final permute below
        else:
            raise TypeError(f"Unexpected copula_encoder type: {type(self.copula_encoder)}")

        # Ensure final shape is [batch, series, time, dim]
        if isinstance(self.copula_encoder, nn.TransformerEncoder):
            # For standard Transformer: current shape is [batch, time, series, embed_dim]
            encoded = encoded.permute(0, 2, 1, 3)
        # For TemporalEncoder: already in [batch, series, time, embed_dim]

        return encoded.contiguous()

    def set_stage(self, stage: int):
        """Set the training stage."""
        assert stage in [1, 2], "Stage must be 1 or 2"
        logger.info(f"TACTiS model: Setting stage to {stage}")
        self.stage = stage

        # Update skip_copula flag based on the new stage
        self.skip_copula = (stage == 1)
        if hasattr(self.decoder, 'skip_copula'):
             self.decoder.skip_copula = self.skip_copula

        # We don't need to initialize components here anymore as they should already
        # be initialized properly based on the stage parameter in __init__
        # Just update the state in the decoder if it's stage 2
        if stage == 2:
             # Ensure copula components are initialized when moving to stage 2
             logger.info("Ensuring all copula components are initialized for Stage 2...")
             self._initialize_copula_components()