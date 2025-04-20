import math
import logging
from typing import Any, Dict, Optional, Type, Tuple
import torch
from torch import nn

# Assuming these are in the same directory or correctly importable
from .dsf_marginal import DSFMarginal
from .attentional_copula import AttentionalCopula

logger = logging.getLogger(__name__)

# Helper functions (copied from original decoder.py for self-containment)
def _merge_series_time_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series, time steps, ...] to one with dimensions [batch, series * time steps, ...]
    """
    if x is None: return None # Handle None case
    assert x.dim() >= 3
    return x.view((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])

def _split_series_time_dims(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series * time steps, ...] to one with dimensions [batch, series, time steps, ...]
    """
    if x is None: return None # Handle None case
    # Adjust assertion for sample dimension potentially being last
    assert x.dim() == len(target_shape) or x.dim() + 1 == len(target_shape)
    return x.view(target_shape)


class CopulaDecoder(nn.Module):
    """
    A decoder which forecast using a distribution built from a copula and marginal distributions.
    Adapted from original TACTiS implementation.
    """

    def __init__(
        self,
        flow_input_dim: int,
        copula_input_dim: Optional[int], # Can be None if skip_copula=True initially
        dsf_marginal: Dict[str, Any],
        attentional_copula: Optional[Dict[str, Any]] = None,
        min_u: float = 0.0,
        max_u: float = 1.0,
        skip_sampling_marginal: bool = False,
        skip_copula: bool = True, # Default to skipping copula initially
    ):
        """
        Parameters:
        -----------
        flow_input_dim: int
            The dimension of the encoded representation for the flow/marginal component.
        copula_input_dim: Optional[int]
            The dimension of the encoded representation for the copula component.
        dsf_marginal: Dict[str, Any]
            Configuration dictionary for the DSFMarginal component.
        attentional_copula: Optional[Dict[str, Any]], default to None
            Configuration dictionary for the AttentionalCopula component. Required if skip_copula=False.
        min_u: float, default to 0.0
        max_u: float, default to 1.0
            The values sampled from the copula will be scaled from [0, 1] to [min_u, max_u] before being sent to the marginal.
        skip_sampling_marginal: bool, default to False
            If set to True, then the output from the copula will not be transformed using the marginal during sampling.
        skip_copula: bool, default to True
            If True, only the marginal component is used (Stage 1). If False, both are used (Stage 2).
        """
        super().__init__()

        self.flow_input_dim = flow_input_dim
        self.copula_input_dim = copula_input_dim
        self.min_u = min_u
        self.max_u = max_u
        self.skip_sampling_marginal = skip_sampling_marginal
        self.attentional_copula_args = attentional_copula
        self.dsf_marginal_args = dsf_marginal
        self.skip_copula = skip_copula

        # --- Initialize Marginal ---
        # Ensure context_dim matches flow_input_dim
        if self.dsf_marginal_args["context_dim"] != self.flow_input_dim:
            logger.warning(f"DSF Marginal context_dim ({self.dsf_marginal_args['context_dim']}) != flow_input_dim ({self.flow_input_dim}). Using flow_input_dim.")
            self.dsf_marginal_args["context_dim"] = self.flow_input_dim
        self.marginal = DSFMarginal(**self.dsf_marginal_args)
        logger.debug(f"Initialized marginal (DSF) in CopulaDecoder with context_dim={self.flow_input_dim}")

        # --- Initialize Copula (if not skipping) ---
        self.copula = None
        if not self.skip_copula:
            self.create_attentional_copula() # Call helper to initialize

        self.copula_loss_val = None
        self.marginal_logdet_val = None

    def create_attentional_copula(self):
        """Initializes the AttentionalCopula component."""
        if self.attentional_copula_args is None:
            logger.error("Cannot create attentional copula: arguments not provided.")
            return

        if self.copula_input_dim is None:
             logger.error("Cannot create attentional copula: copula_input_dim not provided.")
             return

        # Ensure input_dim matches copula_input_dim
        if self.attentional_copula_args.get("input_dim") != self.copula_input_dim:
             logger.warning(f"Attentional Copula input_dim ({self.attentional_copula_args.get('input_dim')}) != copula_input_dim ({self.copula_input_dim}). Using copula_input_dim.")
             self.attentional_copula_args["input_dim"] = self.copula_input_dim

        self.copula = AttentionalCopula(**self.attentional_copula_args)
        self.skip_copula = False # Ensure flag is set correctly
        logger.debug(f"Initialized AttentionalCopula in CopulaDecoder with input_dim={self.copula_input_dim}")


    def loss(
        self,
        flow_encoded: torch.Tensor,
        copula_encoded: Optional[torch.Tensor],
        mask: torch.BoolTensor,
        true_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss components (marginal logdet and copula loss) of the decoder.

        Parameters:
        -----------
        flow_encoded: Tensor [batch, series, time steps, flow_dim]
            Encoded representation for the flow/marginal component.
        copula_encoded: Optional[Tensor] [batch, series, time steps, copula_dim]
            Encoded representation for the copula component (None if skip_copula=True).
        mask: BoolTensor [batch, series, time steps]
            Mask indicating observed (True) vs predicted (False).
        true_value: Tensor [batch, series, time steps]
            True values (normalized externally).

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]:
            - marginal_logdet: Tensor [batch], log determinant from marginals.
            - copula_loss: Tensor [batch], NLL from copula (0 if skipped).
        """
        B, S, T_total, _ = flow_encoded.shape
        N = S * T_total
        device = flow_encoded.device

        # Merge series and time dimensions for processing
        # Shapes become [B, N, D] or [B, N]
        flow_encoded_merged = flow_encoded.reshape(B, N, -1)
        copula_encoded_merged = copula_encoded.reshape(B, N, -1) if copula_encoded is not None else None
        mask_merged = mask.reshape(B, N)
        true_value_merged = true_value.reshape(B, N)

        # Assume mask is constant across batch for splitting hist/pred
        # Note: This differs slightly from original TACTiS which took mask[0,:],
        # but using the full mask_merged should be more robust if masks vary.
        # We need indices, not the mask itself, for splitting.
        # Let's calculate hist/pred lengths based on the mask of the first batch item.
        mask_first = mask[0, 0, :] # Shape [T_total]
        hist_indices = torch.where(mask_first)[0]
        pred_indices = torch.where(~mask_first)[0]
        num_hist_steps = len(hist_indices)
        num_pred_steps = len(pred_indices)
        # num_pred_variables = S * num_pred_steps # Used in original copula loss call

        # --- Marginal Calculation ---
        # DSFMarginal expects context [B, N, D] and x [B, N]
        u_vals_merged, marginal_logdet_batch = self.marginal.forward_logdet(
            flow_encoded_merged, true_value_merged
        )
        # marginal_logdet_batch shape: [B] (already summed over N dimension by DSF)
        self.marginal_logdet_val = marginal_logdet_batch # Store for potential logging

        # --- Copula Calculation (if not skipped) ---
        copula_loss_batch = torch.zeros(B, device=device) # Initialize loss to zero
        if not self.skip_copula and copula_encoded_merged is not None and self.copula is not None:
            # Prepare inputs for copula.loss
            # Need hist/pred splits of encoded representations and u_values

            # Reshape u_vals back to [B, S, T_total] to easily split
            u_vals_all = u_vals_merged.reshape(B, S, T_total)

            # Split based on mask indices (applied to merged N dimension)
            # We need to map T_total indices back to N indices
            # Example: N = S*T, index t in T maps to indices t, t+T, t+2T... in N
            hist_indices_n = torch.cat([hist_indices + s * T_total for s in range(S)]).sort().values
            pred_indices_n = torch.cat([pred_indices + s * T_total for s in range(S)]).sort().values

            hist_encoded_copula_flat = copula_encoded_merged[:, hist_indices_n, :] # [B, S*hist_len, D_copula]
            pred_encoded_copula_flat = copula_encoded_merged[:, pred_indices_n, :] # [B, S*pred_len, D_copula]
            hist_true_u_flat = u_vals_merged[:, hist_indices_n] # [B, S*hist_len]
            pred_true_u_flat = u_vals_merged[:, pred_indices_n] # [B, S*pred_len]

            # Call copula loss
            copula_loss_batch = self.copula.loss(
                hist_encoded=hist_encoded_copula_flat,
                hist_true_u=hist_true_u_flat,
                pred_encoded=pred_encoded_copula_flat,
                pred_true_u=pred_true_u_flat,
                num_series=S,
                num_timesteps=num_pred_steps, # Original uses pred_len here
            ) # Shape [B]
        else:
             # Ensure copula_loss_batch remains zero if copula is skipped
             copula_loss_batch = torch.zeros(B, device=device)


        self.copula_loss_val = copula_loss_batch # Store for potential logging

        # Return the two loss components per batch item
        return marginal_logdet_batch, copula_loss_batch


    def sample(
        self,
        num_samples: int,
        flow_encoded: torch.Tensor,
        copula_encoded: Optional[torch.Tensor],
        mask: torch.BoolTensor,
        true_value: torch.Tensor, # Contains observed history (normalized)
    ) -> torch.Tensor:
        """
        Generate samples using the decoder components.

        Parameters:
        -----------
        num_samples: int
            Number of samples to generate.
        flow_encoded: Tensor [batch, series, time steps, flow_dim]
            Encoded representation for the flow/marginal component.
        copula_encoded: Optional[Tensor] [batch, series, time steps, copula_dim]
            Encoded representation for the copula component (None if skip_copula=True).
        mask: BoolTensor [batch, series, time steps]
            Mask indicating observed (True) vs predicted (False).
        true_value: Tensor [batch, series, time steps]
            True values for observed steps (normalized externally).

        Returns:
        --------
        samples: torch.Tensor [num_samples, batch, series, time steps]
            Generated samples (normalized).
        """
        B, S, T_total, D_flow = flow_encoded.shape
        N = S * T_total
        device = flow_encoded.device

        # Merge dims for processing
        flow_encoded_merged = flow_encoded.reshape(B, N, D_flow)
        copula_encoded_merged = copula_encoded.reshape(B, N, -1) if copula_encoded is not None else None
        mask_merged = mask.reshape(B, N)
        true_value_merged = true_value.reshape(B, N)

        # Get indices for hist/pred based on mask
        mask_first = mask[0, 0, :] # Shape [T_total]
        hist_indices = torch.where(mask_first)[0]
        pred_indices = torch.where(~mask_first)[0]
        num_hist_steps = len(hist_indices)
        num_pred_steps = len(pred_indices)

        # Map T_total indices back to N indices
        hist_indices_n = torch.cat([hist_indices + s * T_total for s in range(S)]).sort().values
        pred_indices_n = torch.cat([pred_indices + s * T_total for s in range(S)]).sort().values

        # --- Prepare History for Copula Sampling ---
        hist_encoded_flow_flat = flow_encoded_merged[:, hist_indices_n, :] # [B, S*hist_len, D_flow]
        hist_true_x_flat = true_value_merged[:, hist_indices_n] # [B, S*hist_len]

        # Transform history to U(0,1) using marginals
        hist_true_u_flat = self.marginal.forward_no_logdet(hist_encoded_flow_flat, hist_true_x_flat) # [B, S*hist_len]

        # --- Sample U(0,1) values for Prediction Period ---
        pred_encoded_flow_flat = flow_encoded_merged[:, pred_indices_n, :] # [B, S*pred_len, D_flow]

        if not self.skip_copula and copula_encoded_merged is not None and self.copula is not None:
            # Use Copula to sample correlated U values
            hist_encoded_copula_flat = copula_encoded_merged[:, hist_indices_n, :] # [B, S*hist_len, D_copula]
            pred_encoded_copula_flat = copula_encoded_merged[:, pred_indices_n, :] # [B, S*pred_len, D_copula]

            # Copula sample method expects hist_u [B, S*hist_len] and pred_encoded [B, S*pred_len, D_copula]
            # Output shape: [B, S*pred_len, num_samples]
            pred_samples_u_flat = self.copula.sample(
                num_samples=num_samples,
                hist_encoded=hist_encoded_copula_flat,
                hist_true_u=hist_true_u_flat,
                pred_encoded=pred_encoded_copula_flat,
            )
        else:
            # Stage 1 or Copula skipped: Sample uniformly
            N_pred = S * num_pred_steps
            # Shape: [B, N_pred, num_samples]
            pred_samples_u_flat = torch.rand(B, N_pred, num_samples, device=device)

        # --- Transform Sampled U values using Marginal Inverse CDF ---
        if not self.skip_sampling_marginal:
            # Scale U samples if needed (original TACTiS logic)
            pred_samples_u_flat = self.min_u + (self.max_u - self.min_u) * pred_samples_u_flat

            # Marginal inverse expects context [B, N_pred, D_flow] and u [B, N_pred, num_samples]
            # Output shape: [B, N_pred, num_samples]
            pred_samples_x_flat = self.marginal.inverse(
                pred_encoded_flow_flat, # Context for prediction steps
                pred_samples_u_flat,    # U values for prediction steps
            )
        else:
            # If skipping marginal transform, output is just the U samples
            pred_samples_x_flat = pred_samples_u_flat

        # --- Combine History and Samples ---
        # Create output tensor matching input true_value shape but with sample dim
        # Target shape: [B, N, num_samples]
        samples_merged = torch.zeros(B, N, num_samples, device=device)

        # Fill in historical values (repeated across samples)
        samples_merged[:, hist_indices_n, :] = hist_true_x_flat.unsqueeze(-1).expand(-1, -1, num_samples)
        # Fill in predicted samples
        samples_merged[:, pred_indices_n, :] = pred_samples_x_flat

        # Reshape back to [num_samples, B, S, T_total]
        samples_final = samples_merged.reshape(B, S, T_total, num_samples).permute(3, 0, 1, 2)

        return samples_final.contiguous()