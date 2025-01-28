import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from copy import deepcopy

# from .encoder import Encoder, TemporalEncoder
# from .decoder import Decoder, CopulaDecoder
# from .positional_encoding import PositionalEncoding
# from .normalization import NormalizationLayer

from pytorch_transformer_ts.tactis_2.model.encoder import Encoder, TemporalEncoder
from pytorch_transformer_ts.tactis_2.model.decoder import Decoder, CopulaDecoder
from pytorch_transformer_ts.tactis_2.model.positional_encoding import PositionalEncoding
from pytorch_transformer_ts.tactis_2.model.normalization import NormalizationLayer

class NormalizationIdentity(nn.Module):
    """
    Identity mapping for normalization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, data: torch.Tensor, *args, **kwargs):
        return data, torch.zeros_like(data), torch.ones_like(data)

    def inverse(self, data: torch.Tensor, *args, **kwargs):
        return data

class NormalizationStandardization(nn.Module):
    """
    Standardization normalization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, data: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8, *args, **kwargs):
        # data: [batch, series, time steps]
        # mask: [batch, series, time steps]
        # Calculate the mean and standard deviation along the time dimension, only considering non-masked values
        mean = (data * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        std = torch.sqrt(
            ((data - mean) ** 2 * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        )
        # Normalize the data
        normalized_data = (data - mean) / (std + eps)
        return normalized_data, mean, std

    def inverse(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *args, **kwargs):
        # Denormalize the data
        denormalized_data = data * std + mean
        return denormalized_data

class TACTiS(nn.Module):
    """
    The top-level module for TACTiS.

    The role of this module is to handle everything outside of the encoder and decoder.
    This consists mainly the data manipulation ahead of the encoder and after the decoder.
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
        super().__init__()

        assert copula_decoder is not None, "Must select exactly one type of decoder"

        assert (not bagging_size) or bagging_size <= num_series, "Bagging size must not be above number of series"

        data_normalization = data_normalization.lower()
        assert data_normalization in {"", "none", "standardization"}
        loss_normalization = loss_normalization.lower()
        assert loss_normalization in {"", "none", "series", "timesteps", "both"}

        self.num_series = num_series
        self.flow_series_embedding_dim = flow_series_embedding_dim
        self.copula_series_embedding_dim = copula_series_embedding_dim
        self.flow_input_encoder_layers = flow_input_encoder_layers
        self.copula_input_encoder_layers = copula_input_encoder_layers
        self.bagging_size = bagging_size
        self.input_encoding_normalization = input_encoding_normalization
        self.loss_normalization = loss_normalization
        self.positional_encoding = positional_encoding
        self.flow_encoder_args = flow_encoder
        self.copula_encoder_args = copula_encoder
        self.flow_temporal_encoder_args = flow_temporal_encoder
        self.copula_temporal_encoder_args = copula_temporal_encoder
        self.copula_decoder_args = copula_decoder

        self.skip_copula = skip_copula

        assert experiment_mode in ["forecasting", "interpolation"]
        self.experiment_mode = experiment_mode

        # Data normalization
        self.data_normalization = {
            "": NormalizationIdentity,
            "none": NormalizationIdentity,
            "standardization": NormalizationStandardization,
        }[data_normalization]

        # Actual encoder
        # Split encoder
        if flow_encoder is not None and copula_encoder is not None:
            self.flow_encoder = Encoder(**flow_encoder)
            if not self.skip_copula:
                self.copula_encoder = Encoder(**copula_encoder)
        elif flow_temporal_encoder is not None and copula_temporal_encoder is not None:
            self.flow_encoder = TemporalEncoder(**flow_temporal_encoder)
            if not self.skip_copula:
                self.copula_encoder = TemporalEncoder(**copula_temporal_encoder)
        self.flow_encoder_embedding_dim = self.flow_encoder.embedding_dim
        if not self.skip_copula:
            self.copula_encoder_embedding_dim = self.copula_encoder.embedding_dim
            copula_decoder["attentional_copula"]["input_dim"] = self.copula_encoder_embedding_dim

        # Split input encoder
        # Series encoding, Positional encoding and input encoder (that transforms [x*m, c, m] into [z])
        self.flow_series_encoder = nn.Embedding(num_embeddings=num_series, embedding_dim=self.flow_series_embedding_dim)
        if not self.skip_copula:
            self.copula_series_encoder = nn.Embedding(
                num_embeddings=num_series,
                embedding_dim=self.copula_series_embedding_dim,
            )

        if positional_encoding is not None:
            self.flow_time_encoding = PositionalEncoding(self.flow_encoder_embedding_dim, **positional_encoding)
            if not self.skip_copula:
                self.copula_time_encoding = PositionalEncoding(self.copula_encoder_embedding_dim, **positional_encoding)
        else:
            self.flow_time_encoding = None
            if not self.skip_copula:
                self.copula_time_encoding = None

        flow_elayers = nn.ModuleList([])
        for i in range(self.flow_input_encoder_layers):
            if i == 0:
                flow_elayers.append(nn.Linear(self.flow_series_embedding_dim + 2, self.flow_encoder_embedding_dim))
            else:
                flow_elayers.append(nn.Linear(self.flow_encoder_embedding_dim, self.flow_encoder_embedding_dim))
            flow_elayers.append(nn.ReLU())
        self.flow_input_encoder = nn.Sequential(*flow_elayers)

        if not self.skip_copula:
            copula_elayers = nn.ModuleList([])
            for i in range(self.copula_input_encoder_layers):
                if i == 0:
                    copula_elayers.append(
                        nn.Linear(self.copula_series_embedding_dim + 2, self.copula_encoder_embedding_dim)
                    )  # +1 for the value, +1 for the mask, and the per series embedding
                else:
                    copula_elayers.append(
                        nn.Linear(self.copula_encoder_embedding_dim, self.copula_encoder_embedding_dim)
                    )
                copula_elayers.append(nn.ReLU())
            self.copula_input_encoder = deepcopy(nn.Sequential(*copula_elayers))

        if copula_decoder is not None:
            flow_input_dim = self.flow_encoder_embedding_dim
            copula_input_dim = (
                None if self.skip_copula else self.copula_encoder_embedding_dim
            )  # Since we do not have a copula encoder yet
            self.decoder = CopulaDecoder(
                flow_input_dim=flow_input_dim,
                copula_input_dim=copula_input_dim,
                skip_copula=self.skip_copula,
                **copula_decoder
            )

        self.stage = 1
        self.copula_loss = None
        self.marginal_logdet = None
        self.current_normalizer = None

    def set_stage(self, stage: int):
        self.stage = stage

    def initialize_stage2(self):
        self.set_stage(2)
        self.skip_copula = False
        if self.copula_encoder_args:
            self.copula_encoder = Encoder(**self.copula_encoder_args)
            self.decoder.attentional_copula_args["input_dim"] = self.copula_encoder.embedding_dim
        elif self.copula_temporal_encoder_args:
            self.copula_encoder = TemporalEncoder(**self.copula_temporal_encoder_args)
            self.decoder.attentional_copula_args["input_dim"] = self.copula_encoder.embedding_dim
        self.copula_encoder_embedding_dim = self.copula_encoder.embedding_dim
        copula_dim = self.copula_encoder_embedding_dim
        self.copula_series_encoder = nn.Embedding(
            num_embeddings=self.num_series,
            embedding_dim=self.copula_series_embedding_dim,
        )
        if self.positional_encoding:
            self.copula_time_encoding = PositionalEncoding(copula_dim, **self.positional_encoding)

        copula_elayers = nn.ModuleList([])
        for i in range(self.copula_input_encoder_layers):
            if i == 0:
                copula_elayers.append(
                    nn.Linear(self.copula_series_embedding_dim + 2, copula_dim)
                )  # +1 for the value, +1 for the mask, and the per series embedding
            else:
                copula_elayers.append(nn.Linear(copula_dim, copula_dim))
            copula_elayers.append(nn.ReLU())
        self.copula_input_encoder = deepcopy(nn.Sequential(*copula_elayers))

        if self.copula_decoder_args:
            self.decoder.create_attentional_copula()

    def set_experiment_mode(self, experiment_mode: str):
        assert experiment_mode in ["forecasting", "interpolation"]
        self.experiment_mode = experiment_mode

    def _apply_bagging(
        self,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: Optional[torch.Tensor] = None,
    ):
        # Select subset of series
        if self.bagging_size:
            num_batches, num_series = hist_value.shape[:2]
            # Sample without replacement
            bags = torch.stack(
                [torch.randperm(num_series, device=hist_value.device)[: self.bagging_size] for _ in range(num_batches)],
                dim=0,
            )
            # Select series
            series_to_keep = [bags[i, :] for i in range(num_batches)]
            series_to_keep = torch.stack(series_to_keep, dim=0)
            flow_series_emb = self.flow_series_encoder(series_to_keep)
            if not self.skip_copula:
                copula_series_emb = self.copula_series_encoder(series_to_keep)
            else:
                copula_series_emb = None
            hist_time, hist_value, pred_time, flow_series_emb, copula_series_emb = self._apply_subsetting(
                series_to_keep,
                hist_time,
                hist_value,
                pred_time,
                flow_series_emb=flow_series_emb,
                copula_series_emb=copula_series_emb,
            )
            if type(pred_value) != type(None):
                pred_value = self._apply_subsetting(
                    series_to_keep,
                    hist_time,
                    hist_value,
                    pred_time,
                    pred_value=pred_value,
                    flow_series_emb=flow_series_emb,
                    copula_series_emb=copula_series_emb,
                )[3]
        else:
            num_batches, num_series = hist_value.shape[:2]
            flow_series_emb = self.flow_series_encoder.weight.repeat(num_batches, 1, 1)
            if not self.skip_copula:
                copula_series_emb = self.copula_series_encoder.weight.repeat(num_batches, 1, 1)
            else:
                copula_series_emb = None

        return hist_time, hist_value, pred_time, pred_value, flow_series_emb, copula_series_emb

    @staticmethod
    def _apply_subsetting(
        series_to_keep,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value=None,
        permute_series=False,
        flow_series_emb: torch.Tensor = None,
        copula_series_emb: torch.Tensor = None,
    ):
        # Subsets series according to series_to_keep
        # If permute_series is True, the series are permuted in the output
        # hist_time: [batch, series, time steps]
        # hist_value: [batch, series, time steps]
        # pred_time: [batch, series, time steps]
        # pred_value: [batch, series, time steps]
        # flow_series_emb: [batch, series, embedding dim]
        # copula_series_emb: [batch, series, embedding dim]
        # series_to_keep: [batch, series]
        num_batches = hist_value.shape[0]
        if permute_series:
            # Permute series
            permutation = torch.randperm(hist_value.shape[1], device=hist_value.device)
            hist_time = torch.stack([hist_time[i, permutation, :] for i in range(num_batches)], dim=0)
            hist_value = torch.stack([hist_value[i, permutation, :] for i in range(num_batches)], dim=0)
            pred_time = torch.stack([pred_time[i, permutation, :] for i in range(num_batches)], dim=0)
            if type(pred_value) != type(None):
                pred_value = torch.stack([pred_value[i, permutation, :] for i in range(num_batches)], dim=0)

        # Select series
        hist_time = torch.stack([hist_time[i, series_to_keep[i, :], :] for i in range(num_batches)], dim=0)
        hist_value = torch.stack([hist_value[i, series_to_keep[i, :], :] for i in range(num_batches)], dim=0)
        pred_time = torch.stack([pred_time[i, series_to_keep[i, :], :] for i in range(num_batches)], dim=0)

        if type(flow_series_emb) != type(None):
            flow_series_emb = torch.stack([flow_series_emb[i, series_to_keep[i, :], :] for i in range(num_batches)], dim=0)
        if type(copula_series_emb) != type(None):
            copula_series_emb = torch.stack([copula_series_emb[i, series_to_keep[i, :], :] for i in range(num_batches)], dim=0)

        if type(pred_value) != type(None):
            pred_value = torch.stack([pred_value[i, series_to_keep[i, :], :] for i in range(num_batches)], dim=0)
            return (
                hist_time,
                hist_value,
                pred_time,
                pred_value,
                flow_series_emb,
                copula_series_emb,
            )
        else:
            return hist_time, hist_value, pred_time, flow_series_emb, copula_series_emb

    def forward(
        self,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hist_time: [batch, series, hist time steps]
        # hist_value: [batch, series, hist time steps]
        # pred_time: [batch, series, pred time steps]
        # pred_value: [batch, series, pred time steps]
        # output: [batch, series, pred time steps]

        # Bagging
        hist_time, hist_value, pred_time, pred_value, flow_series_emb, copula_series_emb = self._apply_bagging(
            hist_time, hist_value, pred_time, pred_value
        )

        # Normalize data
        normalizer = self.data_normalization()
        hist_value, hist_mean, hist_std = normalizer.forward(hist_value, mask=torch.ones_like(hist_value))
        if type(pred_value) != type(None):
            pred_value, _, _ = normalizer.forward(pred_value, mask=torch.ones_like(pred_value), mean=hist_mean, std=hist_std)
        self.current_normalizer = normalizer

        # Create mask
        # The mask is True if the value is not masked, and False if the value is masked.
        # For the flow, we only mask values in the prediction range.
        # For the copula, we mask values in the history range that are masked, and all values in the prediction range.
        if self.experiment_mode == "forecasting":
            flow_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
            copula_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
        elif self.experiment_mode == "interpolation":
            flow_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
            copula_mask = torch.ones_like(flow_mask)
        else:
            raise ValueError(f"Unknown experiment mode {self.experiment_mode}")

        # Concatenate hist and pred
        time = torch.cat([hist_time, pred_time], dim=-1)
        value = torch.cat([hist_value, pred_value], dim=-1)

        # Encode hist with the flow encoder
        # Input shape: [batch, series, time steps]
        # Output shape: [batch, series, time steps, embedding dim]
        flow_encoded = torch.cat(
            [
                flow_series_emb,
                value.unsqueeze(-1),
                flow_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        flow_encoded = self.flow_input_encoder(flow_encoded)
        if self.flow_time_encoding:
            flow_encoded = flow_encoded + self.flow_time_encoding(time)
        flow_encoded = self.flow_encoder.forward(
            flow_encoded
        )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]

        # Encode hist with the copula encoder
        # Input shape: [batch, series, time steps]
        # Output shape: [batch, series, time steps, embedding dim]
        if not self.skip_copula:
            copula_encoded = torch.cat(
                [
                    copula_series_emb,
                    value.unsqueeze(-1),
                    copula_mask.unsqueeze(-1),
                ],
                dim=-1,
            )
            copula_encoded = self.copula_input_encoder(copula_encoded)
            if self.copula_time_encoding:
                copula_encoded = copula_encoded + self.copula_time_encoding(time)
            copula_encoded = self.copula_encoder.forward(
                copula_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
        else:
            copula_encoded = None

        # Decode
        # Input shape: [batch, series, time steps, embedding dim]
        # Output shape: [batch, series, time steps]
        decoded = self.decoder.forward(
            flow_encoded=flow_encoded,
            copula_encoded=copula_encoded,
            mask=copula_mask,
            hist_time=hist_time,
            pred_time=pred_time,
        )

        # Denormalize data
        decoded = normalizer.inverse(decoded, mean=hist_mean, std=hist_std)

        return decoded

    def loss(
        self,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: torch.Tensor,
    ) -> torch.Tensor:
        # hist_time: [batch, series, hist time steps]
        # hist_value: [batch, series, hist time steps]
        # pred_time: [batch, series, pred time steps]
        # pred_value: [batch, series, pred time steps]
        # output: scalar

        # Bagging
        hist_time, hist_value, pred_time, pred_value, flow_series_emb, copula_series_emb = self._apply_bagging(
            hist_time, hist_value, pred_time, pred_value
        )

        # Normalize data
        normalizer = self.data_normalization()
        hist_value, hist_mean, hist_std = normalizer.forward(hist_value, mask=torch.ones_like(hist_value))
        pred_value, pred_mean, pred_std = normalizer.forward(
            pred_value, mask=torch.ones_like(pred_value), mean=hist_mean, std=hist_std
        )
        self.current_normalizer = normalizer

        # Create mask
        # The mask is True if the value is not masked, and False if the value is masked.
        # For the flow, we only mask values in the prediction range.
        # For the copula, we mask values in the history range that are masked, and all values in the prediction range.
        if self.experiment_mode == "forecasting":
            flow_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
            copula_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
        elif self.experiment_mode == "interpolation":
            flow_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
            copula_mask = torch.ones_like(flow_mask)
        else:
            raise ValueError(f"Unknown experiment mode {self.experiment_mode}")

        # Concatenate hist and pred
        time = torch.cat([hist_time, pred_time], dim=-1)
        value = torch.cat([hist_value, pred_value], dim=-1)

        # Encode hist with the flow encoder
        # Input shape: [batch, series, time steps]
        # Output shape: [batch, series, time steps, embedding dim]
        flow_encoded = torch.cat(
            [
                flow_series_emb,
                value.unsqueeze(-1),
                flow_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        flow_encoded = self.flow_input_encoder(flow_encoded)
        if self.flow_time_encoding:
            flow_encoded = flow_encoded + self.flow_time_encoding(time)

        # Encode hist with the copula encoder
        # Input shape: [batch, series, time steps]
        # Output shape: [batch, series, time steps, embedding dim]
        if not self.skip_copula:
            copula_encoded = torch.cat(
                [
                    copula_series_emb,
                    value.unsqueeze(-1),
                    copula_mask.unsqueeze(-1),
                ],
                dim=-1,
            )
            copula_encoded = self.copula_input_encoder(copula_encoded)
            if self.copula_time_encoding:
                copula_encoded = copula_encoded + self.copula_time_encoding(time)

        num_batches, num_series, num_timesteps = value.shape
        num_hist_timesteps = hist_value.shape[-1]
        num_pred_timesteps = pred_value.shape[-1]

        # Compute loss
        # Input shape: [batch, series, time steps, embedding dim]
        # Output shape: scalar
        if self.stage == 1:
            flow_encoded = self.flow_encoder.forward(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_encoder.forward(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]

            if self.skip_copula:
                copula_encoded = None

            _ = self.decoder.loss(
                flow_encoded=flow_encoded,
                copula_encoded=copula_encoded,
                mask=mask,
                true_value=true_value,
            )  # previously returned loss here when loss coefficients weren't used

            self.copula_loss = self.decoder.copula_loss
            self.marginal_logdet = self.decoder.marginal_logdet
            self.unnormalized_copula_loss = torch.clone(self.decoder.copula_loss)
            self.unnormalized_marginal_logdet = torch.clone(self.decoder.marginal_logdet)

            if self.loss_normalization in {"series", "both"}:
                self.copula_loss = self.copula_loss / num_series
                self.marginal_logdet = self.marginal_logdet / num_series
            if self.loss_normalization in {"timesteps", "both"}:
                self.copula_loss = self.copula_loss / num_pred_timesteps
                self.marginal_logdet = self.marginal_logdet / num_pred_timesteps

            return self.marginal_logdet, self.copula_loss

        elif self.stage == 2:
            flow_encoded = self.flow_encoder.forward(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            copula_encoded = self.copula_encoder.forward(
                copula_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]

            loss = self.decoder.loss(
                flow_encoded=flow_encoded,
                copula_encoded=copula_encoded,
                mask=mask,
                true_value=true_value,
            )

            self.copula_loss = self.decoder.copula_loss
            self.marginal_logdet = self.decoder.marginal_logdet
            self.unnormalized_copula_loss = torch.clone(self.decoder.copula_loss)
            self.unnormalized_marginal_logdet = torch.clone(self.decoder.marginal_logdet)

            if self.loss_normalization in {"series", "both"}:
                self.copula_loss = self.copula_loss / num_series
                self.marginal_logdet = self.marginal_logdet / num_series
            if self.loss_normalization in {"timesteps", "both"}:
                self.copula_loss = self.copula_loss / num_pred_timesteps
                self.marginal_logdet = self.marginal_logdet / num_pred_timesteps

            return loss

    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hist_time: [batch, series, hist time steps]
        # hist_value: [batch, series, hist time steps]
        # pred_time: [batch, series, pred time steps]
        # pred_value: [batch, series, pred time steps]
        # output: [batch, series, pred time steps]

        # Bagging
        hist_time, hist_value, pred_time, pred_value, flow_series_emb, copula_series_emb = self._apply_bagging(
            hist_time, hist_value, pred_time, pred_value
        )

        # Normalize data
        normalizer = self.data_normalization()
        hist_value, hist_mean, hist_std = normalizer.forward(hist_value, mask=torch.ones_like(hist_value))
        if type(pred_value) != type(None):
            pred_value, _, _ = normalizer.forward(pred_value, mask=torch.ones_like(pred_value), mean=hist_mean, std=hist_std)
        self.current_normalizer = normalizer

        # Create mask
        # The mask is True if the value is not masked, and False if the value is masked.
        # For the flow, we only mask values in the prediction range.
        # For the copula, we mask values in the history range that are masked, and all values in the prediction range.
        if self.experiment_mode == "forecasting":
            flow_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
            copula_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
        elif self.experiment_mode == "interpolation":
            flow_mask = torch.cat(
                [
                    torch.ones_like(hist_value),
                    torch.zeros_like(pred_value),
                ],
                dim=-1,
            )
            copula_mask = torch.ones_like(flow_mask)
        else:
            raise ValueError(f"Unknown experiment mode {self.experiment_mode}")

        # Concatenate hist and pred
        time = torch.cat([hist_time, pred_time], dim=-1)
        value = torch.cat([hist_value, pred_value], dim=-1)

        # Encode hist with the flow encoder
        # Input shape: [batch, series, time steps]
        # Output shape: [batch, series, time steps, embedding dim]
        flow_encoded = torch.cat(
            [
                flow_series_emb,
                value.unsqueeze(-1),
                flow_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        flow_encoded = self.flow_input_encoder(flow_encoded)
        if self.flow_time_encoding:
            flow_encoded = flow_encoded + self.flow_time_encoding(time)
        flow_encoded = self.flow_encoder.forward(
            flow_encoded
        )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]

        # Encode hist with the copula encoder
        # Input shape: [batch, series, time steps]
        # Output shape: [batch, series, time steps, embedding dim]
        if not self.skip_copula:
            copula_encoded = torch.cat(
                [
                    copula_series_emb,
                    value.unsqueeze(-1),
                    copula_mask.unsqueeze(-1),
                ],
                dim=-1,
            )
            copula_encoded = self.copula_input_encoder(copula_encoded)
            if self.copula_time_encoding:
                copula_encoded = copula_encoded + self.copula_time_encoding(time)
            copula_encoded = self.copula_encoder.forward(
                copula_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
        else:
            copula_encoded = None

        # Sample
        # Input shape: [batch, series, time steps, embedding dim]
        # Output shape: [num samples, batch, series, time steps]
        samples = self.decoder.sample(
            num_samples=num_samples,
            flow_encoded=flow_encoded,
            copula_encoded=copula_encoded,
            mask=copula_mask,
            hist_time=hist_time,
            pred_time=pred_time,
        )

        # Denormalize data
        samples = normalizer.inverse(samples, mean=hist_mean, std=hist_std)

        return samples