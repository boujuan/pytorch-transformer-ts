from math import sqrt, log
from typing import List, Optional, Tuple
from functools import partial
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.geometric import Geometric
from torch.distributions.binomial import Binomial
from einops import rearrange, repeat

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler

def FoldForPred(inp: torch.Tensor, length:int, d_params: torch.Tensor) -> torch.Tensor:
    # out = rearrange(inp, "batch (dy len) dim -> dim batch len dy", dy=dy)
    d_params_str = " ".join(f"dy{i}" for i in range(len(d_params)))
    out = rearrange(inp, f"batch (d_in length) {d_params_str} -> batch length d_in {d_params_str}", length=length, **{f"dy{i}": d for i, d in enumerate(d_params)})
        
    return out

class DecoderLayer(nn.Module):
    def __init__(
        self,
        global_self_attention,
        local_self_attention,
        global_cross_attention,
        local_cross_attention,
        d_model,
        d_y,
        time_windows=1,
        time_window_offset=0,
        dim_feedforward=None,
        dropout_ff=0.1,
        dropout_attn_out=0.0,
        activation="relu",
        norm="layer",
    ):
        super(DecoderLayer, self).__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.local_self_attention = local_self_attention
        self.global_self_attention = global_self_attention
        self.global_cross_attention = global_cross_attention

        self.local_cross_attention = local_cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = Normalization(method=norm, d_model=d_model)
        self.norm5 = Normalization(method=norm, d_model=d_model)

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.time_windows = time_windows
        self.time_window_offset = time_window_offset
        self.d_y = d_y

    def forward(
        self, x, cross, self_mask_seq=None, cross_mask_seq=None, output_cross_attn=False
    ):
        # pre-norm Transformer architecture
        attn = None
        if self.local_self_attention:
            # self attention on each variable in target sequence ind.
            assert self_mask_seq is None
            x1 = self.norm1(x)
            x1 = Localize(x1, self.d_y)
            x1, _ = self.local_self_attention(x1, x1, x1, attn_mask=self_mask_seq)
            x1 = ReverseLocalize(x1, self.d_y)
            x = x + self.dropout_attn_out(x1)

        if self.global_self_attention:
            x1 = self.norm2(x)
            x1 = WindowTime(
                x1,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = WindowTime(
                self_mask_seq,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, _ = self.global_self_attention(
                x1,
                x1,
                x1,
                attn_mask=MakeSelfMaskFromSeq(self_mask_seq),
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = ReverseWindowTime(
                self_mask_seq,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        if self.local_cross_attention:
            # cross attention between target/context on each variable ind.
            assert cross_mask_seq is None
            x1 = self.norm3(x)
            bs, *_ = x1.shape
            x1 = Localize(x1, self.d_y)
            # see above warnings and explanations about a potential
            # silent bug here.
            cross_local = Localize(cross, self.d_y)[: self.d_y * bs]
            x1, _ = self.local_cross_attention(
                x1,
                cross_local,
                cross_local,
                attn_mask=cross_mask_seq,
            )
            x1 = ReverseLocalize(x1, self.d_y)
            x = x + self.dropout_attn_out(x1)

        if self.global_cross_attention:
            x1 = self.norm4(x)
            x1 = WindowTime(
                x1,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross = WindowTime(
                cross,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross_mask_seq = WindowTime(
                cross_mask_seq,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, attn = self.global_cross_attention(
                x1,
                cross,
                cross,
                attn_mask=MakeCrossMaskFromSeq(self_mask_seq, cross_mask_seq),
                output_attn=output_cross_attn,
            )
            cross = ReverseWindowTime(
                cross,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross_mask_seq = ReverseWindowTime(
                cross_mask_seq,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        x1 = self.norm5(x)
        # feedforward layers as 1x1 convs
        x1 = self.dropout_ff(self.activation(self.conv1(x1.transpose(-1, 1))))
        x1 = self.dropout_ff(self.conv2(x1).transpose(-1, 1))
        output = x + x1

        return output, attn


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, emb_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self,
        val_time_emb,
        space_emb,
        cross,
        self_mask_seq=None,
        cross_mask_seq=None,
        output_cross_attn=False,
    ):
        x = self.emb_dropout(val_time_emb) + self.emb_dropout(space_emb)

        attns = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                cross,
                self_mask_seq=self_mask_seq,
                cross_mask_seq=cross_mask_seq,
                output_cross_attn=output_cross_attn,
            )
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns

class EncoderLayer(nn.Module):
    def __init__(
        self,
        global_attention,
        local_attention,
        d_model,
        d_y,
        time_windows=1,
        time_window_offset=0,
        dim_feedforward=None,
        dropout_ff=0.1,
        dropout_attn_out=0.0,
        activation="relu",
        norm="layer",
    ):
        super(EncoderLayer, self).__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.local_attention = local_attention
        self.global_attention = global_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.time_windows = time_windows
        self.time_window_offset = time_window_offset
        self.d_y = d_y

    def forward(self, x, self_mask_seq=None, output_attn=False):
        # uses pre-norm Transformer architecture
        attn = None
        if self.local_attention:
            # attention on tokens of each variable ind. TODO change this to attend to all of turbine's variables eg nd and ws?
            x1 = self.norm1(x)
            x1 = Localize(x1, self.d_y)
             
            x1, _ = self.local_attention(
                x1, x1, x1, attn_mask=self_mask_seq, output_attn=False
            )
            x1 = ReverseLocalize(x1, self.d_y)
            x = x + self.dropout_attn_out(x1)

        if self.global_attention:
            # attention on tokens of every variable together
            x1 = self.norm2(x)

            x1 = WindowTime(
                x1,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )

            self_mask_seq = WindowTime(
                self_mask_seq,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, attn = self.global_attention(
                x1,
                x1,
                x1,
                attn_mask=MakeSelfMaskFromSeq(self_mask_seq),
                output_attn=output_attn,
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = ReverseWindowTime(
                self_mask_seq,
                dy=self.d_y,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        x1 = self.norm3(x)
        # feedforward layers (done here as 1x1 convs)
        x1 = self.dropout_ff(self.activation(self.conv1(x1.transpose(-1, 1))))
        x1 = self.dropout_ff(self.conv2(x1).transpose(-1, 1))
        output = x + x1
        return output, attn


class Encoder(nn.Module):
    def __init__(
        self,
        attn_layers,
        conv_layers,
        norm_layer,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, val_time_emb, space_emb, self_mask_seq=None, output_attn=False):
        x = self.emb_dropout(val_time_emb) + self.emb_dropout(space_emb)

        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(
                x, self_mask_seq=self_mask_seq, output_attn=output_attn
            )
            if len(self.conv_layers) > i:
                if self.conv_layers[i] is not None:
                    x = self.conv_layers[i](x)
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns

def Flatten(inp: torch.Tensor) -> torch.Tensor:
    # spatiotemporal flattening of (batch, length, dim) into (batch, length x dim)
    out = rearrange(inp, "batch len dy -> batch (dy len) 1")
    return out


def Localize(inp: torch.Tensor, variables: int) -> torch.Tensor:
    # split spatiotemporal into individual vars and fold into batch dim
    return rearrange(
        inp,
        "batch (variables len) dim -> (variables batch) len dim",
        variables=variables,
    )


def MakeSelfMaskFromSeq(seq_mask: torch.Tensor):
    if seq_mask is None:
        return None
    batch, length, dim = seq_mask.shape
    assert dim == 1
    mask_rows = repeat(seq_mask, f"batch len 1 -> batch {length} len")
    mask_cols = repeat(seq_mask, f"batch len 1 -> batch len {length}")
    mask = torch.max(mask_rows, mask_cols).bool()
    return mask


def MakeCrossMaskFromSeq(self_seq_mask: torch.Tensor, cross_seq_mask: torch.Tensor):
    if self_seq_mask is None:
        return None

    batch_, cross_len, dim = cross_seq_mask.shape
    assert dim == 1
    batch, self_len, dim = self_seq_mask.shape
    assert batch_ == batch
    assert dim == 1

    mask_cols = repeat(self_seq_mask, f"batch len 1 -> batch len {cross_len}")
    mask_rows = repeat(cross_seq_mask, f"batch len 1 -> batch {self_len} len")
    mask = torch.max(mask_rows, mask_cols).bool()
    return mask


def WindowTime(
    inp: torch.Tensor, dy: int, windows: int, window_offset: int
) -> torch.Tensor:
    # stack
    if windows == 1 or inp is None:
        return inp
    x = rearrange(inp, "batch (dy len) dim -> batch len dy dim", dy=dy)

    if window_offset:
        # shift
        b, l, _, dim = x.shape
        window_len = l // 2
        shift_by = window_len // window_offset
        x = torch.roll(x, -shift_by, dims=1)

    # window and flatten
    x = rearrange(
        x, "batch (windows len) dy dim -> (batch windows) (dy len) dim", windows=windows
    )
    return x


def ReverseWindowTime(
    inp: torch.Tensor, dy: int, windows: int, window_offset: int
) -> torch.Tensor:
    if windows == 1 or inp is None:
        return inp
    # reverse window and stack
    x = rearrange(
        inp,
        "(batch windows) (dy len) dim -> batch (windows len) dy dim",
        dy=dy,
        windows=windows,
    )

    if window_offset:
        # shift
        b, l, _, dim = x.shape
        window_len = l // 2
        shift_by = window_len // window_offset
        x = torch.roll(x, shift_by, dims=1)

    # flatten
    x = rearrange(x, "batch len dy dim -> batch (dy len) dim", dy=dy)
    return x


def ReverseLocalize(inp: torch.Tensor, variables: int) -> torch.Tensor:
    return rearrange(
        inp,
        "(variables batch) len dim -> batch (variables len) dim",
        variables=variables,
    )


def ShiftBeforeWindow(inp: torch.Tensor, windows: int, offset: int = 2):
    # SWIN Transformer style window offsets
    b, l, v, d = inp.shape
    window_len = l // windows
    shift_by = window_len // offset
    return torch.roll(inp, -shift_by, dims=1)


def ReverseShiftBeforeWindow(inp: torch.Tensor, windows: int, offset: int = 2):
    b, l, v, d = inp.shape
    window_len = l // windows
    shift_by = window_len // offset
    return torch.roll(inp, shift_by, dims=1)


def Stack(inp: torch.Tensor, dy: int):
    return rearrange(inp, "batch (dy len) dim -> batch len dy dim", dy=dy)

class ConvBlock(nn.Module):
    def __init__(
        self,
        split_length_into,
        d_model,
        conv_kernel_size=3,
        conv_stride=1,
        pool=True,
        pool_kernel_size=3,
        pool_stride=2,
        activation="gelu",
    ):
        super().__init__()
        self.split_length = split_length_into
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(d_model)

        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unrecognized ConvBlock activation: `{activation}`")

        self.pool = (
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=1)
            if pool
            else return_unchanged
        )

    def conv_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        x = rearrange(
            x, f"batch (sl len) d_model -> (batch sl) d_model len", sl=self.split_length
        )
        x = self.conv_forward(x)
        x = rearrange(
            x, f"(batch sl) d_model len -> batch (sl len) d_model", sl=self.split_length
        )
        return x


class MaskPowerNorm(nn.Module):
    """
    An implementation of masked batch normalization, used for testing the numerical
    stability.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        alpha_fwd=0.9,
        alpha_bkw=0.9,
        affine=True,
        warmup_iters=10000,
        group_num=1,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        self.register_parameter("weight", nn.Parameter(torch.ones(num_features)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_features)))
        self.register_buffer("running_phi", torch.ones(1, num_features, 1, 1))
        self.register_buffer("ema_gz", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("iters", torch.zeros(1).type(torch.LongTensor))

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        self.eps = eps
        self.debug = False
        self.warmup_iters = warmup_iters
        self.gp = GroupScaling1D(group_num=group_num)
        self.group_num = group_num

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, alpha_fwd={afwd}, alpha_bkw={abkw}, "
            "affine={affine}, warmup={warmup_iters}, group_num={group_num}".format(
                **self.__dict__
            )
        )

    def forward(self, inp, pad_mask=None, is_encoder=False):
        """
        inp:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        shaped_input = len(inp.shape) == 2
        if shaped_input:
            inp = inp.unsqueeze(0)
        T, B, C = inp.shape
        inp = self.gp(inp)

        # construct the mask_input, size to be (BxL) x C: L is the real length here
        if pad_mask is None:
            mask_input = inp.clone()
        else:
            # Transpose the bn_mask (B x T -> T x B)
            bn_mask = ~pad_mask
            bn_mask = bn_mask.transpose(0, 1)

        if pad_mask is not None:
            pad_size = (~bn_mask).sum()
            mask_input = inp[bn_mask, :]
        else:
            mask_input = inp.clone()

        mask_input = mask_input.reshape(-1, self.num_features)

        inp = inp.permute(1, 2, 0).contiguous()
        input_shape = inp.size()
        inp = inp.reshape(inp.size(0), self.num_features, -1)
        inp = inp.unsqueeze(-1)

        if self.training:
            self.iters.copy_(self.iters + 1)
            output = PowerFunction.apply(
                inp,
                self.weight,
                self.bias,
                self.running_phi,
                self.eps,
                self.afwd,
                self.abkw,
                self.ema_gz,
                self.debug,
                self.warmup_iters,
                self.iters,
                mask_input,
            )

        else:
            N, C, H, W = inp.size()
            var = self.running_phi
            output = inp / (var + self.eps).sqrt()
            output = self.weight.reshape(1, C, 1, 1) * output + self.bias.reshape(
                1, C, 1, 1
            )

        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        # Reshape it.
        if shaped_input:
            output = output.squeeze(0)

        return output

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps) * self.scale
        x = x / n * self.g
        return x

class Normalization(nn.Module):
    def __init__(self, method, d_model=None):
        super().__init__()
        assert method in ["layer", "scale", "batch", "power", "none"]
        if method == "layer":
            assert d_model
            self.norm = nn.LayerNorm(d_model)
        elif method == "scale":
            self.norm = ScaleNorm(d_model)
        elif method == "power":
            self.norm = MaskPowerNorm(d_model, warmup_iters=1000)
        elif method == "none":
            self.norm = return_unchanged
        else:
            assert d_model
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)

class Time2Vec(nn.Module):
    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            x = torch.diag_embed(x)
            # x.shape = (bs, sequence_length, input_dim, input_dim)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # x_affine.shape = (bs, sequence_length, input_dim, time_embed_dim)
            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1
            )
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
            # x_output.shape = (bs, sequence_length, input_dim * time_embed_dim)
        else:
            x_output = x
        return x_output

def return_unchanged(y):
    return y

class Embedding(nn.Module):
    def __init__(
        self,
        d_y,
        d_x_time,
        d_x_feat,
        d_x_static,
        d_y_lagged,
        d_model,
        time_emb_dim=6,
        method="spatio-temporal",
        downsample_convs=0,
        start_token_len=0,
        null_value=None,
        pad_value=None,
        is_encoder: bool = True,
        position_emb="abs",
        data_dropout=None,
        max_seq_len=None,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
    ):
        super().__init__()
        
        self.is_encoder = is_encoder
        self.embed_type = "augment_x" # or "augment_y" or "augment_x" if true, add y_lagged and x_feat to y, otherwise add to x
        assert self.embed_type in ["original", "augment_y", "augment_x"]
        # if self.embed_type == "augment_y":
        #     self.embed_type = "augment_y" if is_encoder else "augment_x"
         
        assert method in ["spatio-temporal", "temporal"]
        if data_dropout is None:
            self.data_drop = return_unchanged
        else:
            self.data_drop = data_dropout

        self.method = method

        # d_x just the number of time variables eg. month, day, hour
        time_dim = time_emb_dim * d_x_time
        self.time_emb = Time2Vec(d_x_time, embed_dim=time_dim)
        # self.age_emb = nn.Linear(1, time_dim)

        assert position_emb in ["t2v", "abs"]
        self.max_seq_len = max_seq_len
        self.position_emb = position_emb
        if self.position_emb == "t2v":
            # standard periodic pos emb but w/ learnable coeffs
            self.local_emb = Time2Vec(1, embed_dim=d_model + 1)
        elif self.position_emb == "abs":
            # lookup-based learnable pos emb
            assert max_seq_len is not None
            self.local_emb = nn.Embedding(
                num_embeddings=max_seq_len, embedding_dim=d_model
            )
        
        # input dim = input_size + time_Feature + age feature + dynamic real features + static_Features
        # NOTE non augmented version differs from original implementation bc:
        #       1) we add 1 for age to the time dimension,
        #       2) we add the lagged values 
        #       3) we add the static values (log(scale), log(abs(loc)) features, static features, dynamic real features that also influence output such as nacelle direction)
        if self.embed_type == "augment_y":
            y_emb_inp_dim = d_y + d_x_feat + d_y_lagged if self.method == "temporal" else 1 
            self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim + 1 + d_x_static, d_model)
        elif self.embed_type == "augment_x":
            y_emb_inp_dim = d_y if self.method == "temporal" else 1 # spatio temporal embedding will capture 1 value for each point in time and space
            # self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim + 1 + d_x_feat + d_x_static + d_y_lagged, d_model)
            self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim + 1 + d_x_feat + d_y_lagged, d_model)
        elif self.embed_type == "original":
            y_emb_inp_dim = d_y if self.method == "temporal" else 1
            self.val_time_emb = nn.Linear(y_emb_inp_dim, d_model)
            
        # self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim + 1 + d_x_static + d_y_lagged, d_model)
        
        if self.method == "spatio-temporal":
            # self.space_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)
            if self.embed_type == "augment_y":
                split_length_into = d_y + d_x_feat + d_y_lagged
            elif self.embed_type == "augment_x" or self.embed_type == "original":
                split_length_into = d_y
                
            self.space_emb = nn.Embedding(num_embeddings=split_length_into, embedding_dim=d_model)
        else:
            split_length_into = 1
            
        self.static_embed = nn.Linear(
            d_x_static, d_model
        )

        self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [ConvBlock(split_length_into, d_model) for _ in range(downsample_convs)]
        )

        self.d_model = d_model
        self.null_value = null_value
        self.pad_value = pad_value
        
        # turning off parts of the embedding is only really here for ablation studies
        self.use_val = use_val
        self.use_time = use_time
        self.use_given = use_given
        self.use_space = use_space

    def __call__(self,  y: torch.Tensor, y_lagged: torch.Tensor, x_time: torch.Tensor, x_age: torch.Tensor, x_feat: torch.Tensor, x_static: torch.Tensor):
        if self.method == "spatio-temporal":
            emb = self.spatio_temporal_embed
        else:
            emb = self.temporal_embed
        
        return emb(y=y, y_lagged=y_lagged, x_time=x_time, x_age=x_age, x_feat=x_feat, x_static=x_static)

    def make_mask(self, y):
        # we make padding-based masks here due to outdated
        # feature where the embedding randomly drops tokens by setting
        # them to the pad value as a form of regularization
        if self.pad_value is None:
            return None
        return (y == self.pad_value).any(-1, keepdim=True)

    def temporal_embed(self, y: torch.Tensor, y_lagged: torch.Tensor, x_time: torch.Tensor, x_age: torch.Tensor, x_feat: torch.Tensor, x_static: torch.Tensor):
        bs, length, dy = y.shape

        # protect against true NaNs. without
        # `spatio_temporal_embed`'s multivariate "Given"
        # concept there isn't much else we can do here.
        # NaNs should probably be set to a magic number value
        # in the dataset and passed to the null_value arg.
        # TODO this should match embedding below 
        y = torch.nan_to_num(y)
        x_time = torch.nan_to_num(x_time)
        x_age = torch.nan_to_num(x_age)
        x_feat = torch.nan_to_num(x_feat)
        x_static = torch.nan_to_num(x_static)
        y_lagged = torch.nan_to_num(y_lagged)

        if self.is_encoder:
            # optionally mask the context sequence for reconstruction
            y = self.data_drop(y)
        mask = self.make_mask(y)

        # position embedding ("local_emb")
        local_pos = torch.arange(length).type_as(x_time.type(torch.int64)) # CHANGE to(x_time.device)
        if self.position_emb == "t2v":
            # first idx of Time2Vec output is unbounded so we drop it to
            # reuse code as a learnable pos embb
            local_emb = self.local_emb(
                local_pos.view(1, -1, 1).repeat(bs, 1, 1).float()
            )[:, :, 1:]
        elif self.position_emb == "abs":
            assert length <= self.max_seq_len
            local_emb = self.local_emb(local_pos.long().view(1, -1).repeat(bs, 1))

        # time embedding (Time2Vec)
        if not self.use_time:
            x_time = torch.zeros_like(x_time)
            x_age = torch.zeros_like(x_age)
        
        x_emb = torch.cat((self.time_emb(x_time), x_age, x_feat, x_static, y_lagged), dim=-1)

        if not self.use_val:
            y = torch.zeros_like(y)

        val_time_inp = torch.cat((y, x_emb), dim=-1)
        val_time_emb = self.val_time_emb(val_time_inp)

        # "given" embedding. not important for temporal emb
        # when not using a start token
        given = torch.ones((bs, length)).type_as(x_time).long() # to(x_time.device)
        if not self.is_encoder and self.use_given:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given)

        emb = local_emb + val_time_emb + given_emb

        if self.is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        # space emb not used for temporal method
        space_emb = torch.zeros_like(emb)
        var_idxs = None
        return emb, space_emb, var_idxs, mask

    def spatio_temporal_embed(self, y: torch.Tensor, y_lagged: torch.Tensor, x_time: torch.Tensor, x_age: torch.Tensor, x_feat: torch.Tensor, x_static: torch.Tensor):
        # full spatiotemopral emb method. lots of shape rearrange code
        # here to create artifically long (length x dim) spatiotemporal sequence
        batch, length, dy = y.shape
        _, _, dx_feat = x_feat.shape
        _, _, dx_static = x_static.shape
        _, _, dy_lagged = y_lagged.shape 

        # position emb ("local_emb")
        # TODO NOTE: non augmented version differs from original implementation here bc dy is only the outputs, not also dy_c, should I add dx_feat here?
        if self.embed_type == "augment_y":
            y_dim = dy + dx_feat + dy_lagged
        elif self.embed_type == "augment_x" or self.embed_type == "original":
            y_dim = dy
            
        # local_pos = repeat(
        #         torch.arange(length).to(x_time.device), f"length -> {batch} ({y_dim} length)"
        # ) 
        local_pos = repeat(
                torch.arange(length).type_as(x_time.type(torch.int64)), f"length -> {batch} ({y_dim} length)" # CHANGE
        ) 
        
        if self.position_emb == "t2v":
            # periodic pos emb
            local_emb = self.local_emb(local_pos.float().unsqueeze(-1).float())[
                :, :, 1:
            ]
        elif self.position_emb == "abs":
            # lookup pos emb
            local_emb = self.local_emb(local_pos.long())

        # time emb
        if not self.use_time:
            x_time = torch.zeros_like(x_time)
            x_age = torch.zeros_like(x_age)
        x_time = torch.nan_to_num(x_time)
        x_age = torch.nan_to_num(x_age)
        x_static = torch.nan_to_num(x_static)
        
        x_time = repeat(x_time, f"batch len x_dim -> batch ({y_dim} len) x_dim")
        x_age = repeat(x_age, f"batch len x_dim -> batch ({y_dim} len) x_dim")
        # x_static = repeat(x_static, f"batch len x_dim -> batch ({y_dim} len) x_dim")
        
        if self.embed_type == "augment_y":
            
            x_emb = torch.cat((self.time_emb(x_time), x_age), dim=-1)
            
            # protect against NaNs in y, but keep track for Given emb
            y = torch.cat([y, x_feat, y_lagged], dim=-1)
            true_null = torch.isnan(y)
            y = torch.nan_to_num(y)
            if not self.use_val:
                y = torch.zeros_like(y)
            
            y_original = y.clone()
            y_original = Flatten(y_original)
            y = self.data_drop(y)
            y = Flatten(y)
            mask = self.make_mask(y)
            
        elif self.embed_type == "augment_x":
            y_lagged = torch.nan_to_num(y_lagged)
            x_feat = torch.nan_to_num(x_feat)
            
            x_feat = repeat(x_feat, f"batch len x_dim -> batch ({dy} len) x_dim")
            y_lagged = repeat(y_lagged, f"batch len x_dim -> batch ({dy} len) x_dim")
            # x_emb = torch.cat((self.time_emb(x_time), x_age, x_feat, x_static, y_lagged), dim=-1)
            x_emb = torch.cat((self.time_emb(x_time), x_age, x_feat, y_lagged), dim=-1)

            # protect against NaNs in y, but keep track for Given emb
            true_null = torch.isnan(y)
            y = torch.nan_to_num(y)
            if not self.use_val:
                y = torch.zeros_like(y)
            
            y_original = y.clone()
            y_original = Flatten(y_original)
            y = self.data_drop(y)
            y = Flatten(y)
            mask = self.make_mask(y)
        
        elif self.embed_type == "original":
            pass

        # concat time_emb, y --> FF --> val_time_emi
        # NOTE: this differs from the original spacetimeformer implementation bc we concatenate 
        # the outputs y to the time embedding, in addition to the static vars, dynamic real features, and y_lagged
         
        val_time_inp = torch.cat((y, x_emb), dim=-1)
        val_time_emb = self.val_time_emb(val_time_inp) # y_emb_inp_dim (1) + time_dim (24) + 1 + d_x_feat (6) + d_x_static (14) + d_y_lagged (138)
        static_emb = repeat(self.static_embed(x_static), f"batch len x_static_dim -> batch ({dy} len) x_static_dim")
        # "given" embedding
        if self.use_given:
            
            given = torch.ones((batch, length, y_dim)).type_as(x_time).long() # CHANGE .to(x_time.device)  # start as True
            
            if not self.is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0  # (False)

            # if y was NaN, set Given = False
            given *= ~true_null

            # flatten now to make the rest easier to figure out
            given = rearrange(given, "batch len dy -> batch (dy len)")

            # use given embeddings to identify data that was dropped out
            given *= (y == y_original).squeeze(-1)

            if self.null_value is not None:
                # mask null values that were set to a magic number in the dataset itself
                null_mask = (y != self.null_value).squeeze(-1)
                given *= null_mask

            given_emb = self.given_emb(given)
        else:
            given_emb = 0.0

        val_time_emb = local_emb + val_time_emb + given_emb + static_emb

        if self.is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # space embedding
            
        # var_idx = repeat(
        #     torch.arange(y_dim).long().to(x_time.device), f"dy -> {batch} (dy {length})"
        # )
        var_idx = repeat(
            torch.arange(y_dim).type_as(x_time).long(), f"dy -> {batch} (dy {length})" # CHANGE
        )

        var_idx_true = var_idx.clone()
        if not self.use_space:
            var_idx = torch.zeros_like(var_idx)
        space_emb = self.space_emb(var_idx)

        return val_time_emb, space_emb, var_idx_true, mask

class TriangularCausalMask:
    def __init__(self, B, L, queries):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).type_as(queries.type(torch.bool)) # CHANGE.to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask:
    def __init__(self, B, H, L, index, scores, values):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).type_as(values.type(torch.bool)).triu(1) # CHANGE .to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].type_as(values) #.to(device)
        self._mask = indicator.view(scores.shape).type_as(values.type(torch.bool)) # .to(device)

    @property
    def mask(self):
        return self._mask

def create_subsequence_mask(o, r=0.15, lm=3, stateful=True, sync=False):
    # mask random subsequences of the inp
    # (borrowed from IBM codeflare)
    if r <= 0:
        return torch.zeros_like(o).bool()
    device = o.device
    if o.ndim == 2:
        o = o[None]
    n_masks, mask_dims, mask_len = o.shape
    if sync == "random":
        sync = random.random() > 0.5
    dims = 1 if sync else mask_dims
    if stateful:
        numels = n_masks * dims * mask_len
        pm = torch.tensor([1 / lm], device=device)
        pu = torch.clip(pm * (r / max(1e-6, 1 - r)), 1e-3, 1)
        zot, proba_a, proba_b = (
            (torch.as_tensor([False, True], device=device), pu, pm)
            if random.random() > pm
            else (torch.as_tensor([True, False], device=device), pm, pu)
        )
        max_len = max(
            1,
            2
            * torch.div(numels, (1 / pm + 1 / pu), rounding_mode="floor").long().item(),
        )
        for i in range(10):
            _dist_a = (Geometric(probs=proba_a).sample([max_len]) + 1).long()
            _dist_b = (Geometric(probs=proba_b).sample([max_len]) + 1).long()
            dist_a = _dist_a if i == 0 else torch.cat((dist_a, _dist_a), dim=0)
            dist_b = _dist_b if i == 0 else torch.cat((dist_b, _dist_b), dim=0)
            add = torch.add(dist_a, dist_b)
            if torch.gt(torch.sum(add), numels):
                break
        dist_len = torch.argmax((torch.cumsum(add, 0) >= numels).float()) + 1
        if dist_len % 2:
            dist_len += 1
        repeats = torch.cat((dist_a[:dist_len], dist_b[:dist_len]), -1).flatten()
        zot = zot.repeat(dist_len)
        mask = torch.repeat_interleave(zot, repeats)[:numels].reshape(
            n_masks, dims, mask_len
        )
    else:
        probs = torch.tensor(r, device=device)
        mask = Binomial(1, probs).sample((n_masks, dims, mask_len)).bool()
    if sync:
        mask = mask.repeat(1, mask_dims, 1)
    return mask


class ReconstructionDropout(nn.Module):
    def __init__(
        self,
        drop_full_timesteps=0.0,
        drop_standard=0.0,
        drop_seq=0.0,
        drop_max_seq_len=5,
        skip_all_drop=1.0,
    ):
        super().__init__()
        self.drop_full_timesteps = drop_full_timesteps
        self.drop_standard = drop_standard
        self.drop_seq = drop_seq
        self.drop_max_seq_len = drop_max_seq_len
        self.skip_all_drop = skip_all_drop

    def forward(self, y):
        bs, length, dim = y.shape
        # dev = y.device

        if self.training and self.skip_all_drop < 1.0:
            # mask full timesteps
            full_timestep_mask = torch.bernoulli(
                (1.0 - self.drop_full_timesteps) * torch.ones(bs, length, 1)
            ).type_as(y) # CHANGE.to(dev)

            # mask each element indp
            standard_mask = torch.bernoulli(
                (1.0 - self.drop_standard) * torch.ones(bs, length, dim)
            ).type_as(y) # .to(dev)

            # subsequence mask
            seq_mask = (
                1.0
                - create_subsequence_mask(
                    y.transpose(1, 2), r=self.drop_seq, lm=self.drop_max_seq_len
                )
                .transpose(1, 2)
                .float()
            )

            # skip all dropout occasionally so when there is no dropout
            # at test time the model has seen that before. (I am not sure
            # the usual activation strength adjustment makes sense here)
            skip_all_drop_mask = torch.bernoulli(
                1.0 - self.skip_all_drop * torch.ones(bs, 1, 1)
            ).type_as(y) #.to(dev)

            mask = 1.0 - (
                (1.0 - (full_timestep_mask * standard_mask * seq_mask))
                * skip_all_drop_mask
            )

            return y * mask
        else:
            return y

    def __repr__(self):
        return f"Timesteps {self.drop_full_timesteps}, Standard {self.drop_standard}, Seq (max len = {self.drop_max_seq_len}) {self.drop_seq}, Skip All Drop {self.skip_all_drop}"


class RandomMask(nn.Module):
    def __init__(self, prob, change_to_val):
        super().__init__()
        self.prob = prob
        self.change_to_val = change_to_val

    def forward(self, y):
        bs, length, dy = y.shape
        if not self.training or self.change_to_val is None:
            return y
        mask = torch.bernoulli((1.0 - self.prob) * torch.ones(bs, length, 1))
        mask.requires_grad = False
        mask = mask.type_as(y) # .to(y.device)
        masked_y = (y * mask) + (self.change_to_val * (1.0 - mask))
        return masked_y

    def __repr__(self):
        return f"RandomMask(prob = {self.prob}, val = {self.change_to_val}"


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(
            L_K, (L_Q, sample_k)
        )  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(
            -2
        )

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, values=V)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn) # CHANGE.to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log1p(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log1p(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.transpose(2, 1).contiguous(), attn


from performer_pytorch import FastAttention as _FastAttention


class PerformerAttention(_FastAttention):
    def __init__(
        self,
        mask_flag=False,
        dim_heads=None,
        ortho_scaling=0,
        feature_redraw_interval=1000,
        kernel="softmax",
    ):
        assert dim_heads is not None
        super().__init__(
            dim_heads=dim_heads,
            ortho_scaling=ortho_scaling,
            nb_features=max(100, int(dim_heads * log(dim_heads))),
            causal=mask_flag,
            generalized_attention=kernel == "relu",
            kernel_fn=nn.ReLU() if kernel == "relu" else "N/A",
        )
        self.redraw_interval = feature_redraw_interval
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        if self.training:
            if self.calls_since_last_redraw >= self.redraw_interval:
                self.redraw_projection_matrix(queries.device)
                self.calls_since_last_redraw.zero_()
            else:
                self.calls_since_last_redraw += 1

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        v = super().forward(queries, keys, values)

        return v.transpose(1, 2), None


class AttentionLayer(nn.Module):
    def __init__(
        self, 
        attention, 
        d_model, 
        d_queries_keys, 
        d_values,
        n_heads, 
        dropout_qkv=0.0,
        mix=False
    ):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention()
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.dropout_qkv(self.query_projection(queries)).view(B, L, H, -1)
        keys = self.dropout_qkv(self.key_projection(keys)).view(B, S, H, -1)
        values = self.dropout_qkv(self.value_projection(values)).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            attn_mask=attn_mask,
            output_attn=output_attn,
        )

        if output_attn and attn is None:
            onehot_values = (
                torch.eye(S).unsqueeze(0).repeat(B, 1, 1).unsqueeze(2).type_as(values) # CHANGE.to(values.device)
            )
            with torch.no_grad():
                attn, _ = self.inner_attention(
                    queries=queries,
                    keys=keys,
                    values=onehot_values,
                    attn_mask=attn_mask,
                )
                attn = attn.transpose(2, 1)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        if not output_attn:
            assert attn is None

        return self.out_projection(out), attn


class SpacetimeformerModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int, # 1 + self.num_feat_dynamic_real + len(self.time_features)
        num_time_features: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        # Spacetimeformer arguments
        max_seq_len: int = None,
        attn_factor: int = 5,
        d_model: int = 200,
        d_queries_keys: int = 30,
        d_values: int = 30,
        n_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 800,
        activation: str = "gelu",
        dropout_emb: float = 0.1,
        dropout_attn_matrix: float = 0.0,
        dropout_attn_out: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_qkv: float = 0.0,
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_attn_kernel: str = "relu",
        start_token_len: int = 0,
        time_emb_dim: int = 6,
        pos_emb_type: str = "abs",
        performer_redraw_interval: int = 100,
        attn_time_windows: int = 1,
        use_shifted_time_windows: bool = False,
        embed_method: str = "spatio-temporal",
        norm: str = "batch",
        use_final_norm: bool = True,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        null_value: float = None,
        pad_value: float = None,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
        recon_mask_skip_all: float = 1.0,
        recon_mask_max_seq_len: int = 5,
        recon_mask_drop_seq: float = 0.2,
        recon_mask_drop_standard: float = 0.1,
        recon_mask_drop_full: float = 0.05,
        # univariate input
        input_size: int = 1,
        target_shape: Optional[Tuple[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: Optional[str] = "std",
        num_parallel_samples: int = 100,
    ):
        super().__init__()
        ### START GLUON INTERFACE ###
        self.input_size = input_size
        self.target_shape = target_shape or (1,)
        self.num_time_features = num_time_features
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )

        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        # make sure zero is first lag
        # CHANGE
        # if 0 in self.lags_seq:
        #     del self.lags_seq[self.lags_seq.index(0)]
        # self.lags_seq.insert(0, 0)
        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        
        self.scaling = scaling
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_args_proj(d_model)

        ### END GLUON INTERFACE ###

        if num_encoder_layers:
            assert intermediate_downsample_convs <= num_encoder_layers - 1
        if embed_method == "temporal":
            assert (
                local_self_attn == "none"
            ), "local attention not compatible with Temporal-only embedding"
            assert (
                local_cross_attn == "none"
            ), "Local Attention not compatible with Temporal-only embedding"
            split_length_into = 1
        else:
            split_length_into = self.input_size

        self.pad_value = pad_value
        self.embed_method = embed_method
        self.start_token_len = start_token_len

        # generates random masks of context sequence for encoder to reconstruct
        recon_dropout = ReconstructionDropout(
            drop_full_timesteps=recon_mask_drop_full,
            drop_standard=recon_mask_drop_standard,
            drop_seq=recon_mask_drop_seq,
            drop_max_seq_len=recon_mask_max_seq_len,
            skip_all_drop=recon_mask_skip_all,
        )

        # embeddings. seperate enc/dec in case the variable indices are not aligned
        self.enc_embedding = Embedding(
            # d_y=self.input_size + self.num_feat_dynamic_real - self.num_time_features - 1,
            d_y=self.input_size,
            d_x_time=self.num_time_features, # time features excluding age
            d_x_feat=self.num_feat_dynamic_real - self.num_time_features - 1, # subtracdt time features and age
            d_x_static=sum(self.embedding_dimension) + self.num_feat_static_real + self.input_size * 2,
            d_y_lagged=self.input_size * (len(self.lags_seq) - 1), # subtract the zero lag
            d_model=d_model,
            time_emb_dim=time_emb_dim,
            downsample_convs=initial_downsample_convs,
            method=embed_method,
            null_value=null_value,
            pad_value=pad_value,
            start_token_len=start_token_len,
            is_encoder=True,
            position_emb=pos_emb_type,
            max_seq_len=max_seq_len,
            data_dropout=recon_dropout,
            use_val=use_val,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
        )
        self.dec_embedding = Embedding(
            d_y=self.input_size,
            d_x_time=self.num_time_features, # time features excluding age
            d_x_feat=self.num_feat_dynamic_real - self.num_time_features - 1, # subtract time features and age
            d_x_static=sum(self.embedding_dimension) + self.num_feat_static_real + self.input_size * 2,
            d_y_lagged=self.input_size * (len(self.lags_seq) - 1), # subtract the 0 lag 
            d_model=d_model,
            time_emb_dim=time_emb_dim,
            downsample_convs=initial_downsample_convs,
            method=embed_method,
            null_value=null_value,
            pad_value=pad_value,
            start_token_len=start_token_len,
            is_encoder=False,
            position_emb=pos_emb_type,
            max_seq_len=max_seq_len,
            data_dropout=None,
            use_val=use_val,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
        )

        # Select Attention Mechanisms
        attn_kwargs = {
            "d_model": d_model,
            "n_heads": n_heads,
            "d_qk": d_queries_keys,
            "d_v": d_values,
            "dropout_qkv": dropout_qkv,
            "dropout_attn_matrix": dropout_attn_matrix,
            "attn_factor": attn_factor,
            "performer_attn_kernel": performer_attn_kernel,
            "performer_redraw_interval": performer_redraw_interval,
        }
        
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    global_attention=self._attn_switch(
                        global_self_attn,
                        **attn_kwargs,
                    ),
                    local_attention=self._attn_switch(
                        local_self_attn,
                        **attn_kwargs,
                    ),
                    d_model=d_model,
                    d_y=self.input_size if embed_method == "spatio-temporal" else 1,
                    time_windows=attn_time_windows,
                    # encoder layers alternate using shifted windows, if applicable
                    time_window_offset=2
                    if use_shifted_time_windows and (l % 2 == 1)
                    else 0,
                    dim_feedforward=dim_feedforward,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                )
                for l in range(num_encoder_layers)
            ],
            conv_layers=[
                ConvBlock(split_length_into=split_length_into, d_model=d_model)
                for l in range(intermediate_downsample_convs)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )

        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    global_self_attention=self._attn_switch(
                        global_self_attn,
                        **attn_kwargs,
                    ),
                    local_self_attention=self._attn_switch(
                        local_self_attn,
                        **attn_kwargs,
                    ),
                    global_cross_attention=self._attn_switch(
                        global_cross_attn,
                        **attn_kwargs,
                    ),
                    local_cross_attention=self._attn_switch(
                        local_cross_attn,
                        **attn_kwargs,
                    ),
                    d_model=d_model,
                    time_windows=attn_time_windows,
                    # decoder layers alternate using shifted windows, if applicable
                    time_window_offset=2
                    if use_shifted_time_windows and (l % 2 == 1)
                    else 0,
                    dim_feedforward=dim_feedforward,
                    # temporal embedding effectively has 1 variable
                    # for the purposes of time windowing.
                    d_y=self.input_size if embed_method == "spatio-temporal" else 1,
                    # d_y=self.input_size if embed_method == "spatio-temporal" else 1,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                )
                for l in range(num_decoder_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )
        # import pandas as pd
        # pd.DataFrame({
        #     "source": "new_2",
        #     **{p_name: [p.shape] for p_name, p in self.named_parameters()}}).to_csv("/Users/ahenry/Desktop/params_new_2.csv")
        
        # self.reshape_layer = nn.Linear(self.prediction_length*(self.input_size * len(self.lags_seq) + self._number_of_features - (self.num_time_features + 1)), 
        #                                self.prediction_length)
        

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size * 2  # the log(scale) and log(abs(loc)) features
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        shift: int
            shift the lags by this amount back.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.lags_seq]

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            lag_index = max(0, lag_index) # CHANGE
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def _check_shapes(
        self,
        prior_input: torch.Tensor,
        inputs: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> None:
        assert len(prior_input.shape) == len(inputs.shape)
        assert (
            len(prior_input.shape) == 2 and self.input_size == 1
        ) or prior_input.shape[2] == self.input_size
        assert (len(inputs.shape) == 2 and self.input_size == 1) or inputs.shape[
            -1
        ] == self.input_size
        assert (
            features is None or features.shape[2] == self._number_of_features
        ), f"{features.shape[2]}, expected {self._number_of_features}"

    def create_network_inputs(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        # time feature
        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # target 
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, loc, scale = self.scaler(context, observed_context)

        inputs = (
            (torch.cat((past_target, future_target), dim=1) - loc) / scale
            if future_target is not None
            else (past_target - loc) / scale
        )

        inputs_length = (
            self._past_length + self.prediction_length
            if future_target is not None
            else self._past_length
        )
        assert inputs.shape[1] == inputs_length

        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        log_abs_loc = (
            loc.sign() * loc.abs().log1p()
            if self.input_size == 1
            else loc.squeeze(1).sign() * loc.squeeze(1).abs().log1p()
        )
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_abs_loc, log_scale),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, time_feat.shape[1], -1
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # self._check_shapes(prior_input, inputs, features)

        # sequence = torch.cat((prior_input, inputs), dim=1)
        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs,
            subsequences_length=subsequences_length,
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], -1
        )

        # lagged past_target or [past_target, future_target]; expanded_static_feat; time_feat (time features + age + feat_dynamic_real) 
        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        # reshaped_lagged_sequence = transformer_inputs[:, :, :self.input_size*self.lags_seq]
        # expanded_static_feat = transformer_inputs[:, self.input_size*self.lags_seq:-self.num_feat_dynamic_real]
        # time_feat = transformer_inputs[:, : -self.num_feat_dynamic_real:] 
        return transformer_inputs, loc, scale, static_feat

    def create_dim_string(self, param):
        return ' '.join([f"dy{s}" for s in range(len(param.shape[2:]))])
    
    def output_params(self, transformer_inputs):
        enc_vt_emb, enc_s_emb, _, enc_mask_seq  \
            = self.enc_embedding(
                y=transformer_inputs[:, :self.context_length,  :self.input_size], 
                y_lagged=transformer_inputs[:, :self.context_length, self.input_size:self.input_size*len(self.lags_seq)], 
                x_time=transformer_inputs[:, :self.context_length, -self.num_feat_dynamic_real:][:, :, :self.num_time_features], 
                x_age=transformer_inputs[:, :self.context_length, -self.num_feat_dynamic_real:][:, :, self.num_time_features:(self.num_time_features + 1)],
                x_feat=transformer_inputs[:, :self.context_length, -self.num_feat_dynamic_real:][:, :, self.num_time_features + 1:],
                x_static=transformer_inputs[:, :self.context_length, self.input_size*len(self.lags_seq):-self.num_feat_dynamic_real]
        )
        
        dec_vt_emb, dec_s_emb, _, dec_mask_seq \
            = self.dec_embedding(
                y=transformer_inputs[:, self.context_length:,  :self.input_size],
                y_lagged=transformer_inputs[:, self.context_length:, self.input_size:self.input_size*len(self.lags_seq)], 
                x_time=transformer_inputs[:, self.context_length:, -self.num_feat_dynamic_real:][:, :, :self.num_time_features], 
                x_age=transformer_inputs[:, self.context_length:, -self.num_feat_dynamic_real:][:, :, self.num_time_features:(self.num_time_features + 1)],
                x_feat=transformer_inputs[:, self.context_length:, -self.num_feat_dynamic_real:][:, :, self.num_time_features + 1:],
                x_static=transformer_inputs[:, self.context_length:, self.input_size*len(self.lags_seq):-self.num_feat_dynamic_real]
        )
            
        enc_out, _ = self.encoder(
            val_time_emb=enc_vt_emb,
            space_emb=enc_s_emb,
            self_mask_seq=enc_mask_seq,
            output_attn=False
        )

        if enc_mask_seq is not None:
            enc_dec_mask_seq = enc_mask_seq.clone()
        else:
            enc_dec_mask_seq = enc_mask_seq
        
        dec_output, _ = self.decoder(
                val_time_emb=dec_vt_emb,
                space_emb=dec_s_emb,
                cross=enc_out,
                self_mask_seq=dec_mask_seq,
                cross_mask_seq=enc_dec_mask_seq,
                output_cross_attn=False,
        ) # [batch, time * (input_size * len(lag_seq) + self._num_features), d_model]
        # Note rather than a linear project to 1 output dim for each flattened spatio-temporal value, output to number of distr params
        # dec_output = torch.transpose(dec_output, 1, 2)
        # dec_output = torch.transpose(self.reshape_layer(dec_output), 1, 2)
        forecast_out = self.param_proj(dec_output) # this outputs #distr_params (3) for each element in 1th dim
        forecast_out = tuple(
            rearrange(param, 
            f"batch (d_y length) {self.create_dim_string(param)} -> batch length d_y {self.create_dim_string(param)}", 
            **{f"dy{s}": v for s, v in enumerate(param.shape[2:])}, length=self.prediction_length) for param in forecast_out)
        forecast_out = tuple(param.squeeze(3) for param in forecast_out)
        return forecast_out

    @torch.jit.ignore
    def output_loss(
        self, params, future_target, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.loss(target=future_target, 
                                      distr_args=tuple(x.double() for x in sliced_params), 
                                      loc=loc, scale=scale)

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(tuple(x.double() for x in sliced_params), loc=loc, scale=scale)

    # for prediction
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor, # past time features + age + feat_dynamic_reals
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
        output_distr_params: Optional[dict] = {} 
    ) -> torch.Tensor:
        
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        encoder_inputs, loc, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values
        )
        # reshaped_lagged_sequence = encoder_inputs[:, :, :self.input_size*len(self.lags_seq)]
        # expanded_static_feat = encoder_inputs[:, :, self.input_size*len(self.lags_seq):-self.num_feat_dynamic_real]
        # time_feat = encoder_inputs[:, :, -self.num_feat_dynamic_real:] 

        # embed context sequence
        enc_vt_emb, enc_s_emb, enc_var_idx, enc_mask_seq = self.enc_embedding(
            y=encoder_inputs[:, :, :self.input_size], 
            y_lagged=encoder_inputs[:, :self.context_length, self.input_size:self.input_size*len(self.lags_seq)],
            x_time=encoder_inputs[:, :, -self.num_feat_dynamic_real:][:, :, :self.num_time_features], 
            x_age=encoder_inputs[:, :, -self.num_feat_dynamic_real:][:, :, self.num_time_features:self.num_time_features + 1], 
            x_feat=encoder_inputs[:, :, -self.num_feat_dynamic_real:][:, :, self.num_time_features + 1:],
            x_static=encoder_inputs[:, :, self.input_size*len(self.lags_seq):-self.num_feat_dynamic_real]
        )

        if enc_mask_seq is not None:
            enc_dec_mask_seq = enc_mask_seq.clone()
            repeated_enc_dec_mask_seq = enc_dec_mask_seq.repeat_interleave(
                                    repeats=self.num_parallel_samples, dim=0
                        )
        else:
            enc_dec_mask_seq = enc_mask_seq
            repeated_enc_dec_mask_seq = None

        # encode context sequence
        enc_out, _ = self.encoder(
            val_time_emb=enc_vt_emb,
            space_emb=enc_s_emb,
            self_mask_seq=enc_mask_seq,
            output_attn=False,
        )

        repeated_loc = loc.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_feat.shape[1], -1
        )

        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        
        if True:
            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=self.prediction_length,
                shift=1,
            )

            # shape [n_continuity_groups*n_parallel_samples, context_length, size of lagged repeated_past_target]
            lags_shape = lagged_sequence.shape

            # shape [n_continuity_groups*n_parallel_samples, context_length * size of lagged repeated_past_target]
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )
            
            # embed target sequence
            # TODO where does the future masking happen QUESTION
            rep_feat = repeated_features[:, :, expanded_static_feat.shape[-1]:]
            dec_vt_emb, dec_s_emb, _, dec_mask_seq = self.dec_embedding(
                y=reshaped_lagged_sequence[:, :, :self.input_size],
                y_lagged=reshaped_lagged_sequence[:, :, self.input_size:self.input_size*len(self.lags_seq)], 
                x_time=rep_feat[:, :, :self.num_time_features],
                x_age=rep_feat[:, :, self.num_time_features:self.num_time_features + 1],
                x_feat=rep_feat[:, :, self.num_time_features + 1:],
                x_static=repeated_features[:, :, :expanded_static_feat.shape[-1]]
            )
            
            # decode target sequence w/ encoded context, (batch_size, pred_len*(input_size * len(lag_seq) + self._input_features), output_dim)
            dec_output, _ = self.decoder(
                val_time_emb=dec_vt_emb,
                space_emb=dec_s_emb,
                cross=repeated_enc_out,
                self_mask_seq=dec_mask_seq,
                cross_mask_seq=repeated_enc_dec_mask_seq,
                output_cross_attn=False,
            )
            # Note shouldnt I be generating all predictions in the horizon in one go, rather than with ascestral sampling?
            #      YES don't need a loop, output_dim = pred_len * input_size * num_distr_params, select input_size from dec_output, then reshape
            params = self.param_proj(dec_output)
            params = tuple(
                rearrange(param, 
                f"batch (d_y length) {self.create_dim_string(param)} -> batch length d_y {self.create_dim_string(param)}", 
                **{f"dy{s}": v for s, v in enumerate(param.shape[2:])}, length=self.prediction_length) for param in params)
            params = tuple(param.squeeze(3) for param in params)
            # shapes [n_continuity_groups*n_parallel_samples, 1, input_size], [n_continuity_groups*n_parallel_samples, 1, input_size, rank], [n_continuity_groups*n_parallel_samples, 1, input_size]
            
            distr = self.output_distribution( # params batch of num_parallel_samples passed to distribution...all identical, so mean and stdev values output are all identical too
                params, scale=repeated_scale, loc=repeated_loc
            )
            
            if output_distr_params:
                if hasattr(distr, "base_dist"):
                    distr_params = list(distr.base_dist.arg_constraints.keys())
                    
                else:
                    distr_params = list(distr.arg_constraints.keys())
                return tuple(getattr(distr, output_distr_params[tgt_key])[::num_parallel_samples, :, :] 
                                            for tgt_key in distr_params)
            else:
                next_sample = distr.sample().float()
                return next_sample.reshape(
                (-1, num_parallel_samples, self.prediction_length) + (self.input_size,), 
                )
        else:
            # TODO HIGH can we implement asectral sampling?
            future_samples = []
            if output_distr_params:
                future_params = []
                
            for k in range(self.prediction_length):
                # self._check_shapes(repeated_past_target, next_sample, next_features)
                # sequence = torch.cat((repeated_past_target, next_sample), dim=1)
                
                lagged_sequence = self.get_lagged_subsequences(
                    sequence=repeated_past_target,
                    subsequences_length=1 + k,
                    shift=1,
                )

                # shape [n_continuity_groups*n_parallel_samples, context_length, size of lagged repeated_past_target]
                lags_shape = lagged_sequence.shape

                # shape [n_continuity_groups*n_parallel_samples, context_length * size of lagged repeated_past_target]
                reshaped_lagged_sequence = lagged_sequence.reshape(
                    lags_shape[0], lags_shape[1], -1
                )
                
                rep_feat = repeated_features[:, : k + 1, expanded_static_feat.shape[-1]:]
                rep_static_feats = repeated_features[:, : k + 1, :expanded_static_feat.shape[-1]] 
                dec_vt_emb, dec_s_emb, _, dec_mask_seq = self.dec_embedding(
                    y=reshaped_lagged_sequence[:, :, :self.input_size],
                    y_lagged=reshaped_lagged_sequence[:, :, self.input_size:self.input_size*len(self.lags_seq)], 
                    x_time=rep_feat[:, :, :self.num_time_features],
                    x_age=rep_feat[:, :, self.num_time_features:self.num_time_features + 1],
                    x_feat=rep_feat[:, :, self.num_time_features + 1:],
                    x_static=rep_static_feats
                )
                
                # decode target sequence w/ encoded context, (batch_size, pred_len*(input_size * len(lag_seq) + self._input_features), output_dim)
                dec_output, _ = self.decoder(
                    val_time_emb=dec_vt_emb,
                    space_emb=dec_s_emb,
                    cross=repeated_enc_out,
                    self_mask_seq=dec_mask_seq,
                    cross_mask_seq=repeated_enc_dec_mask_seq,
                    output_cross_attn=False,
                )
                
                # shapes [n_continuity_groups*n_parallel_samples, 1, input_size], [n_continuity_groups*n_parallel_samples, 1, input_size, rank], [n_continuity_groups*n_parallel_samples, 1, input_size]
                params = self.param_proj(dec_output)
                
                dim_string = lambda param: ' '.join([f"dy{s}" for s in range(len(param.shape[2:]))])
                params = tuple(
                    rearrange(param, 
                    f"batch (d_y length) {dim_string(param)} -> batch length d_y {dim_string(param)}", 
                    **{f"dy{s}": v for s, v in enumerate(param.shape[2:])}, length=self.prediction_length) for param in params)
                params = tuple(param.squeeze(3) for param in params)
                
                # adds scaled mean, variance, stddev params
                distr = self.output_distribution( # params batch of num_parallel_samples passed to distribution...all identical, so mean and stdev values output are all identical too
                    params, scale=repeated_scale, loc=repeated_loc
                )
                
                # attributes loc, cov_factor, cov_diag = params[0], params[1], params[2] before scaling
                if output_distr_params:
                    distr_params = list(distr.base_dist.arg_constraints.keys())
                    future_params.append(tuple(getattr(distr, output_distr_params[tgt_key])[::num_parallel_samples, :, :] 
                                            for tgt_key in distr_params))
                
                next_sample = distr.sample().float()
                repeated_past_target = torch.cat(
                    (repeated_past_target, (next_sample - repeated_loc) / repeated_scale),
                    dim=1,
                )
                future_samples.append(next_sample)
                
            if output_distr_params:
                return tuple(torch.cat([params[p] for params in future_params], dim=1) for p in range(len(distr_params)))
            else:
                concat_future_samples = torch.cat(future_samples, dim=1)
                return concat_future_samples.reshape(
                    (-1, num_parallel_samples, self.prediction_length) + self.target_shape,
                )

    def _attn_switch(
        self,
        attn_str: str,
        d_model: int,
        n_heads: int,
        d_qk: int,
        d_v: int,
        dropout_qkv: float,
        dropout_attn_matrix: float,
        attn_factor: int,
        performer_attn_kernel: str,
        performer_redraw_interval: int,
    ):
        if attn_str == "full":
            # standard full (n^2) attention
            Attn = AttentionLayer(
                attention=partial(FullAttention, attention_dropout=dropout_attn_matrix),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "prob":
            # Informer-style ProbSparse cross attention
            Attn = AttentionLayer(
                attention=partial(
                    ProbAttention,
                    factor=attn_factor,
                    attention_dropout=dropout_attn_matrix,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "performer":
            # Performer Linear Attention
            Attn = AttentionLayer(
                attention=partial(
                    PerformerAttention,
                    dim_heads=d_qk,
                    kernel=performer_attn_kernel,
                    feature_redraw_interval=performer_redraw_interval,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "none":
            Attn = None
        else:
            raise ValueError(f"Unrecognized attention str code '{attn_str}'")
        return Attn

class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, queries=queries)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x