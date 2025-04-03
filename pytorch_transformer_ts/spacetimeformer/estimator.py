from typing import Any, Dict, Iterable, List, Optional

import polars as pl
import numpy as np
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
# from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    AsLazyFrame,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler

from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
from pytorch_transformer_ts.spacetimeformer.module import SpacetimeformerModel

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class SpacetimeformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        # Spacetimeformer arguments
        attn_factor: int = 5,
        start_token_len: int = 0,
        time_emb_dim: int = 6,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 800,
        d_model: int = 200,
        d_queries_keys: int = 30,
        d_values: int = 30,
        n_heads: int = 4,
        input_size: int = 1,
        activation: str = "gelu",
        dropout_emb: float = 0.1,
        dropout_attn_matrix: float = 0.0,
        dropout_attn_out: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_qkv: float = 0.0,
        pos_emb_type: str = "abs",
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_attn_kernel: str = "relu",
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
        out_dim: int = None,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
        recon_mask_skip_all: float = 1.0,
        recon_mask_max_seq_len: int = 5,
        recon_mask_drop_seq: float = 0.2,
        recon_mask_drop_standard: float = 0.1,
        recon_mask_drop_full: float = 0.05,
        # loss: DistributionLoss = NegativeLogLikelihood(),
        scaling: Optional[str] = "std",
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 100,
            **trainer_kwargs,
        }
        super().__init__(trainer_kwargs=trainer_kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape
        self.distr_output.args_dim = {k: int(v / input_size) for k, v in self.distr_output.args_dim.items()}
        self.distr_output.dim = 1
        
        self.max_seq_len = self.context_length + self.prediction_length
        
        # self.loss = loss
        self.attn_factor = attn_factor
        self.start_token_len = start_token_len
        self.time_emb_dim = time_emb_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.d_model = d_model
        self.d_queries_keys = d_queries_keys
        self.d_values = d_values
        self.n_heads = n_heads
        self.dropout_emb = dropout_emb
        self.dropout_attn_matrix = dropout_attn_matrix
        self.dropout_attn_out = dropout_attn_out
        self.dropout_ff = dropout_ff
        self.dropout_qkv = dropout_qkv
        self.pos_emb_type = pos_emb_type
        self.global_self_attn = global_self_attn
        self.local_self_attn = local_self_attn
        self.global_cross_attn = global_cross_attn
        self.local_cross_attn = local_cross_attn
        self.performer_attn_kernel = performer_attn_kernel
        self.performer_redraw_interval = performer_redraw_interval
        self.attn_time_windows = attn_time_windows
        self.use_shifted_time_windows = use_shifted_time_windows
        self.embed_method = embed_method
        self.activation = activation
        self.norm = norm
        self.use_final_norm = use_final_norm
        self.initial_downsample_convs = initial_downsample_convs
        self.intermediate_downsample_convs = intermediate_downsample_convs
        self.null_value = null_value
        self.pad_value = pad_value
        self.out_dim = out_dim
        self.use_val = use_val
        self.use_time = use_time
        self.use_space = use_space
        self.use_given = use_given
        self.recon_mask_skip_all = recon_mask_skip_all
        self.recon_mask_max_seq_len = recon_mask_max_seq_len
        self.recon_mask_drop_seq = recon_mask_drop_seq
        self.recon_mask_drop_standard = recon_mask_drop_standard
        self.recon_mask_drop_full = recon_mask_drop_full

        self.input_size = input_size
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.lags_seq = lags_seq
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )
    
    @staticmethod
    def get_params(trial, context_length_choices):
        """ generate dictionary of tunable parameters compatible with optuna """
        # in paper: lr=1e-4, 3 encoder layers, 3 decoder layers, 4 heads, d_v=d_qk=30, d_model=200, d_ff=800, attn_factor=5, dropout_emb=0.2, dropout_qkv=0.0, dropout_attn_matrix=0.0, dropout_ff=0.3, dropout_attn_out=0.0
        # global_self_attn=global_cross_attn=local_self_attn=performer, activation=gelu, norm=batch, 
        # batch_size=128
        d_qkv = trial.suggest_categorical("d_model", [20, 30, 40])
        return {
            "context_length": trial.suggest_categorical("context_length", context_length_choices),
            # "max_epochs": trial.suggest_int("max_epochs", 1, 10, 2),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_encoder_layers": trial.suggest_categorical("num_encoder_layers", [3, 5, 6]),
            "num_decoder_layers": trial.suggest_categorical("num_decoder_layers", [3, 5, 6]),
            "d_model": trial.suggest_categorical("d_model", [100, 150, 200]),
            "d_queries_keys": d_qkv,
            "d_values": d_qkv,
            "n_heads": trial.suggest_categorical("n_heads", [4, 6, 8])
            # "num_batches_per_epoch":trial.suggest_int("num_batches_per_epoch", 100, 200, 100),   
        }
        
    def create_transformation(self, use_lazyframe=True) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                # AsNumpyArray(
                #     field=FieldName.TARGET,
                #     # in the following line, we add 1 for the time dimension
                #     expected_ndim=1 + len(self.distr_output.event_shape),
                # ),
                # AsLazyFrame(
                #     field=FieldName.FEAT_STATIC_CAT,
                #     expected_ndim=1,
                #     dtype=pl.Int
                # ),
                # AsLazyFrame(
                #     field=FieldName.FEAT_STATIC_REAL,
                #     expected_ndim=1
                # ),
                AsLazyFrame(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape)
                ) if use_lazyframe else AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension 
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    # expected_ndim=1 + 1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=pl.Float32 if use_lazyframe else np.float32
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
            ]
        )

    def _create_instance_splitter(self, module: SpacetimeformerLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: SpacetimeformerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        # TODO can set number of epochs/steps per epoch in PyTorchLightningEstimator
        data = Cyclic(data).stream() # continuously samples windows from dataset, bc these windows are sampled w replacement, so total number of windows is large, define epoch as 100 steps, each step samples some number of batches
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: SpacetimeformerLightningModule,
        **kwargs,
    ) -> Iterable:
        # want val dataloader to end, go over each time series, get last window and calculate likelihood wrt future inputs, deterministic
        # WARNING: could be large if not limited by training arguments
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: SpacetimeformerLightningModule,
        **kwargs
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module.model,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs
        )

    def create_lightning_module(self) -> SpacetimeformerLightningModule:
        model_params = dict(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=1
            + self.num_feat_dynamic_real
            + len(self.time_features),
            num_time_features=len(self.time_features),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            # Spacetimeformer arguments
            attn_factor=self.attn_factor,
            max_seq_len=self.max_seq_len,
            start_token_len=self.start_token_len,
            time_emb_dim=self.time_emb_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            d_model=self.d_model,
            d_queries_keys=self.d_queries_keys,
            d_values=self.d_values,
            n_heads=self.n_heads,
            dropout_emb=self.dropout_emb,
            dropout_attn_matrix=self.dropout_attn_matrix,
            dropout_attn_out=self.dropout_attn_out,
            dropout_ff=self.dropout_ff,
            dropout_qkv=self.dropout_qkv,
            pos_emb_type=self.pos_emb_type,
            global_self_attn=self.global_self_attn,
            local_self_attn=self.local_self_attn,
            global_cross_attn=self.global_cross_attn,
            local_cross_attn=self.local_cross_attn,
            performer_attn_kernel=self.performer_attn_kernel,
            performer_redraw_interval=self.performer_redraw_interval,
            attn_time_windows=self.attn_time_windows,
            use_shifted_time_windows=self.use_shifted_time_windows,
            embed_method=self.embed_method,
            activation=self.activation,
            norm=self.norm,
            use_final_norm=self.use_final_norm,
            initial_downsample_convs=self.initial_downsample_convs,
            intermediate_downsample_convs=self.intermediate_downsample_convs,
            null_value=self.null_value,
            pad_value=self.pad_value,
            out_dim=self.out_dim,
            use_val=self.use_val,
            use_time=self.use_time,
            use_space=self.use_space,
            use_given=self.use_given,
            recon_mask_skip_all=self.recon_mask_skip_all,
            recon_mask_max_seq_len=self.recon_mask_max_seq_len,
            recon_mask_drop_seq=self.recon_mask_drop_seq,
            recon_mask_drop_standard=self.recon_mask_drop_standard,
            recon_mask_drop_full=self.recon_mask_drop_full,
            ###########
            input_size=self.input_size,
            target_shape=self.target_shape,
            distr_output=self.distr_output,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )

        # return SpacetimeformerLightningModule(model=model, loss=self.loss)
        # return SpacetimeformerLightningModule(model=model)
        return SpacetimeformerLightningModule(model=model_params)
