from typing import Any, Dict, Iterable, List, Optional
import logging
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
from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDatamodule

import lightning

# Set up logging
logger = logging.getLogger(__name__)

from pytorch_transformer_ts.utils.step_scaling import resolve_steps

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
        use_lazyframe: bool = False,
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
        base_batch_size_for_scheduler_steps: int = 2048,
        base_limit_train_batches: Optional[int] = None,
        num_batches_per_epoch: Optional[int] = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        gradient_clip_val: float = 1000.0,
        warmup_steps: float = 0.1, # Warmup steps
        # --- Scheduler specific arguments ---
        eta_min_fraction: float = 0.01,  # Fraction of initial LR for cosine decay eta_min
        steps_to_decay: Optional[float] = 0.9,  # Optional manual T_max value for CosineAnnealingLR
        use_pytorch_dataloader: bool = False,
        **kwargs,
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 100,
            **trainer_kwargs,
        }
        super().__init__(trainer_kwargs=trainer_kwargs)

        self.use_lazyframe = use_lazyframe
        
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
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min_fraction = eta_min_fraction
        self.gradient_clip_val = gradient_clip_val
        self.warmup_steps = warmup_steps # Store warmup steps
        self.steps_to_decay = steps_to_decay # Store manual T_max value
        self.base_batch_size_for_scheduler_steps = base_batch_size_for_scheduler_steps
        self.base_limit_train_batches = base_limit_train_batches

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )
        
        self.use_pytorch_dataloader = use_pytorch_dataloader
        
        # Log any remaining kwargs
        if kwargs:
            logger.warning(f"SpacetimeformerEstimator received unused kwargs: {kwargs}")
    
    @staticmethod
    def get_params(trial, tuning_phase=None, dynamic_kwargs=None):
        # in paper: lr=1e-4, 3 encoder layers, 3 decoder layers, 4 heads, d_v=d_qk=30, d_model=200, d_ff=800, attn_factor=5, dropout_emb=0.2, dropout_qkv=0.0, dropout_attn_matrix=0.0, dropout_ff=0.3, dropout_attn_out=0.0
        # global_self_attn=global_cross_attn=local_self_attn=performer, activation=gelu, norm=batch, 
        # batch_size=128
        if dynamic_kwargs is None:
            dynamic_kwargs = {}
        d_qkv = trial.suggest_categorical("d_qkv", [16, 32, 64, 128])
        return {
            
            # --- Input Params ---
            # "context_length_factor": trial.suggest_categorical("context_length_factor", dynamic_kwargs.get("context_length_factor", [2, 3, 4])),
            "context_length_factor": trial.suggest_categorical("context_length_factor", dynamic_kwargs.get("context_length_factor", [5])),
            "batch_size": trial.suggest_categorical("batch_size", dynamic_kwargs.get("batch_size", [64, 128, 256, 512, 1024])),
            
            # --- Architecture Params ---
            "num_encoder_layers": trial.suggest_categorical("num_encoder_layers", dynamic_kwargs.get("num_encoder_layers", [2, 3, 4, 5])),
            "num_decoder_layers": trial.suggest_categorical("num_decoder_layers", dynamic_kwargs.get("num_decoder_layers", [2, 3, 4, 5])),
            "d_model": trial.suggest_categorical("d_model", dynamic_kwargs.get("d_model", [128, 256, 512])),
            "d_queries_keys": d_qkv,
            "d_values": d_qkv,
            "n_heads": trial.suggest_categorical("n_heads", dynamic_kwargs.get("n_heads", [4, 6, 8])),
            # "attn_factor": trial.suggest_categorical("attn_factor", dynamic_kwargs.get("attn_factor", [1, 3, 5])),
            
            # start_token_len: int = 0,
            # time_emb_dim: int = 6,
            # activation: str = "gelu",
            # pos_emb_type: str = "abs",
            # global_self_attn: str = "performer",
            # local_self_attn: str = "performer",
            # global_cross_attn: str = "performer",
            # local_cross_attn: str = "performer",
            # performer_attn_kernel: str = "relu",
            # performer_redraw_interval: int = 100,
            # attn_time_windows: int = 1,
            # use_shifted_time_windows: bool = False,
            # embed_method: str = "spatio-temporal",
            # norm: str = "batch",
            # use_final_norm: bool = True,
            # initial_downsample_convs: int = 0,
            # intermediate_downsample_convs: int = 0,
            # null_value: float = None,
            # pad_value: float = None,
            # use_val: bool = True,
            # use_time: bool = True,
            # use_space: bool = True,
            # use_given: bool = True,
            # recon_mask_skip_all: float = 1.0,
            # recon_mask_max_seq_len: int = 5,
            # recon_mask_drop_seq: float = 0.2,
            # recon_mask_drop_standard: float = 0.1,
            # recon_mask_drop_full: float = 0.05,
            
            # --- Optimizer Params ---
            "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=False),
            "weight_decay": trial.suggest_categorical("weight_decay", dynamic_kwargs.get("weight_decay", [0.0, 1e-8, 1e-6, 1e-4])),
            
            # --- Dropout & Clipping ---  
            "dropout_qkv": trial.suggest_float("dropout_qkv", 0.0, 0.3),
            "dropout_ff": trial.suggest_float("dropout_ff", 0.0, 0.3),
            "dropout_attn_out": trial.suggest_float("dropout_attn_out", 0.0, 0.3),
            # "dropout_attn_matrix": trial.suggest_float("dropout_attn_matrix", 0.0, 0.3), # only needed for Prob/Full attention
            "dropout_emb": trial.suggest_float("dropout_emb", 0.0, 0.3),
            "gradient_clip_val": trial.suggest_categorical("gradient_clip_val", dynamic_kwargs.get("gradient_clip_val", [0, 1.0, 3.0, 5.0, 10.0])),

            # --- LR Scheduler Params ---
            "eta_min_fraction": trial.suggest_float("eta_min_fraction", 1e-5, 0.05, log=True), # Tune eta_min fraction
        }
        
    def create_transformation(self, use_lazyframe=True) -> Transformation:
        if use_lazyframe is None and hasattr(self, "use_lazyframe"):
            use_lazyframe = self.use_lazyframe
        else:
            use_lazyframe = False
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
        if self.num_batches_per_epoch is not None:
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
        
    def create_pytorch_data_module(self,
        train_data_path: str,
        val_data_path: str,
        **kwargs
        ) -> lightning.pytorch.LightningDataModule:
        
        # Parameters
        #     ----------
        #     data_path
        #         Path to the pickle file containing training data.
        
        
        return WindForecastingDatamodule(
            train_data_path=train_data_path, 
            val_data_path=val_data_path, 
            train_sampler=self.train_sampler, 
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            time_features=self.time_features,
            val_sampler=None, 
            train_repeat=self.num_batches_per_epoch is not None,
            val_repeat=False,
            batch_size=self.batch_size,
            num_workers=kwargs.get('num_workers', 4), 
            pin_memory=kwargs.get('pin_memory', True)
            )

    # def create_pytorch_training_data_loader(
    #     self,
    #     data_path: str,
    #     module: SpacetimeformerLightningModule,
    #     **kwargs,
    #     ) -> torch.utils.data.DataLoader:
    #     """
    #     Create a PyTorch DataLoader for training.
        
    #     Parameters
    #     ----------
    #     data_path
    #         Path to the pickle file containing training data.
    #     module
    #         The Spacetimeformer lightning module.
        
    #     Returns
    #     -------
    #     A PyTorch DataLoader for training.
    #     """
    #     # Import here to avoid circular imports
    #     from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDataset
        
    #     data = WindForecastingDataset(
    #         data_path=data_path,
    #         context_length=self.context_length,
    #         prediction_length=self.prediction_length,
    #         time_features=self.time_features,
    #         sampler=self.train_sampler,  # Pass the GluonTS sampler
    #         repeat=self.num_batches_per_epoch is not None # will just repeat over same dataset if we only provide one
    #     )
        
    #     # Return DataLoader - PyTorch Lightning will add DistributedSampler automatically
    #     return torch.utils.data.DataLoader(
    #         data,
    #         batch_size=self.batch_size,
    #         shuffle=False,  # Will be overridden by DistributedSampler in DDP
    #         # worker_init_fn=self.__class__._worker_init_fn,
    #         num_workers=kwargs.get('num_workers', 4),
    #         pin_memory=kwargs.get('pin_memory', True),
    #         persistent_workers=kwargs.get('persistent_workers', True),
    #         # drop_last=True,  # Important for DDP to avoid uneven batch sizes
    #     )
    
    
    # def create_pytorch_validation_data_loader(
    #     self,
    #     data_path: str,
    #     module: SpacetimeformerLightningModule,
    #     **kwargs,
    # ) -> torch.utils.data.DataLoader:
    #     """
    #     Create a PyTorch DataLoader for validation.
        
    #     Parameters
    #     ----------
    #     data_path
    #         Path to the pickle file containing validation data.
    #     module
    #         The Spacetimeformer lightning module.
        
    #     Returns
    #     -------
    #     A PyTorch DataLoader for validation.
    #     """
    #     # Import here to avoid circular imports
    #     from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingInferenceDataset
        
    #     dataset = WindForecastingInferenceDataset(
    #         data_path=data_path,
    #         context_length=self.context_length,
    #         prediction_length=self.prediction_length,
    #         time_features=self.time_features,
    #         repeat=False
    #     )
        
    #     # Return DataLoader - PyTorch Lightning will add DistributedSampler automatically
    #     return torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,  # Never shuffle validation data
    #         # worker_init_fn=self.__class__._worker_init_fn,
    #         num_workers=kwargs.get('num_workers', 4),
    #         pin_memory=kwargs.get('pin_memory', True),
    #         persistent_workers=kwargs.get('persistent_workers', True),
    #         skip_indices=kwargs.get("skip_indices", 1)
    #         # drop_last=False,  # Keep all validation samples
    #     )

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
            num_feat_dynamic_real=(0 if self.use_pytorch_dataloader else 1) # 1 is for age
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
        
        # Calculate absolute steps from fractional values if needed
        # This allows setting warmup/decay as fractions of each training stage
        max_epochs = self.trainer_kwargs.get("max_epochs", 100)  # Default to 100 if not specified
        
        # Calculate effective batches per epoch considering limit_train_batches and DDP
        effective_batches_per_epoch = self.num_batches_per_epoch
        
        # Adjust for distributed training (DDP) - data is split across GPUs
        strategy = self.trainer_kwargs.get("strategy")
        devices = self.trainer_kwargs.get("devices", 1)
        
        # Check if using DDP strategy (handles both string and object forms)
        is_ddp = False
        if strategy == "ddp":
            is_ddp = True
        elif hasattr(strategy, "__class__") and "DDP" in strategy.__class__.__name__:
            is_ddp = True
        
        if is_ddp and devices > 1:
            # In DDP, each GPU processes 1/devices of the data per epoch
            original_batches = effective_batches_per_epoch
            effective_batches_per_epoch = effective_batches_per_epoch // devices
            logger.info(f"DDP detected ({devices} GPUs): adjusting steps per epoch {original_batches:,} -> {effective_batches_per_epoch:,}")
        elif devices > 1:
            logger.info(f"Multi-GPU detected ({devices} devices) but strategy={strategy} - no step adjustment applied")
        
        # First, check for trainer limit_train_batches setting
        limit_train_batches = self.trainer_kwargs.get("limit_train_batches", None)
        if limit_train_batches is not None and limit_train_batches != "null":
            # If limit_train_batches is set, that becomes our effective batch count
            try:
                limit_value = int(limit_train_batches)
                if limit_value > 0:
                    original_batches = effective_batches_per_epoch
                    effective_batches_per_epoch = min(effective_batches_per_epoch or limit_value, limit_value)
                    logger.info(f"limit_train_batches detected: {original_batches} -> {effective_batches_per_epoch}")
            except (ValueError, TypeError):
                # If it's not a valid number, ignore it
                pass
        
        if effective_batches_per_epoch:
            # Calculate total steps per stage using the correctly adjusted batch count
            steps = max_epochs * effective_batches_per_epoch if effective_batches_per_epoch else 0
            
        logger.info(f"Training schedule calculation:")
        logger.info(f"  Max epochs: {max_epochs}")
        logger.info(f"  Effective steps per epoch: {effective_batches_per_epoch:,}")
        
        resolved_warmup = resolve_steps(self.warmup_steps, steps, "warmup_steps")
        resolved_decay = resolve_steps(self.steps_to_decay, steps, "steps_to_decay")
        
        # return SpacetimeformerLightningModule(model=model, loss=self.loss)
        # return SpacetimeformerLightningModule(model=model)
        return SpacetimeformerLightningModule(
            model_config=model_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eta_min_fraction=self.eta_min_fraction,  # Pass eta_min fraction
            # Pass gradient clipping val
            gradient_clip_val=self.gradient_clip_val,
            # Pass training stage params
            warmup_steps=resolved_warmup, # Pass warmup steps
            steps_to_decay=resolved_decay, # Pass manual T_max value
            # Pass batch size parameters for scheduler step scaling
            batch_size=self.batch_size,
            base_batch_size_for_scheduler_steps=self.base_batch_size_for_scheduler_steps,
            base_limit_train_batches=self.base_limit_train_batches,
            # Pass num_batches_per_epoch for scheduler calculations
            num_batches_per_epoch=self.num_batches_per_epoch
        )
