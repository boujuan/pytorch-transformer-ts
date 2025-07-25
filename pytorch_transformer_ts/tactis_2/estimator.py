from math import log
from typing import Any, Dict, Iterable, List, Optional
import logging

import polars as pl
import numpy as np

import lightning
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
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
    VstackFeatures
)
from gluonts.transform.sampler import InstanceSampler
# from gluonts.transform.field import RenameFields
from gluonts.model.forecast_generator import SampleForecastGenerator

from .module import TACTiS2Model

from .lightning_module import TACTiS2LightningModule

# Set up logging
logger = logging.getLogger(__name__)

from pytorch_transformer_ts.utils.step_scaling import resolve_steps
from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDatamodule

# Define standard field names for different operations
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

class TACTiS2Estimator(PyTorchLightningEstimator):
    """
    Estimator for TACTiS2 model.
    
    This class handles training and inference for the TACTiS2 model, following the
    GluonTS estimator pattern.
    """
    
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int,
        # --- Scheduler specific arguments ---
        eta_min_fraction_s1: float = 0.01,  # Fraction of initial LR for Stage 1 cosine decay eta_min
        eta_min_fraction_s2: float = 0.01,  # Fraction of initial LR for Stage 2 cosine decay eta_min
        steps_to_decay_s1 = None,  # T_max for Stage 1 (int=absolute, float=fraction of stage)
        steps_to_decay_s2 = None,  # T_max for Stage 2 (int=absolute, float=fraction of stage)
        # --- TACTiS2 specific arguments ---
        # Passed directly to TACTiS2Model/TACTiS
        flow_series_embedding_dim: int = 5,
        copula_series_embedding_dim: int = 48,
        flow_input_encoder_layers: int = 6, # Marginal CDF Encoder input layers
        copula_input_encoder_layers: int = 1, # Attentional Copula Encoder input layers
        marginal_embedding_dim_per_head: int = 8,
        marginal_num_heads: int = 5,
        marginal_num_layers: int = 4, # Marginal CDF Encoder transformer layers
        copula_embedding_dim_per_head: int = 8,
        copula_num_heads: int = 5,
        copula_num_layers: int = 2, # Attentional Copula Encoder transformer layers
        decoder_dsf_num_layers: int = 2,
        decoder_dsf_hidden_dim: int = 256,
        decoder_mlp_num_layers: int = 3,
        decoder_mlp_hidden_dim: int = 16,
        decoder_transformer_num_layers: int = 3,
        decoder_transformer_embedding_dim_per_head: int = 16,
        decoder_transformer_num_heads: int = 6,
        decoder_num_bins: int = 50, # Corresponds to AttentionalCopula resolution
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        loss_normalization: str = "series",
        encoder_type: str = "standard",
        # Attentional Copula specific MLP params (passed down) - Aligned with AttentionalCopula class
        ac_mlp_num_layers: int = 2, # Default: Number of layers in AC's internal MLP
        ac_mlp_dim: int = 128,      # Default: Dimension of AC's internal MLP layers
        stage2_activation_function: str = "ReLU", # Default: Activation in Stage 2 components (copula input encoder, copula main encoder, AC MLP)
        stage1_activation_function: str = "ReLU", # Activation function for Stage 1 components (flow input encoder, flow main encoder, marginal conditioner)
        # Passed to TACTiS2LightningModule
        initial_stage: int = 1,
        stage2_start_epoch: int = 10,
        lr_stage1: float = 1.8e-3,
        lr_stage2: float = 7.0e-4,
        weight_decay_stage1: float = 0.0,
        weight_decay_stage2: float = 0.0,
        dropout_rate: float = 0.1,
        gradient_clip_val_stage1: float = 1000.0,
        gradient_clip_val_stage2: float = 1000.0,
        warmup_steps_s1 = 1000, # Warmup steps for stage 1 (int=absolute, float=fraction of stage)
        warmup_steps_s2 = 500,  # Warmup steps for stage 2 (int=absolute, float=fraction of stage)
        # General Estimator arguments
        use_lazyframe: bool = False,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        scaling: Optional[str] = "std", # External scaling through GluonTS
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
        input_size: int = 1, # Number of target series
        use_pytorch_dataloader: bool = False,
        **kwargs,        
    ) -> None:
        # Prepare base trainer kwargs
        trainer_kwargs = {
            "max_epochs": 100,
            **trainer_kwargs,
        }

        # Note: DDP strategy configuration is handled by run_model.py
        super().__init__(trainer_kwargs=trainer_kwargs)
        
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        
        self.use_lazyframe = use_lazyframe
        
        # --- Store TACTiS2 specific parameters ---
        # Model architecture params
        self.flow_series_embedding_dim = flow_series_embedding_dim
        self.copula_series_embedding_dim = copula_series_embedding_dim
        self.flow_input_encoder_layers = flow_input_encoder_layers
        self.copula_input_encoder_layers = copula_input_encoder_layers
        self.marginal_embedding_dim_per_head = marginal_embedding_dim_per_head
        self.marginal_num_heads = marginal_num_heads
        self.marginal_num_layers = marginal_num_layers
        self.copula_embedding_dim_per_head = copula_embedding_dim_per_head
        self.copula_num_heads = copula_num_heads
        self.copula_num_layers = copula_num_layers
        self.decoder_dsf_num_layers = decoder_dsf_num_layers
        self.decoder_dsf_hidden_dim = decoder_dsf_hidden_dim
        self.decoder_mlp_num_layers = decoder_mlp_num_layers
        self.decoder_mlp_hidden_dim = decoder_mlp_hidden_dim
        self.decoder_transformer_num_layers = decoder_transformer_num_layers
        self.decoder_transformer_embedding_dim_per_head = decoder_transformer_embedding_dim_per_head
        self.decoder_transformer_num_heads = decoder_transformer_num_heads
        self.decoder_num_bins = decoder_num_bins
        self.bagging_size = bagging_size
        self.input_encoding_normalization = input_encoding_normalization
        self.loss_normalization = loss_normalization
        self.encoder_type = encoder_type
        # Store AC params (aligned with AttentionalCopula class)
        self.ac_mlp_num_layers = ac_mlp_num_layers
        self.ac_mlp_dim = ac_mlp_dim
        self.stage2_activation_function = stage2_activation_function
        self.stage1_activation_function = stage1_activation_function
        # Training stage / optimizer params
        self.initial_stage = initial_stage
        self.stage2_start_epoch = stage2_start_epoch
        self.eta_min_fraction_s1 = eta_min_fraction_s1  # Store eta_min fractions
        self.eta_min_fraction_s2 = eta_min_fraction_s2
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.weight_decay_stage1 = weight_decay_stage1
        self.weight_decay_stage2 = weight_decay_stage2
        self.dropout_rate = dropout_rate
        self.gradient_clip_val_stage1 = gradient_clip_val_stage1
        self.gradient_clip_val_stage2 = gradient_clip_val_stage2
        self.warmup_steps_s1 = warmup_steps_s1 # Store stage 1 warmup steps
        self.warmup_steps_s2 = warmup_steps_s2 # Store stage 2 warmup steps
        self.steps_to_decay_s1 = steps_to_decay_s1 # Store manual T_max value for stage 1
        self.steps_to_decay_s2 = steps_to_decay_s2 # Store manual T_max value for stage 2
        self.base_batch_size_for_scheduler_steps = base_batch_size_for_scheduler_steps
        self.base_limit_train_batches = base_limit_train_batches
 
        # Common parameters
        self.input_size = input_size
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality if cardinality and num_feat_static_cat > 0 else [1]
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.num_parallel_samples = num_parallel_samples
        
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        
        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )
        
        self.lags_seq = lags_seq or [0]
        
        if time_features is None:
            self.time_features = time_features_from_frequency_str(self.freq)
        else:
            self.time_features = time_features
            
        self.use_pytorch_dataloader = use_pytorch_dataloader
        
        # Log any remaining kwargs
        if kwargs:
            logger.warning(f"TACTiS2Estimator received unused kwargs: {kwargs}")
            
    @staticmethod
    def get_params(trial, tuning_phase: int = 0, dynamic_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get parameters for hyperparameter tuning.
        
        Parameters
        ----------
        trial
            Optuna trial object.
        tuning_phase
            The current phase of tuning (default: 0).
        dynamic_kwargs
            Optional dictionary of dynamic keyword arguments.
        
        Returns
        -------
        Dict of parameter values.
        """
        if dynamic_kwargs is None:
            dynamic_kwargs = {}
        # Optional logging
        logger.debug(f"get_params called with tuning_phase={tuning_phase}, dynamic_kwargs={dynamic_kwargs}")
        if dynamic_kwargs and 'resample_freq' in dynamic_kwargs:
            logger.debug(f"Available resample frequencies: {dynamic_kwargs['resample_freq']}")
            # Could potentially use dynamic_kwargs['resample_freq'] to adjust search space
            
        params = {
            # --- General ---
            # "context_length_factor": trial.suggest_categorical("context_length_factor", dynamic_kwargs.get("context_length_factor", [3, 4, 5])),
            "context_length_factor": trial.suggest_categorical("context_length_factor", dynamic_kwargs.get("context_length_factor", [5])),
            "encoder_type": trial.suggest_categorical("encoder_type", ["standard", "temporal"]),
            "stage2_activation_function": trial.suggest_categorical("stage2_activation_function", dynamic_kwargs.get("stage2_activation_function", ["relu"])), # Tune activation for Stage 2 components
            "stage1_activation_function": trial.suggest_categorical("stage1_activation_function", dynamic_kwargs.get("stage1_activation_function", ["relu"])),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]), # Tune batch size

            # --- Marginal CDF Encoder ---
            "marginal_embedding_dim_per_head": trial.suggest_categorical("marginal_embedding_dim_per_head", dynamic_kwargs.get("marginal_embedding_dim_per_head", [16, 32, 64, 128, 256, 512])),
            "marginal_num_heads": trial.suggest_int("marginal_num_heads", 2, 6),
            "marginal_num_layers": trial.suggest_int("marginal_num_layers", 3, 5),
            "flow_input_encoder_layers": trial.suggest_int("flow_input_encoder_layers", 3, 5),
            "flow_series_embedding_dim": trial.suggest_categorical("flow_series_embedding_dim", dynamic_kwargs.get("flow_series_embedding_dim", [5, 8, 16, 32, 64, 128, 256])), # Renamed from marginal_ts_embedding_dim

            # --- Attentional Copula Encoder ---
            "copula_embedding_dim_per_head": trial.suggest_categorical("copula_embedding_dim_per_head", dynamic_kwargs.get("copula_embedding_dim_per_head", [8, 16, 32, 64, 128, 256])),
            "copula_num_heads": trial.suggest_int("copula_num_heads", 2, 6),
            "copula_num_layers": trial.suggest_int("copula_num_layers", 1, 3),
            "copula_input_encoder_layers": trial.suggest_int("copula_input_encoder_layers", 2, 4),
            "copula_series_embedding_dim": trial.suggest_categorical("copula_series_embedding_dim", dynamic_kwargs.get("copula_series_embedding_dim", [16, 32, 48, 64, 128, 256])), # Renamed from copula_ts_embedding_dim

            # --- Attentional Copula MLP (Aligned with AttentionalCopula class) ---
            "ac_mlp_num_layers": trial.suggest_int("ac_mlp_num_layers", 3, 6), # Tune number of layers
            "ac_mlp_dim": trial.suggest_categorical("ac_mlp_dim", dynamic_kwargs.get("ac_mlp_dim", [32, 64, 128, 256])), # Tune layer dimension

            # --- Decoder ---
            "decoder_dsf_num_layers": trial.suggest_int("decoder_dsf_num_layers", 1, 4),
            "decoder_dsf_hidden_dim": trial.suggest_categorical("decoder_dsf_hidden_dim", dynamic_kwargs.get("decoder_dsf_hidden_dim", [48, 64, 128, 256, 512])),
            "decoder_mlp_num_layers": trial.suggest_int("decoder_mlp_num_layers", 2, 5),
            "decoder_mlp_hidden_dim": trial.suggest_categorical("decoder_mlp_hidden_dim", dynamic_kwargs.get("decoder_mlp_hidden_dim", [8, 16, 32, 48, 64, 128, 256])),
            "decoder_transformer_num_layers": trial.suggest_int("decoder_transformer_num_layers", 2, 5),
            "decoder_transformer_embedding_dim_per_head": trial.suggest_categorical("decoder_transformer_embedding_dim_per_head", dynamic_kwargs.get("decoder_transformer_embedding_dim_per_head", [32, 64, 128])),
            "decoder_transformer_num_heads": trial.suggest_int("decoder_transformer_num_heads", 3, 5),
            "decoder_num_bins": trial.suggest_categorical("decoder_num_bins", dynamic_kwargs.get("decoder_num_bins", [50, 100, 200, 300])), # Corresponds to AttentionalCopula resolution

            # --- Optimizer Params ---
            "lr_stage1": trial.suggest_float("lr_stage1", 2e-6, 1e-5, log=True),
            "lr_stage2": trial.suggest_float("lr_stage2", 1e-6, 9e-6, log=True),
            "weight_decay_stage1": trial.suggest_categorical("weight_decay_stage1", dynamic_kwargs.get("weight_decay_stage1", [0.0, 1e-6, 1e-7])),
            "weight_decay_stage2": trial.suggest_categorical("weight_decay_stage2", dynamic_kwargs.get("weight_decay_stage2", [0.0, 2e-5, 1e-5, 5e-6, 1e-6])),

            # --- Dropout & Clipping ---
            "dropout_rate": trial.suggest_categorical("dropout_rate", dynamic_kwargs.get("dropout_rate", [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015])), # Tune dropout rate
            "gradient_clip_val_stage1": trial.suggest_categorical("gradient_clip_val_stage1", dynamic_kwargs.get("gradient_clip_val_stage1", [0, 1.0, 3.0, 5.0])),
            "gradient_clip_val_stage2": trial.suggest_categorical("gradient_clip_val_stage2", dynamic_kwargs.get("gradient_clip_val_stage2", [0, 1.0, 3.0, 5.0, 10.0])),

            # --- LR Scheduler Params ---
            "eta_min_fraction_s1": trial.suggest_float("eta_min_fraction_s1", 1e-3, 0.05, log=True), # Tune eta_min fraction for Stage 1
            "eta_min_fraction_s2": trial.suggest_float("eta_min_fraction_s2", 1e-5, 0.01, log=True), # Tune eta_min fraction for Stage 2

        }
        return params
    
    def create_transformation(self, use_lazyframe=True) -> Transformation:
        """
        Create the transformation pipeline.
        This is crucial for converting raw data into the format needed by the model.
        
        Parameters
        ----------
        use_lazyframe
            Whether to use LazyFrames for data processing (default is True)
            
        Returns
        -------
        A transformation pipeline that takes the raw data and converts it into the format
        expected by the model.
        """
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
                # Target field transformation - Ensure it's treated as 2D (time, num_series)
                AsLazyFrame(
                    field=FieldName.TARGET,
                    expected_ndim=2
                ) if use_lazyframe else AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
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
    
    def _create_instance_splitter(self, module: TACTiS2LightningModule, mode: str):
        """
        Create an instance splitter for creating training/validation examples.
        
        Parameters
        ----------
        module
            The TACTiS2 lightning module.
        mode
            The mode of the splitter, either "training", "validation", or "test".
        
        Returns
        -------
        An instance splitter transformation.
        """
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
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        )
    
    def create_training_data_loader(
        self,
        data: Dataset,
        module: TACTiS2LightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        """
        Create the training data loader.
        
        Parameters
        ----------
        data
            The training dataset.
        module
            The TACTiS2 lightning module.
        shuffle_buffer_length
            Length of the shuffle buffer.
        
        Returns
        -------
        An iterable over the training dataset.
        """
        if self.num_batches_per_epoch is not None:
            data = Cyclic(data).stream()
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
        module: TACTiS2LightningModule,
        **kwargs,
    ) -> Iterable:
        """
        Create the validation data loader.
        
        Parameters
        ----------
        data
            The validation dataset.
        module
            The TACTiS2 lightning module.
        
        Returns
        -------
        An iterable over the validation dataset.
        """
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
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

    
    def create_pytorch_training_data_loader(
        self,
        data_path: str,
        module: TACTiS2LightningModule,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for training.
        
        Parameters
        ----------
        data_path
            Path to the pickle file containing training data.
        module
            The TACTiS2 lightning module.
        
        Returns
        -------
        A PyTorch DataLoader for training.
        """
        # Import here to avoid circular imports
        from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDataset
        
        data = WindForecastingDataset(
            data_path=data_path,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            time_features=self.time_features,
            sampler=self.train_sampler,  # Pass the GluonTS sampler
            repeat=self.num_batches_per_epoch is not None # will just repeat over same dataset if we only provide one
        )
        
        # Return DataLoader - PyTorch Lightning will add DistributedSampler automatically
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,  # Will be overridden by DistributedSampler in DDP
            # worker_init_fn=self.__class__._worker_init_fn,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            persistent_workers=kwargs.get('persistent_workers', True),
            # drop_last=True,  # Important for DDP to avoid uneven batch sizes
        )
        
    def create_pytorch_validation_data_loader(
        self,
        data_path: str,
        module: TACTiS2LightningModule,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for validation.
        
        Parameters
        ----------
        data_path
            Path to the pickle file containing validation data.
        module
            The TACTiS2 lightning module.
        
        Returns
        -------
        A PyTorch DataLoader for validation.
        """
        # Import here to avoid circular imports
        from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingInferenceDataset
        
        dataset = WindForecastingInferenceDataset(
            data_path=data_path,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            time_features=self.time_features,
            repeat=False
        )
        
        # Return DataLoader - PyTorch Lightning will add DistributedSampler automatically
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle validation data
            # worker_init_fn=self.__class__._worker_init_fn,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            persistent_workers=kwargs.get('persistent_workers', True),
            # drop_last=False,  # Keep all validation samples
        )
    
    def create_predictor(
        self,
        transformation: Transformation,
        module: TACTiS2LightningModule,
        **kwargs
    ) -> PyTorchPredictor:
        """
        Create a predictor from a trained model.
        
        Parameters
        ----------
        transformation
            Transformation to apply to the input data.
        module
            Trained TACTiS2 lightning module.
        
        Returns
        -------
        A PyTorch predictor that can be used for making predictions.
        """
        prediction_splitter = self._create_instance_splitter(module, "test")
        
        return PyTorchPredictor(
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            input_transform=transformation + prediction_splitter,
            forecast_generator=SampleForecastGenerator(),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def create_lightning_module(self) -> TACTiS2LightningModule:
        """
        Create a new TACTiS2 lightning module.
        
        Returns
        -------
        A TACTiS2 lightning module.
        """
        # Gather all configured parameters for the TACTiS2Model into a dictionary
        model_config = {
            # Data dimensions
            # "freq": self.freq,  # DEBUG Temporarily commented out - will be included in future tuned models
            "num_series": self.input_size,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            # TACTiS specific parameters from __init__
            "flow_series_embedding_dim": self.flow_series_embedding_dim,
            "copula_series_embedding_dim": self.copula_series_embedding_dim,
            "flow_input_encoder_layers": self.flow_input_encoder_layers,
            "copula_input_encoder_layers": self.copula_input_encoder_layers,
            "marginal_embedding_dim_per_head": self.marginal_embedding_dim_per_head,
            "marginal_num_heads": self.marginal_num_heads,
            "marginal_num_layers": self.marginal_num_layers,
            "copula_embedding_dim_per_head": self.copula_embedding_dim_per_head,
            "copula_num_heads": self.copula_num_heads,
            "copula_num_layers": self.copula_num_layers,
            "decoder_dsf_num_layers": self.decoder_dsf_num_layers,
            "decoder_dsf_hidden_dim": self.decoder_dsf_hidden_dim,
            "decoder_mlp_num_layers": self.decoder_mlp_num_layers,
            "decoder_mlp_hidden_dim": self.decoder_mlp_hidden_dim,
            "decoder_transformer_num_layers": self.decoder_transformer_num_layers,
            "decoder_transformer_embedding_dim_per_head": self.decoder_transformer_embedding_dim_per_head,
            "decoder_transformer_num_heads": self.decoder_transformer_num_heads,
            "decoder_num_bins": self.decoder_num_bins,
            "bagging_size": self.bagging_size,
            "input_encoding_normalization": self.input_encoding_normalization,
            "loss_normalization": self.loss_normalization,
            "encoder_type": self.encoder_type, # Pass encoder type
            "dropout_rate": self.dropout_rate, # Pass dropout rate
            # "stage1_activation_function": self.stage1_activation_function, # Will be passed directly to LightningModule
            # Attentional Copula specific MLP params (Aligned with AttentionalCopula class)
            "ac_mlp_num_layers": self.ac_mlp_num_layers,
            "ac_mlp_dim": self.ac_mlp_dim,
            # "stage2_activation_function": self.stage2_activation_function, # Will be passed directly to LightningModule
            # GluonTS compatability parameters
            "cardinality": self.cardinality,
            "num_feat_dynamic_real": self.num_feat_dynamic_real,
            "num_feat_static_real": self.num_feat_static_real,
            "num_feat_static_cat": self.num_feat_static_cat,
            "embedding_dimension": self.embedding_dimension,
            "scaling": self.scaling,
            "lags_seq": self.lags_seq,
            "num_parallel_samples": self.num_parallel_samples,
        }
        
        # Calculate absolute steps from fractional values if needed
        # This allows setting warmup/decay as fractions of each training stage
        max_epochs = self.trainer_kwargs.get("max_epochs", 100)  # Default to 100 if not specified
        
        # Calculate epochs per stage
        epochs_stage1 = self.stage2_start_epoch
        epochs_stage2 = max_epochs - self.stage2_start_epoch
        
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
            steps_stage1 = epochs_stage1 * effective_batches_per_epoch if effective_batches_per_epoch else 0
            steps_stage2 = epochs_stage2 * effective_batches_per_epoch if effective_batches_per_epoch else 0
            
        logger.info(f"Training schedule calculation:")
        logger.info(f"  Max epochs: {max_epochs}")
        logger.info(f"  Stage 1: epochs 0-{epochs_stage1} ({epochs_stage1} epochs, {steps_stage1:,} steps)")
        logger.info(f"  Stage 2: epochs {self.stage2_start_epoch}-{max_epochs} ({epochs_stage2} epochs, {steps_stage2:,} steps)")
        logger.info(f"  Effective steps per epoch: {effective_batches_per_epoch:,}")
        
        resolved_warmup_s1 = resolve_steps(self.warmup_steps_s1, steps_stage1, "warmup_steps_s1")
        resolved_warmup_s2 = resolve_steps(self.warmup_steps_s2, steps_stage2, "warmup_steps_s2")
        resolved_decay_s1 = resolve_steps(self.steps_to_decay_s1, steps_stage1, "steps_to_decay_s1")
        resolved_decay_s2 = resolve_steps(self.steps_to_decay_s2, steps_stage2, "steps_to_decay_s2")
        
        return TACTiS2LightningModule(
            model_config=model_config, # Pass config dict
            # Pass stage-specific optimizer params
            lr_stage1=self.lr_stage1,
            eta_min_fraction_s1=self.eta_min_fraction_s1,  # Pass eta_min fractions
            eta_min_fraction_s2=self.eta_min_fraction_s2,
            lr_stage2=self.lr_stage2,
            weight_decay_stage1=self.weight_decay_stage1,
            weight_decay_stage2=self.weight_decay_stage2,
            # Pass gradient clipping values
            gradient_clip_val_stage1=self.gradient_clip_val_stage1,
            gradient_clip_val_stage2=self.gradient_clip_val_stage2,
            # Pass training stage params
            stage=self.initial_stage,
            stage2_start_epoch=self.stage2_start_epoch,
            warmup_steps_s1=resolved_warmup_s1, # Pass resolved warmup steps for stage 1
            warmup_steps_s2=resolved_warmup_s2, # Pass resolved warmup steps for stage 2
            steps_to_decay_s1=resolved_decay_s1, # Pass resolved T_max value for stage 1
            steps_to_decay_s2=resolved_decay_s2, # Pass resolved T_max value for stage 2
            # Pass activation functions explicitly as keyword arguments
            stage1_activation_function=self.stage1_activation_function,
            stage2_activation_function=self.stage2_activation_function,
            # Pass batch size parameters for scheduler step scaling
            batch_size=self.batch_size,
            base_batch_size_for_scheduler_steps=self.base_batch_size_for_scheduler_steps,
            base_limit_train_batches=self.base_limit_train_batches,
            # Pass num_batches_per_epoch for scheduler calculations
            num_batches_per_epoch=self.num_batches_per_epoch,
        )
