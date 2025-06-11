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

from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule

# Set up logging
logger = logging.getLogger(__name__)

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


class AutoformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        # Autoformer arguments
        n_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        input_size: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,
        factor: int = 1,
        moving_avg: int = 25,
        context_length: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        use_lazyframe: bool = False,
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
        warmup_steps: int = 1000, # Warmup steps
        # --- Scheduler specific arguments ---
        eta_min_fraction: float = 0.01,  # Fraction of initial LR for cosine decay eta_min
        steps_to_decay: Optional[int] = None,  # Optional manual T_max value for CosineAnnealingLR
        use_pytorch_dataloader: bool = False,
        **kwargs,
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
        # self.loss = loss
        self.use_lazyframe = use_lazyframe

        self.input_size = input_size
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.factor = factor
        self.moving_avg = moving_avg

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
            logger.warning(f"AutoformerEstimator received unused kwargs: {kwargs}")
    
    @staticmethod
    def get_params(trial, tuning_phase=None, dynamic_kwargs=None):
        """ generate dictionary of tunable parameters compatible with optuna"""
        # in paper: 2 encoder layers and 1 decoder layer. batch_size=32, init_lr=1e-4, early stopping with 10 epchs, dmodel=512, d_ff=2048
        if dynamic_kwargs is None:
            dynamic_kwargs = {}
            
        return {
            # --- Input Params ---
            "context_length_factor": trial.suggest_categorical("context_length_factor", dynamic_kwargs.get("context_length_factor", [2, 3, 4])),
            # "max_epochs": trial.suggest_int("max_epochs", 1, 10, 2),
            "batch_size": trial.suggest_categorical("batch_size", dynamic_kwargs.get("batch_size", [64, 128, 256, 512, 1024])),
            
            # --- Architecture Params ---
            "num_encoder_layers": trial.suggest_categorical("num_encoder_layers", dynamic_kwargs.get("num_encoder_layers", [2, 3, 4])),
            "num_decoder_layers": trial.suggest_categorical("num_decoder_layers", dynamic_kwargs.get("num_decoder_layers", [1, 2, 3])),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", dynamic_kwargs.get("dim_feedforward", [512, 1028, 2048])),
            # "d_model": trial.suggest_categorical("d_model", dynamic_kwargs.get("d_model", [128, 256, 512])),
            "n_heads": trial.suggest_categorical("n_heads", dynamic_kwargs.get("n_heads", [4, 6, 8])),
            "factor": trial.suggest_categorical("factor", dynamic_kwargs.get("factor", [1, 3, 5])),
            "moving_avg": trial.suggest_categorical("moving_avg", dynamic_kwargs.get("moving_avg", [9, 15, 25])), # TODO this should vary with context length?
            # "num_batches_per_epoch":trial.suggest_int("num_batches_per_epoch", 100, 200, 100),   
            
            # activation: str = "gelu",
            
            # --- Optimizer Params ---
            "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=False),
            "weight_decay": trial.suggest_categorical("weight_decay", dynamic_kwargs.get("weight_decay", [0.0, 1e-8, 1e-6, 1e-4])),
        
            # --- Dropout & Clipping ---  
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
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

    def _create_instance_splitter(self, module: AutoformerLightningModule, mode: str):
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
        module: AutoformerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
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
        module: AutoformerLightningModule,
        **kwargs,
    ) -> Iterable:
        # data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            # num_batches_per_epoch=self.num_batches_per_epoch, #CHANGE
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
        
        from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDatamodule
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
            pin_memory=kwargs.get('pin_memory', True), 
            persistent_workers=kwargs.get('persistent_workers', True),
            )
    # def create_pytorch_training_data_loader(
    #     self,
    #     data_path: str,
    #     module: AutoformerLightningModule,
    #     **kwargs,
    #     ) -> torch.utils.data.DataLoader:
    #     """
    #     Create a PyTorch DataLoader for training.
        
    #     Parameters
    #     ----------
    #     data_path
    #         Path to the pickle file containing training data.
    #     module
    #         The Autoformer lightning module.
        
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
    #     module: AutoformerLightningModule,
    #     **kwargs,
    # ) -> torch.utils.data.DataLoader:
    #     """
    #     Create a PyTorch DataLoader for validation.
        
    #     Parameters
    #     ----------
    #     data_path
    #         Path to the pickle file containing validation data.
    #     module
    #         The Autoformer lightning module.
        
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
        module: AutoformerLightningModule,
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

    def create_lightning_module(self) -> AutoformerLightningModule:
        model_params = dict(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=(0 if self.use_pytorch_dataloader else 1) # 1 is for age
            + self.num_feat_dynamic_real
            + len(self.time_features),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            # autoformer arguments
            n_heads=self.n_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            activation=self.activation,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward,
            factor=self.factor,
            moving_avg=self.moving_avg,
            # univariate input
            input_size=self.input_size,
            distr_output=self.distr_output,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )

        # return AutoformerLightningModule(model=model, loss=self.loss)
        return AutoformerLightningModule(
            model_config=model_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eta_min_fraction=self.eta_min_fraction,  # Pass eta_min fraction
            # Pass gradient clipping val
            gradient_clip_val=self.gradient_clip_val,
            # Pass training stage params
            warmup_steps=self.warmup_steps, # Pass warmup steps
            steps_to_decay=self.steps_to_decay, # Pass manual T_max value
            # Pass batch size parameters for scheduler step scaling
            batch_size=self.batch_size,
            base_batch_size_for_scheduler_steps=self.base_batch_size_for_scheduler_steps,
            base_limit_train_batches=self.base_limit_train_batches,
            # Pass num_batches_per_epoch for scheduler calculations
            num_batches_per_epoch=self.num_batches_per_epoch
        )
