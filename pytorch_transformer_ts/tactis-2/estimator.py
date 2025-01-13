from typing import Any, Dict, Iterable, List, Optional

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
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from .lightning_module import TACTiSLightningModule
from .module import TACTiSModel

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "past_time_feat",
    "past_static_feat",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]

class TACTiSEstimator(PyTorchLightningEstimator):
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
        # TACTiS arguments
        num_series: int = 1,
        num_layers_encoder: int = 2,
        num_layers_decoder: int = 2,
        num_parallel_samples: int = 100,
        dropout: float = 0.1,
        dim_feedforward: int = 32,
        num_heads: int = 8,
        positional_encoding: str = "sin_cos",
        normalization: str = "batchnorm",
        activation: str = "gelu",
        # GluonTS arguments
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: Optional[str] = "mean",
        time_features: Optional[List[TimeFeature]] = None,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        **kwargs,
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 1.0,
            "callbacks": [EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")],
        }
        super().__init__(trainer_kwargs=trainer_kwargs, **kwargs)

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality or [1]
        self.embedding_dimension = embedding_dimension
        self.num_parallel_samples = num_parallel_samples
        self.lags_seq = lags_seq
        self.scaling = scaling
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        self.distr_output = distr_output
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.num_series = num_series
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.positional_encoding = positional_encoding
        self.normalization = normalization
        self.activation = activation

        self.model_kwargs = {
            "freq": self.freq,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "num_feat_dynamic_real": self.num_feat_dynamic_real,
            "num_feat_static_cat": self.num_feat_static_cat,
            "num_feat_static_real": self.num_feat_static_real,
            "cardinality": self.cardinality,
            "embedding_dimension": self.embedding_dimension,
            "num_parallel_samples": self.num_parallel_samples,
            "lags_seq": self.lags_seq,
            "scaling": self.scaling,
            "time_features": self.time_features,
            "distr_output": self.distr_output,
            "num_layers_encoder": self.num_layers_encoder,
            "num_layers_decoder": self.num_layers_decoder,
            "num_series": self.num_series,
            "dropout": self.dropout,
            "dim_feedforward": self.dim_feedforward,
            "num_heads": self.num_heads,
            "positional_encoding": self.positional_encoding,
            "normalization": self.normalization,
            "activation": self.activation,
        }

    def create_transformation(self) -> Transformation:
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
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # In the multivariate case, we add an axis with size 1
                    expected_ndim=1,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AsNumpyArray(field=FieldName.FEAT_TIME, expected_ndim=2),
            ]
        )

    def _create_instance_splitter(self, module: TACTiSLightningModule, mode: str):
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
        module: TACTiSLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
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
        module: TACTiSLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: TACTiSLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    def create_lightning_module(self) -> pl.LightningModule:
        model = TACTiSModel(**self.model_kwargs)
        return TACTiSLightningModule(model=model)