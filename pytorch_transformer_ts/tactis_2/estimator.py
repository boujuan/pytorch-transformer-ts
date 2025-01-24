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
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
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

from lightning_module import TACTiSLightningModule
from module import TACTiSModel

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

def create_transformation(
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_cat: int = 1,
        num_feat_static_real: int = 1,
        cardinality: Optional[List[int]] = None,
        add_time_feature: bool = True,
        add_age_feature: bool = True,
        add_observed_values_indicator: bool = False,
        time_features: Optional[List[TimeFeature]] = None,
    ) -> Transformation:
        remove_field_names = []
        if num_feat_static_real <= 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if num_feat_dynamic_real <= 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0] * num_feat_static_cat)]
                if num_feat_static_cat > 0
                else []
            )
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0] * num_feat_static_real)]
                if num_feat_static_real > 0
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
                    # we expect an extra dim for the multivariate case
                    expected_ndim=1,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                # time feature
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features
                    or time_features_from_frequency_str(freq),
                    pred_length=prediction_length,
                ),
                # age feature
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if num_feat_dynamic_real > 0
                        else []
                    ),
                ),
                # Move "target" to the end
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            ]
        )

def create_instance_splitter(
        target_field: str,
        is_train: bool,
        past_length: int,
        future_length: int,
        time_series_fields: List[str],
        instance_sampler: Optional[InstanceSampler] = None,
    ) -> InstanceSplitter:
        return InstanceSplitter(
            target_field=target_field,
            is_train=is_train,
            past_length=past_length,
            future_length=future_length,
            time_series_fields=time_series_fields,
            instance_sampler=instance_sampler,
        )

class TACTiSEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        # TACTiS parameters
        context_length: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        # TACTiS specific parameters
        num_series: int = 1,
        num_layers_encoder: int = 2,
        num_layers_decoder: int = 2,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        scaling: Optional[str] = "mean",
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        **kwargs,
    ) -> None:
        """
        Estimator class for the TACTiS model.

        Args:
            freq: Frequency of the data.
            prediction_length: Length of the prediction horizon.
            context_length: Number of time steps prior to prediction time that the model takes as inputs (default: `prediction_length`).
            num_feat_dynamic_real: Number of dynamic real features (default: 0).
            num_feat_static_cat: Number of static categorical features (default: 0).
            num_feat_static_real: Number of static real features (default: 0).
            cardinality: List of cardinalities, one for each static categorical feature.
            embedding_dimension: Dimension of the embedding space for each static categorical feature (default: `[min(50, (cat+1)//2) for cat in cardinality]`).
            num_series: Number of time series (default: 1).
            num_layers_encoder: Number of layers in the encoder (default: 2).
            num_layers_decoder: Number of layers in the decoder (default: 2).
            num_parallel_samples: Number of parallel samples to generate during inference (default: 100).
            batch_size: Batch size during training (default: 32).
            num_batches_per_epoch: Number of batches per epoch during training (default: 50).
            trainer_kwargs: Additional arguments to be passed to the `Trainer` object.
            scaling: Scaling method to use for the input data (default: "mean").
            distr_output: Distribution output to use (default: `StudentTOutput()`).
            lags_seq: Indices of the lagged target values to use as inputs to the model (default: None, in which case these are automatically determined based on freq).
            time_features: List of time features to use as inputs to the model (default: None, in which case these are automatically determined based on freq).
        """
        default_trainer_kwargs = {"max_epochs": 100}
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length

        # Feature parameters
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = embedding_dimension

        # TACTiS specific parameters
        self.num_series = num_series
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.num_parallel_samples = num_parallel_samples
        self.scaling = scaling
        self.lags_seq = lags_seq
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.distr_output = distr_output
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        )
        self.validation_sampler = ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        return create_transformation(
            self.freq,
            self.context_length,
            self.prediction_length,
            num_feat_dynamic_real=self.num_feat_dynamic_real,
            num_feat_static_cat=self.num_feat_static_cat,
            num_feat_static_real=self.num_feat_static_real,
            cardinality=self.cardinality,
            time_features=self.time_features,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        model = TACTiSModel(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=self.num_feat_dynamic_real,
            num_feat_static_real=self.num_feat_static_real,
            num_feat_static_cat=self.num_feat_static_cat,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            num_series=self.num_series,
            num_layers_encoder=self.num_layers_encoder,
            num_layers_decoder=self.num_layers_decoder,
            num_parallel_samples=self.num_parallel_samples,
            scaling=self.scaling,
            distr_output=self.distr_output,
            lags_seq=self.lags_seq,
        )
        return TACTiSLightningModule(model=model)

    def _create_instance_splitter(self, module: TACTiSLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return create_instance_splitter(
            target_field=FieldName.TARGET,
            is_train=(mode == "training"),
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            instance_sampler=instance_sampler,
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