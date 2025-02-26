from typing import Any, Dict, Iterable, List, Optional

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
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
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

from .lightning_module import TACTiS2LightningModule
from .module import TACTiS2Model

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
    Estimator for TACTiS2 models.
    
    This class is responsible for creating and configuring the components needed to train
    and use a TACTiS2 model with GluonTS.
    """
    
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_feat_dynamic_real: int = 1,
        num_feat_static_cat: int = 1,
        num_feat_static_real: int = 1,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        # TACTiS2 specific parameters
        flow_series_embedding_dim: int = 32,
        copula_series_embedding_dim: int = 32,
        flow_input_encoder_layers: int = 1,
        copula_input_encoder_layers: int = 1,
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        data_normalization: str = "none",
        loss_normalization: str = "series",
        # Training stage parameters
        initial_stage: int = 1,
        stage2_start_epoch: int = 10,
        # Other parameters for the model
        distr_output: Optional[DistributionOutput] = None,
        loss: Optional[DistributionLoss] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        input_size: int = 1,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
    ) -> None:
        context_length = context_length or 2 * prediction_length
        
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality if cardinality is not None else [1]
        self.embedding_dimension = embedding_dimension
        
        self.flow_series_embedding_dim = flow_series_embedding_dim
        self.copula_series_embedding_dim = copula_series_embedding_dim
        self.flow_input_encoder_layers = flow_input_encoder_layers
        self.copula_input_encoder_layers = copula_input_encoder_layers
        self.bagging_size = bagging_size
        self.input_encoding_normalization = input_encoding_normalization
        self.data_normalization = data_normalization
        self.loss_normalization = loss_normalization
        
        self.initial_stage = initial_stage
        self.stage2_start_epoch = stage2_start_epoch
        
        self.distr_output = distr_output or StudentTOutput()
        self.loss = loss or NegativeLogLikelihood()
        self.scaling = scaling
        self.num_parallel_samples = num_parallel_samples
        
        self.input_size = input_size
        self.lags_seq = lags_seq
        
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )
        
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        
        trainer_kwargs = trainer_kwargs or {}
        self.trainer_kwargs = trainer_kwargs
        
        super().__init__(trainer_kwargs=trainer_kwargs)

    def create_transformation(self) -> Transformation:
        """
        Create transformation for the dataset.
        
        Returns
        -------
        Transformation
            Transformation to apply to the dataset
        """
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        
        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_CAT,
                        value=[0],
                    )
                ]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL,
                        value=[0.0],
                    )
                ]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=np.long,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 if self.input_size == 1 else 2,
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
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
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
        Create instance splitter based on mode.
        
        Parameters
        ----------
        module: TACTiS2LightningModule
            The Lightning module
        mode: str
            The mode to use (training, validation, or test)
            
        Returns
        -------
        Transformation
            The instance splitter transformation
        """
        assert mode in ["training", "validation", "test"]
        
        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_future=self.prediction_length,
            ),
            "validation": ValidationSplitSampler(
                min_future=self.prediction_length,
            ),
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
                FieldName.FEAT_DYNAMIC_REAL,
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
        Create a data loader for training data.
        
        Parameters
        ----------
        data: Dataset
            The training dataset
        module: TACTiS2LightningModule
            The Lightning module
        shuffle_buffer_length: Optional[int]
            Length of the shuffle buffer
            
        Returns
        -------
        Iterable
            Data loader for training data
        """
        data = Cyclic(data).stream()
        transformation = self._create_instance_splitter(module, "training")
        transformed_data = transformation.apply(data, is_train=True)
        return as_stacked_batches(
            iter(transformed_data),
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
        Create a data loader for validation data.
        
        Parameters
        ----------
        data: Dataset
            The validation dataset
        module: TACTiS2LightningModule
            The Lightning module
            
        Returns
        -------
        Iterable
            Data loader for validation data
        """
        transformation = self._create_instance_splitter(module, "validation")
        transformed_data = transformation.apply(data, is_train=True)
        return as_stacked_batches(
            iter(transformed_data),
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: TACTiS2LightningModule,
        **kwargs
    ) -> PyTorchPredictor:
        """
        Create a predictor for the model.
        
        Parameters
        ----------
        transformation: Transformation
            The preprocessing transformation
        module: TACTiS2LightningModule
            The Lightning module
            
        Returns
        -------
        PyTorchPredictor
            Predictor for the model
        """
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

    def create_lightning_module(self) -> TACTiS2LightningModule:
        """
        Create the TACTiS2 Lightning module.
        
        Returns
        -------
        TACTiS2LightningModule
            The Lightning module for TACTiS2
        """
        model_params = dict(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=1 + len(self.time_features) + self.num_feat_dynamic_real,
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            # TACTiS specific parameters
            flow_series_embedding_dim=self.flow_series_embedding_dim,
            copula_series_embedding_dim=self.copula_series_embedding_dim,
            flow_input_encoder_layers=self.flow_input_encoder_layers,
            copula_input_encoder_layers=self.copula_input_encoder_layers,
            bagging_size=self.bagging_size,
            input_encoding_normalization=self.input_encoding_normalization,
            data_normalization=self.data_normalization,
            loss_normalization=self.loss_normalization,
            # Other parameters
            distr_output=self.distr_output,
            input_size=self.input_size,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )
        
        model = TACTiS2Model(**model_params)
        
        return TACTiS2LightningModule(
            model=model,
            loss=self.loss,
            stage=self.initial_stage,
            stage2_start_epoch=self.stage2_start_epoch,
        )
