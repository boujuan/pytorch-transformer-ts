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
    ExpandDimArray,
    SetFieldIfNotPresent,
    SelectFields
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.transform.field import RenameFields
from gluonts.model.forecast_generator import DistributionForecastGenerator

from .module import TACTiS2Model
from .lightning_module import TACTiS2LightningModule

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
        # TACTiS2 specific arguments
        flow_series_embedding_dim: int = 32,
        copula_series_embedding_dim: int = 32,
        flow_input_encoder_layers: int = 2,
        copula_input_encoder_layers: int = 2,
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        data_normalization: str = "none",
        loss_normalization: str = "series",
        initial_stage: int = 1,
        stage2_start_epoch: int = 10,
        use_lazyframe: bool = False,
        # Common parameters
        context_length: Optional[int] = None,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: Optional[str] = "std",
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        input_size: int = 1,
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 100,
            **trainer_kwargs,
        }
        super().__init__(trainer_kwargs=trainer_kwargs)
        
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        
        self.use_lazyframe = use_lazyframe
        
        # TACTiS2 specific parameters
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
        
        # Common parameters
        self.input_size = input_size
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = cardinality if cardinality and num_feat_static_cat > 0 else [1]
        self.embedding_dimension = embedding_dimension
        self.distr_output = distr_output
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
        
    @staticmethod
    def get_params(trial):
        """
        Get parameters for hyperparameter tuning.
        
        Parameters
        ----------
        trial
            Optuna trial object.
        context_length_choices
            List of possible context lengths.
        
        Returns
        -------
        Dict of parameter values.
        """
        # Example hyperparameters to tune
        params = {
             "context_length_factor": trial.suggest_categorical("context_length_factor", [1, 2, 3, 4]),
            "flow_series_embedding_dim": trial.suggest_int("flow_series_embedding_dim", 16, 64),
            "copula_series_embedding_dim": trial.suggest_int("copula_series_embedding_dim", 16, 64),
            "flow_input_encoder_layers": trial.suggest_int("flow_input_encoder_layers", 1, 3),
            "copula_input_encoder_layers": trial.suggest_int("copula_input_encoder_layers", 1, 3),
            "data_normalization": trial.suggest_categorical("data_normalization", ["none", "series"]),
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
                AsLazyFrame(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape)
                ) if use_lazyframe else AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension 
                    expected_ndim=1 + len(self.distr_output.event_shape),
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
            forecast_generator=DistributionForecastGenerator(self.distr_output),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def create_lightning_module(self) -> TACTiS2LightningModule:
        """
        Create a new TACTiS2 lightning module.
        
        Returns
        -------
        A TACTiS2 lightning module.
        """
        model = TACTiS2Model(
            num_series=self.input_size,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            # TACTiS2 specific parameters
            flow_series_embedding_dim=self.flow_series_embedding_dim,
            copula_series_embedding_dim=self.copula_series_embedding_dim,
            flow_input_encoder_layers=self.flow_input_encoder_layers,
            copula_input_encoder_layers=self.copula_input_encoder_layers,
            bagging_size=self.bagging_size,
            input_encoding_normalization=self.input_encoding_normalization,
            data_normalization=self.data_normalization,
            loss_normalization=self.loss_normalization,
            # Common parameters
            cardinality=self.cardinality,
            num_feat_dynamic_real=self.num_feat_dynamic_real,
            num_feat_static_real=self.num_feat_static_real,
            num_feat_static_cat=self.num_feat_static_cat,
            embedding_dimension=self.embedding_dimension,
            distr_output=self.distr_output,
            scaling=self.scaling,
            lags_seq=self.lags_seq,
            num_parallel_samples=self.num_parallel_samples,
        )
        
        return TACTiS2LightningModule(
            model=model,
            lr=1e-3,  # Learning rate can be made configurable
            weight_decay=1e-8,  # Weight decay can be made configurable
            stage=self.initial_stage,
            stage2_start_epoch=self.stage2_start_epoch,
        )
