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
    ExpandDimArray
)
from gluonts.transform.sampler import InstanceSampler

# CHANGE
from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.module import InformerModel

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


class InformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        # Informer arguments
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        d_model: int = 512,
        n_heads: int = 8,
        input_size: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,
        attn: str = "prob",
        factor: int = 5,
        distil: bool = True,
        context_length: Optional[int] = None,
        num_feat_dynamic_real: int = 0, # QUESTION why no feat_dynamic_cat
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        use_lazyframe=False,
        # loss: DistributionLoss = NegativeLogLikelihood(), CHANGE
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
        # self.loss = loss CHANGE
        self.use_lazyframe = use_lazyframe

        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.attn = attn
        self.factor = factor
        self.distil = distil

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
        
        # TODO if idx = 0 is returned from sampler then full entry[ts_field] is returned (from slice [...,:idx])
        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length,
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length,
        )
    
    @staticmethod
    def get_params(trial, context_length_choices):
        """ generate dictionary of tunable parameters compatible with optuna """
        # in paper: batch-size=32, num_decoder_layers=2, num_epochs=8, learning rate starts at 1e-4 and decays by 0.5 every epoch
        # encoder is a "3-layer stack and a 1-layer stack" e.g. num_encoder_layers=3
        # d_model=512, num_heads=8, dim_feedforward=2048, lr=0.0001
        return {
            "context_length": trial.suggest_categorical("context_length", context_length_choices),
            # "max_epochs": trial.suggest_int("max_epochs", 1, 10, 2),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_encoder_layers": trial.suggest_categorical("num_encoder_layers", [2, 3, 4]),
            "num_decoder_layers": trial.suggest_categorical("num_decoder_layers", [1, 2, 3]),
            # "dim_feedforward": trial.suggest_categorical("dim_feedforward", [512, 1024, 2048]),
            "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
            "n_heads": trial.suggest_categorical("n_heads", [4, 6, 8])
            # "num_batches_per_epoch":trial.suggest_int("num_batches_per_epoch", 100, 200, 100),   
        }
    
     
    def create_transformation(self, use_lazyframe=None) -> Transformation:
        if use_lazyframe is None and hasattr(self, "use_lazyframe"):
            use_lazyframe = self.use_lazyframe
        else:
            use_lazyframe = False
        # this is called in src/gluonts/torch/model/estimator.py
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
                    # dtype=pl.Float32
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

    
    def _create_instance_splitter(self, module: InformerLightningModule, mode: str):
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
        module: InformerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream() # will just repeat over same dataset if we only provide one
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
        module: InformerLightningModule,
        **kwargs,
    ) -> Iterable:
        # data = Cyclic(data).stream() # CHANGE
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            # num_batches_per_epoch=self.num_batches_per_epoch,
        )

    
    def create_predictor(
        self,
        transformation: Transformation,
        module: InformerLightningModule,
        **kwargs # CHANGE
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

    def create_lightning_module(self) -> InformerLightningModule:
        model_params = dict(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=1 # 1 is for age 
            + self.num_feat_dynamic_real
            + len(self.time_features),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            # Informer arguments
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            activation=self.activation,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward,
            attn=self.attn,
            factor=self.factor,
            distil=self.distil,
            # univariate input
            input_size=self.input_size,
            distr_output=self.distr_output,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )

        # return InformerLightningModule(model=model, loss=self.loss) CHANGE
        return InformerLightningModule(model=model_params)
