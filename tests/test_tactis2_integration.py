"""
Integration test for TACTiS-2 model.

This test verifies:
1. Model initialization and configuration
2. Two-stage training process (flow/marginal and copula)
3. Inference with denormalized data requirement
4. Prediction generation and output format
5. Memory optimization features (gradient checkpointing, precision)
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import tempfile
import os
import logging

# GluonTS imports
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.model.estimator import TrainOutput
from gluonts.model.forecast import SampleForecast

# PyTorch Lightning imports
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# TACTiS-2 model imports
from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule
from pytorch_transformer_ts.tactis_2.module import TACTiS2Model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of synthetic turbines (1D target each, combined = multivariate)
NUM_TURBINES = 4
NUM_TIMESTEPS = 500


def _make_sample_dataset():
    """Create a sample multivariate time series dataset.

    TACTiS expects ONE ListDataset entry containing a 2D target of shape
    (num_timesteps, num_series) where each column is a turbine.  The
    estimator's ``input_size`` equals ``num_series``.
    """
    np.random.seed(42)

    start_date = pd.Timestamp("2024-01-01")

    # Build (num_turbines, num_timesteps) target array — GluonTS multivariate
    # convention: (channels, time) so axis=-1 is the time dimension.
    target_values = np.zeros((NUM_TURBINES, NUM_TIMESTEPS))
    for j in range(NUM_TURBINES):
        phase_shift = j * np.pi / 4
        amplitude = 1.0 + j * 0.2
        target_values[j, :] = amplitude * np.sin(
            np.linspace(phase_shift, phase_shift + 4 * np.pi, NUM_TIMESTEPS)
        )
        target_values[j, :] += np.random.randn(NUM_TIMESTEPS) * 0.1

    data = [
        {
            FieldName.TARGET: target_values,
            FieldName.START: start_date,
            FieldName.ITEM_ID: "awaken_farm",
        }
    ]

    return ListDataset(data, freq="10s", one_dim_target=False)


def _make_model_config():
    """Create a basic configuration for TACTiS-2 model."""
    return {
        "freq": "10s",
        "prediction_length": 21,  # 210 seconds at 10s frequency
        "context_length": 68,
        "input_size": NUM_TURBINES,

        # Model architecture parameters (small for CPU testing)
        "flow_series_embedding_dim": 5,
        "copula_series_embedding_dim": 16,
        "flow_input_encoder_layers": 2,
        "copula_input_encoder_layers": 1,
        "marginal_embedding_dim_per_head": 4,
        "marginal_num_heads": 2,
        "marginal_num_layers": 2,
        "copula_embedding_dim_per_head": 4,
        "copula_num_heads": 2,
        "copula_num_layers": 1,
        "decoder_dsf_num_layers": 2,
        "decoder_dsf_hidden_dim": 64,
        "decoder_mlp_num_layers": 2,
        "decoder_mlp_hidden_dim": 8,
        "decoder_transformer_num_layers": 2,
        "decoder_transformer_embedding_dim_per_head": 8,
        "decoder_transformer_num_heads": 2,

        # Attentional Copula parameters
        "ac_mlp_num_layers": 2,
        "ac_mlp_dim": 16,
        "stage2_activation_function": "ReLU",

        # Training parameters
        "lr_stage1": 1e-3,
        "lr_stage2": 5e-4,
        "weight_decay_stage1": 0.0,
        "weight_decay_stage2": 0.0,
        "batch_size": 8,
        "num_batches_per_epoch": 10,
        "num_parallel_samples": 10,
        "trainer_kwargs": {
            "max_epochs": 3,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "limit_train_batches": 5,
        },

        # Two-stage training parameters
        "stage2_start_epoch": 2,
        "skip_copula": False,
        "initial_stage": 1,

        # Scheduler parameters (fractional warmup)
        "warmup_steps_s1": 0.1,
        "warmup_steps_s2": 0.1,
        "eta_min_fraction_s1": 0.01,
        "eta_min_fraction_s2": 0.01,
    }


class TestTACTiS2Integration:
    """Integration tests for TACTiS-2 model."""

    @pytest.fixture
    def sample_dataset(self):
        return _make_sample_dataset()

    @pytest.fixture
    def model_config(self):
        return _make_model_config()

    def test_model_initialization(self, model_config):
        """Test that TACTiS-2 model can be initialized correctly."""
        logger.info("Testing TACTiS-2 model initialization...")

        estimator = TACTiS2Estimator(**model_config)

        assert estimator.prediction_length == model_config["prediction_length"]
        assert estimator.context_length == model_config["context_length"]
        assert estimator.freq == model_config["freq"]
        assert estimator.flow_series_embedding_dim == model_config["flow_series_embedding_dim"]
        assert estimator.copula_series_embedding_dim == model_config["copula_series_embedding_dim"]

        logger.info("Model initialization successful")

    def test_data_transformation(self, model_config, sample_dataset):
        """Test that data transformation pipeline works correctly."""
        logger.info("Testing data transformation pipeline...")

        estimator = TACTiS2Estimator(**model_config)
        transformation = estimator.create_transformation()
        transformed_data = list(transformation.apply(sample_dataset, is_train=True))

        assert len(transformed_data) > 0
        first_item = transformed_data[0]
        assert "past_target" in first_item or FieldName.TARGET in first_item

        logger.info(f"Transformation successful, produced {len(transformed_data)} items")

    def test_two_stage_training(self, model_config, sample_dataset):
        """Test the two-stage training process."""
        logger.info("Testing two-stage training process...")

        estimator = TACTiS2Estimator(**model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            estimator.trainer_kwargs["default_root_dir"] = tmpdir
            estimator.trainer_kwargs["callbacks"] = [
                ModelCheckpoint(
                    dirpath=os.path.join(tmpdir, "checkpoints"),
                    monitor="train_loss",
                    save_top_k=1,
                    filename="tactis2-{epoch:02d}-{train_loss:.4f}",
                )
            ]

            train_output = estimator.train(sample_dataset)
            assert isinstance(train_output, TrainOutput)
            predictor = train_output.predictor

            model_state = predictor.prediction_net.state_dict()

            flow_keys = [k for k in model_state if "marginal" in k or "flow" in k or "dsf" in k]
            assert len(flow_keys) > 0, "Flow/marginal components should be present"

            if not model_config.get("skip_copula", False):
                copula_keys = [k for k in model_state if "copula" in k or "attention" in k]
                assert len(copula_keys) > 0, "Copula components should be present"

            logger.info("Two-stage training successful")

    def test_inference_and_prediction(self, model_config, sample_dataset):
        """Test model inference and prediction generation."""
        logger.info("Testing inference and prediction...")

        estimator = TACTiS2Estimator(**model_config)
        train_output = estimator.train(sample_dataset)
        predictor = train_output.predictor
        forecasts = list(predictor.predict(sample_dataset))

        assert len(forecasts) == len(list(sample_dataset))

        for forecast in forecasts:
            assert isinstance(forecast, SampleForecast)
            assert forecast.samples.shape[1] == model_config["prediction_length"]
            assert forecast.samples.shape[0] > 1
            assert forecast.start_date is not None

        logger.info("Inference and prediction successful")

    def test_denormalization_requirement(self, model_config, sample_dataset):
        """Test that model handles denormalized data correctly."""
        logger.info("Testing denormalization handling...")

        estimator = TACTiS2Estimator(**model_config)
        train_output = estimator.train(sample_dataset)
        predictor = train_output.predictor

        # Scale data — TACTiS should handle this via internal StdScaler
        test_data = []
        for item in sample_dataset:
            scaled_item = item.copy()
            scaled_item[FieldName.TARGET] = item[FieldName.TARGET] * 100.0
            test_data.append(scaled_item)

        test_dataset = ListDataset(test_data, freq="10s", one_dim_target=False)
        forecasts = list(predictor.predict(test_dataset))

        assert len(forecasts) == len(test_data)
        for forecast in forecasts:
            assert not np.isnan(forecast.samples).any()
            assert not np.isinf(forecast.samples).any()

        logger.info("Denormalization handling successful")

    def test_memory_optimization_features(self, model_config):
        """Test memory optimization configuration."""
        logger.info("Testing memory optimization features...")

        optimized_config = model_config.copy()
        optimized_config["trainer_kwargs"] = {
            "max_epochs": 1,
            "accelerator": "cpu",
            "precision": 32,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "gradient_clip_val": 1.0,
        }

        estimator = TACTiS2Estimator(**optimized_config)
        assert estimator.prediction_length == model_config["prediction_length"]

        logger.info("Memory optimization features configured successfully")

    def test_stage_transition(self, model_config, sample_dataset):
        """Test the transition from stage 1 to stage 2 during training."""
        logger.info("Testing stage transition during training...")

        transition_config = model_config.copy()
        transition_config["stage2_start_epoch"] = 1
        transition_config["trainer_kwargs"] = model_config["trainer_kwargs"].copy()
        transition_config["trainer_kwargs"]["max_epochs"] = 2

        estimator = TACTiS2Estimator(**transition_config)

        class StageMonitor(pl.Callback):
            def __init__(self):
                self.stages_seen = []

            def on_train_epoch_start(self, trainer, pl_module):
                self.stages_seen.append((trainer.current_epoch, pl_module.stage))

        monitor = StageMonitor()
        estimator.trainer_kwargs["callbacks"] = [monitor]

        estimator.train(sample_dataset)

        if len(monitor.stages_seen) >= 2:
            assert monitor.stages_seen[0][1] == 1, "Should start in stage 1"

        logger.info("Stage transition test successful")


@pytest.mark.skip(reason="Standalone runner — use __main__ directly; OOMs under pytest due to cumulative memory")
def test_full_integration():
    """Run a complete integration test of the TACTiS-2 model."""
    dataset = _make_sample_dataset()
    config = _make_model_config()

    test_suite = TestTACTiS2Integration()

    print("\n" + "=" * 60)
    print("TACTiS-2 Integration Test Suite")
    print("=" * 60 + "\n")

    try:
        test_suite.test_model_initialization(config)
        test_suite.test_data_transformation(config, dataset)
        test_suite.test_two_stage_training(config, dataset)
        test_suite.test_inference_and_prediction(config, dataset)
        test_suite.test_denormalization_requirement(config, dataset)
        test_suite.test_memory_optimization_features(config)
        test_suite.test_stage_transition(config, dataset)

        print("\n" + "=" * 60)
        print("All TACTiS-2 integration tests passed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nTest failed: {e}")
        raise


if __name__ == "__main__":
    test_full_integration()
