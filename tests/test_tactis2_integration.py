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


class TestTACTiS2Integration:
    """Integration tests for TACTiS-2 model."""
    
    def sample_dataset(self):
        """Create a sample multivariate time series dataset."""
        np.random.seed(42)
        
        # Create synthetic multivariate time series data
        num_series = 3
        num_timesteps = 500
        num_features = 4  # Multiple features for multivariate testing
        
        # Generate time series data
        data = []
        start_date = pd.Timestamp("2024-01-01")
        
        for i in range(num_series):
            # Create correlated multivariate series with some pattern
            base_signal = np.sin(np.linspace(0, 4*np.pi, num_timesteps))
            noise = np.random.randn(num_timesteps, num_features) * 0.1
            
            # Create different patterns for each feature
            target_values = np.zeros((num_timesteps, num_features))
            for j in range(num_features):
                phase_shift = j * np.pi / 4
                amplitude = 1.0 + j * 0.2
                target_values[:, j] = amplitude * np.sin(np.linspace(phase_shift, phase_shift + 4*np.pi, num_timesteps))
                target_values[:, j] += noise[:, j]
            
            data.append({
                FieldName.TARGET: target_values,
                FieldName.START: start_date,
                FieldName.ITEM_ID: f"turbine_{i}",
                FieldName.FEAT_STATIC_CAT: [i],  # Static categorical feature
                FieldName.FEAT_STATIC_REAL: [float(i) * 0.5],  # Static real feature
            })
        
        # For multivariate, we need to specify one_dim_target=False
        return ListDataset(data, freq="10s", one_dim_target=False)
    
    def model_config(self):
        """Create a basic configuration for TACTiS-2 model."""
        return {
            "freq": "10s",
            "prediction_length": 21,  # 210 seconds at 10S frequency
            "context_length": 68,  # Context for the model
            
            # Model architecture parameters
            "flow_series_embedding_dim": 5,
            "copula_series_embedding_dim": 16,  # Reduced for testing
            "flow_input_encoder_layers": 2,  # Reduced for testing
            "copula_input_encoder_layers": 1,
            "marginal_embedding_dim_per_head": 4,  # Reduced for testing
            "marginal_num_heads": 2,  # Reduced for testing
            "marginal_num_layers": 2,  # Reduced for testing
            "copula_embedding_dim_per_head": 4,  # Reduced for testing
            "copula_num_heads": 2,  # Reduced for testing
            "copula_num_layers": 1,  # Reduced for testing
            "decoder_dsf_num_layers": 2,
            "decoder_dsf_hidden_dim": 64,  # Reduced for testing
            "decoder_mlp_num_layers": 2,
            "decoder_mlp_hidden_dim": 8,  # Reduced for testing
            "decoder_transformer_num_layers": 2,
            "decoder_transformer_embedding_dim_per_head": 8,  # Reduced for testing
            "decoder_transformer_num_heads": 2,  # Reduced for testing
            
            # Attentional Copula parameters
            "ac_mlp_num_layers": 2,
            "ac_mlp_dim": 16,
            "ac_activation_function": "ReLU",
            
            # Training parameters
            "lr_stage1": 1e-3,
            "lr_stage2": 5e-4,
            "weight_decay_stage1": 0.0,
            "weight_decay_stage2": 0.0,
            "batch_size": 8,  # Small batch size for testing
            "num_batches_per_epoch": 10,  # For testing
            "trainer_kwargs": {
                "max_epochs": 3,  # Short training for testing
                "accelerator": "cpu",  # Use CPU for testing
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            },
            
            # Two-stage training parameters
            "stage2_start_epoch": 2,  # Start stage 2 early for testing
            "skip_copula": False,  # Enable full two-stage training
            
            # Scheduler parameters
            "warmup_fraction_s1": 0.1,
            "warmup_fraction_s2": 0.1,
            "eta_min_fraction_s1": 0.01,
            "eta_min_fraction_s2": 0.01,
        }
    
    def test_model_initialization(self, model_config):
        """Test that TACTiS-2 model can be initialized correctly."""
        logger.info("Testing TACTiS-2 model initialization...")
        
        # Create estimator
        estimator = TACTiS2Estimator(**model_config)
        
        # Verify estimator attributes
        assert estimator.prediction_length == model_config["prediction_length"]
        assert estimator.context_length == model_config["context_length"]
        assert estimator.freq == model_config["freq"]
        
        # Verify model configuration propagation
        assert estimator.flow_series_embedding_dim == model_config["flow_series_embedding_dim"]
        assert estimator.copula_series_embedding_dim == model_config["copula_series_embedding_dim"]
        
        logger.info("✓ Model initialization successful")
    
    def test_data_transformation(self, model_config, sample_dataset):
        """Test that data transformation pipeline works correctly."""
        logger.info("Testing data transformation pipeline...")
        
        # Create estimator
        estimator = TACTiS2Estimator(**model_config)
        
        # Create transformation
        transformation = estimator.create_transformation()
        
        # Apply transformation to dataset
        transformed_data = list(transformation.apply(sample_dataset, is_train=True))
        
        # Verify transformed data structure
        assert len(transformed_data) > 0
        first_item = transformed_data[0]
        
        # Check required fields are present
        assert FieldName.FEAT_STATIC_CAT in first_item
        assert FieldName.FEAT_STATIC_REAL in first_item
        assert "past_target" in first_item or FieldName.TARGET in first_item
        
        logger.info(f"✓ Transformation successful, produced {len(transformed_data)} items")
    
    def test_two_stage_training(self, model_config, sample_dataset):
        """Test the two-stage training process."""
        logger.info("Testing two-stage training process...")
        
        # Create estimator with two-stage configuration
        estimator = TACTiS2Estimator(**model_config)
        
        # Create a temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update trainer kwargs with checkpoint callback
            estimator.trainer_kwargs["default_root_dir"] = tmpdir
            estimator.trainer_kwargs["callbacks"] = [
                ModelCheckpoint(
                    dirpath=os.path.join(tmpdir, "checkpoints"),
                    monitor="train_loss",
                    save_top_k=1,
                    filename="tactis2-{epoch:02d}-{train_loss:.4f}",
                )
            ]
            
            # Train the model
            logger.info(f"Starting training with stage2_start_epoch={model_config['stage2_start_epoch']}")
            predictor = estimator.train(sample_dataset)
            
            # Verify predictor is created
            assert isinstance(predictor, PyTorchPredictor)
            
            # Check that the model has both flow and copula components
            model_state = predictor.prediction_net.state_dict()
            
            # Check for flow (marginal) components
            flow_keys = [k for k in model_state.keys() if "marginal" in k or "flow" in k or "dsf" in k]
            assert len(flow_keys) > 0, "Flow/marginal components should be present"
            
            # Check for copula components (if not skipped)
            if not model_config.get("skip_copula", False):
                copula_keys = [k for k in model_state.keys() if "copula" in k or "attention" in k]
                assert len(copula_keys) > 0, "Copula components should be present"
            
            logger.info(f"✓ Two-stage training successful")
            logger.info(f"  - Found {len(flow_keys)} flow/marginal parameters")
            if not model_config.get("skip_copula", False):
                logger.info(f"  - Found {len([k for k in model_state.keys() if 'copula' in k])} copula parameters")
    
    def test_inference_and_prediction(self, model_config, sample_dataset):
        """Test model inference and prediction generation."""
        logger.info("Testing inference and prediction...")
        
        # Train a simple model
        estimator = TACTiS2Estimator(**model_config)
        predictor = estimator.train(sample_dataset)
        
        # Generate predictions
        forecasts = list(predictor.predict(sample_dataset))
        
        # Verify forecast structure
        assert len(forecasts) == len(list(sample_dataset))
        
        for forecast in forecasts:
            assert isinstance(forecast, SampleForecast)
            
            # Check forecast dimensions
            # Shape should be (num_samples, prediction_length, num_features)
            assert forecast.samples.shape[1] == model_config["prediction_length"]
            
            # Check that samples are generated (probabilistic output)
            assert forecast.samples.shape[0] > 1  # Multiple samples for uncertainty
            
            # Verify forecast has proper time index
            assert forecast.start_date is not None
            
            logger.info(f"  - Forecast shape: {forecast.samples.shape}")
            logger.info(f"  - Forecast start: {forecast.start_date}")
            
        logger.info("✓ Inference and prediction successful")
    
    def test_denormalization_requirement(self, model_config, sample_dataset):
        """Test that model handles denormalized data correctly."""
        logger.info("Testing denormalization handling...")
        
        # Create estimator
        estimator = TACTiS2Estimator(**model_config)
        
        # Note: TACTiS-2 uses internal batch-wise StdScaler
        # The model should handle normalization internally
        
        # Train the model
        predictor = estimator.train(sample_dataset)
        
        # Create test data with different scales
        test_data = []
        for item in sample_dataset:
            # Scale the data differently
            scaled_item = item.copy()
            scaled_item[FieldName.TARGET] = item[FieldName.TARGET] * 100.0  # Scale up
            test_data.append(scaled_item)
        
        test_dataset = ListDataset(test_data, freq="10s", one_dim_target=False)
        
        # Generate predictions on scaled data
        forecasts = list(predictor.predict(test_dataset))
        
        # Verify predictions are generated (model should handle scaling internally)
        assert len(forecasts) == len(test_data)
        
        for forecast in forecasts:
            # Check that predictions are in reasonable range
            # The model should internally normalize/denormalize
            assert not np.isnan(forecast.samples).any()
            assert not np.isinf(forecast.samples).any()
        
        logger.info("✓ Denormalization handling successful")
    
    def test_memory_optimization_features(self, model_config):
        """Test memory optimization features like gradient checkpointing."""
        logger.info("Testing memory optimization features...")
        
        # Enable memory optimization features
        optimized_config = model_config.copy()
        optimized_config.update({
            "gradient_checkpointing": True,
            "trainer_kwargs": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "precision": "16-mixed" if torch.cuda.is_available() else 32,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "gradient_clip_val": 1.0,
            }
        })
        
        # Create estimator with optimization
        estimator = TACTiS2Estimator(**optimized_config)
        
        # Verify configuration
        if "gradient_checkpointing" in optimized_config:
            assert estimator.gradient_checkpointing == True
        
        logger.info("✓ Memory optimization features configured successfully")
    
    def test_stage_transition(self, model_config, sample_dataset):
        """Test the transition from stage 1 to stage 2 during training."""
        logger.info("Testing stage transition during training...")
        
        # Configure for stage transition testing
        transition_config = model_config.copy()
        transition_config["stage2_start_epoch"] = 1  # Transition after first epoch
        transition_config["trainer_kwargs"]["max_epochs"] = 2  # Train for 2 epochs
        
        # Create estimator
        estimator = TACTiS2Estimator(**transition_config)
        
        # Create a custom callback to monitor stage transitions
        class StageMonitor(pl.Callback):
            def __init__(self):
                self.stages_seen = []
            
            def on_train_epoch_start(self, trainer, pl_module):
                current_stage = pl_module.stage
                self.stages_seen.append((trainer.current_epoch, current_stage))
                logger.info(f"  Epoch {trainer.current_epoch}: Stage {current_stage}")
        
        monitor = StageMonitor()
        estimator.trainer_kwargs["callbacks"] = [monitor]
        
        # Train the model
        predictor = estimator.train(sample_dataset)
        
        # Verify stage transition occurred
        if len(monitor.stages_seen) >= 2:
            # Check that we started in stage 1
            assert monitor.stages_seen[0][1] == 1, "Should start in stage 1"
            
            # Check that we transitioned to stage 2 (if configured)
            if not transition_config.get("skip_copula", False):
                if len(monitor.stages_seen) > transition_config["stage2_start_epoch"]:
                    assert monitor.stages_seen[transition_config["stage2_start_epoch"]][1] == 2, \
                        f"Should transition to stage 2 at epoch {transition_config['stage2_start_epoch']}"
        
        logger.info("✓ Stage transition test successful")


def test_full_integration():
    """Run a complete integration test of the TACTiS-2 model."""
    test_suite = TestTACTiS2Integration()
    
    # Create fixtures
    dataset = test_suite.sample_dataset()
    config = test_suite.model_config()
    
    # Run all tests
    print("\n" + "="*60)
    print("TACTiS-2 Integration Test Suite")
    print("="*60 + "\n")
    
    try:
        test_suite.test_model_initialization(config)
        test_suite.test_data_transformation(config, dataset)
        test_suite.test_two_stage_training(config, dataset)
        test_suite.test_inference_and_prediction(config, dataset)
        test_suite.test_denormalization_requirement(config, dataset)
        test_suite.test_memory_optimization_features(config)
        test_suite.test_stage_transition(config, dataset)
        
        print("\n" + "="*60)
        print("✓ All TACTiS-2 integration tests passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_full_integration()