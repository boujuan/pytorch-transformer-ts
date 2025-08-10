#!/usr/bin/env python3
"""
Stage 2 Training Monitor for TACTiS-2

This module provides comprehensive monitoring for Stage 2 training, including:
- Copula loss component tracking
- Multivariate validation metrics (Energy Score, Sample Diversity)
- Stage-aware optimization metrics for Optuna
- Gradient flow monitoring for copula parameters

The Stage2Monitor callback integrates with PyTorch Lightning training
to provide proper visibility into copula learning dynamics.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Any
import logging
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)


class Stage2Monitor(Callback):
    """
    Comprehensive callback for monitoring TACTiS-2 Stage 2 training progress.
    
    This callback provides:
    1. Real-time copula loss tracking
    2. Multivariate validation metrics (Energy Score, Sample Diversity)
    3. Stage-aware optimization metrics for Optuna
    4. Gradient flow monitoring for copula parameters
    
    The callback automatically detects when Stage 2 begins and switches
    to appropriate monitoring modes.
    """
    
    def __init__(self, 
                 sample_validation_every_n_epochs: int = 5,
                 num_validation_samples: int = 100,
                 compute_energy_score: bool = True,
                 log_gradient_norms: bool = True,
                 max_batch_size_for_sampling: int = 8):
        """
        Parameters:
        -----------
        sample_validation_every_n_epochs : int
            Frequency for expensive sample-based validation metrics
        num_validation_samples : int  
            Number of samples to generate for multivariate metrics
        compute_energy_score : bool
            Whether to compute Energy Score (computationally expensive)
        log_gradient_norms : bool
            Whether to log gradient norms for copula parameters
        max_batch_size_for_sampling : int
            Maximum batch size to use for sampling (to prevent OOM)
        """
        super().__init__()
        self.sample_validation_every_n_epochs = sample_validation_every_n_epochs
        self.num_validation_samples = num_validation_samples
        self.compute_energy_score = compute_energy_score
        self.log_gradient_norms = log_gradient_norms
        self.max_batch_size_for_sampling = max_batch_size_for_sampling
        
        # Track metrics across epochs for trend analysis
        self.copula_loss_history = []
        self.energy_score_history = []
        self.sample_diversity_history = []
        
        # Store previous values for change detection
        self._prev_copula_loss = None
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor copula-specific metrics during training."""
        
        # Only monitor in Stage 2
        if not self._is_stage_2(pl_module):
            return
            
        # Access the TACTiS model to get loss components
        tactis_model = self._get_tactis_model(pl_module)
        
        if self._has_loss_components(tactis_model):
            marginal_logdet = tactis_model.marginal_logdet.item()
            copula_loss = tactis_model.copula_loss.item() 
            total_loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
            
            # Log the key Stage 2 metrics
            pl_module.log("train_marginal_logdet", marginal_logdet, 
                         on_step=True, on_epoch=True, prog_bar=False)
            pl_module.log("train_copula_loss", copula_loss, 
                         on_step=True, on_epoch=True, prog_bar=True)
            
            # Log copula contribution ratio
            if total_loss != 0:
                copula_ratio = abs(copula_loss / total_loss)
                pl_module.log("train_copula_ratio", copula_ratio, 
                             on_step=True, on_epoch=True, prog_bar=False)
            
            # Track copula loss for trend analysis
            self.copula_loss_history.append(copula_loss)
            
            # Log copula loss change every 50 steps
            if batch_idx % 50 == 0:
                if self._prev_copula_loss is not None:
                    copula_change = copula_loss - self._prev_copula_loss
                    pl_module.log("train_copula_change", copula_change, 
                                 on_step=True, on_epoch=False, prog_bar=False)
                self._prev_copula_loss = copula_loss
                
            # Log gradient norms if requested
            if self.log_gradient_norms and batch_idx % 100 == 0:
                self._log_copula_gradient_norms(pl_module)
                
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor validation metrics with copula focus."""
        
        # Only monitor in Stage 2
        if not self._is_stage_2(pl_module):
            return
            
        # Access loss components for validation
        tactis_model = self._get_tactis_model(pl_module)
        
        if self._has_loss_components(tactis_model):
            marginal_logdet = tactis_model.marginal_logdet.item()
            copula_loss = tactis_model.copula_loss.item()
            
            # Log validation copula metrics
            pl_module.log("val_marginal_logdet", marginal_logdet, 
                         on_step=False, on_epoch=True, prog_bar=False)
            pl_module.log("val_copula_loss", copula_loss, 
                         on_step=False, on_epoch=True, prog_bar=True)
            
            # CRITICAL: Use copula loss as the primary validation metric for Stage 2
            # This is what Optuna will use for optimization
            pl_module.log("val_copula_loss_primary", copula_loss, 
                         on_step=False, on_epoch=True, prog_bar=False)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute expensive multivariate metrics periodically."""
        
        # Only in Stage 2 and at specified intervals
        if (not self._is_stage_2(pl_module) or
            trainer.current_epoch % self.sample_validation_every_n_epochs != 0):
            return
            
        logger.info(f"Computing Stage 2 multivariate metrics at epoch {trainer.current_epoch}")
        
        try:
            # Get a validation batch for sampling
            val_dataloader = trainer.val_dataloaders[0] if trainer.val_dataloaders else None
            if val_dataloader is None:
                logger.warning("No validation dataloader available for Stage 2 metrics")
                return
                
            # Get first batch for metrics computation
            batch = next(iter(val_dataloader))
            batch = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Compute sample-based metrics
            metrics = self._compute_multivariate_metrics(pl_module, batch)
            
            # Log multivariate metrics
            for metric_name, value in metrics.items():
                pl_module.log(f"val_{metric_name}", value, on_epoch=True, prog_bar=False)
                
            # Track energy score for model selection
            if 'energy_score' in metrics:
                self.energy_score_history.append(metrics['energy_score'])
                # Log as potential alternative model selection metric
                pl_module.log("val_energy_score_primary", metrics['energy_score'], 
                             on_epoch=True, prog_bar=True)
                
            logger.info(f"Stage 2 multivariate metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to compute Stage 2 multivariate metrics: {e}")
    
    def _is_stage_2(self, pl_module) -> bool:
        """Check if model is currently in Stage 2."""
        return hasattr(pl_module, 'stage') and pl_module.stage == 2
    
    def _get_tactis_model(self, pl_module):
        """Get the TACTiS model from the lightning module."""
        if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'tactis'):
            return pl_module.model.tactis
        return None
    
    def _has_loss_components(self, tactis_model) -> bool:
        """Check if TACTiS model has computed loss components."""
        return (tactis_model is not None and 
                hasattr(tactis_model, 'marginal_logdet') and 
                hasattr(tactis_model, 'copula_loss'))
    
    def _log_copula_gradient_norms(self, pl_module):
        """Log gradient norms for copula parameters."""
        copula_grad_norm = 0.0
        marginal_grad_norm = 0.0
        copula_param_count = 0
        marginal_param_count = 0
        
        tactis_model = self._get_tactis_model(pl_module)
        if tactis_model is None:
            return
        
        for name, param in tactis_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if self._is_copula_parameter(name):
                    copula_grad_norm += grad_norm**2
                    copula_param_count += 1
                elif self._is_marginal_parameter(name):
                    marginal_grad_norm += grad_norm**2
                    marginal_param_count += 1
        
        if copula_param_count > 0:
            copula_grad_norm = (copula_grad_norm**0.5) / copula_param_count
            pl_module.log("train_copula_grad_norm", copula_grad_norm, 
                         on_step=True, on_epoch=False, prog_bar=False)
        
        if marginal_param_count > 0:
            marginal_grad_norm = (marginal_grad_norm**0.5) / marginal_param_count
            pl_module.log("train_marginal_grad_norm", marginal_grad_norm, 
                         on_step=True, on_epoch=False, prog_bar=False)
            
    def _is_copula_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to copula components."""
        copula_keywords = ['copula', 'attentional', 'decoder.copula']
        return any(keyword in param_name.lower() for keyword in copula_keywords)
    
    def _is_marginal_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to marginal/flow components."""
        marginal_keywords = ['marginal', 'flow', 'decoder.marginal', 'dsf']
        return any(keyword in param_name.lower() for keyword in marginal_keywords)
        
    def _compute_multivariate_metrics(self, pl_module, batch) -> Dict[str, float]:
        """Compute multivariate validation metrics using sampling."""
        metrics = {}
        
        try:
            pl_module.eval()
            with torch.no_grad():
                # Extract batch components
                past_target = batch["past_target"]
                future_target = batch["future_target"]
                past_time_feat = batch["past_time_feat"]
                future_time_feat = batch["future_time_feat"]
                
                batch_size = past_target.shape[0]
                limited_batch_size = min(batch_size, self.max_batch_size_for_sampling)
                
                # Generate samples using the model's sample method
                tactis_model = self._get_tactis_model(pl_module)
                if tactis_model and hasattr(tactis_model, 'sample'):
                    samples = tactis_model.sample(
                        num_samples=self.num_validation_samples,
                        hist_time=past_time_feat[:limited_batch_size],
                        hist_value=past_target[:limited_batch_size],
                        pred_time=future_time_feat[:limited_batch_size]
                    )
                    
                    # Convert to numpy for metric computation
                    samples_np = samples.cpu().numpy()  # [num_samples, batch, series, pred_len]
                    targets_np = future_target[:limited_batch_size].cpu().numpy()  # [batch, series, pred_len]
                    
                    # Compute sample diversity (uncertainty measure)
                    sample_diversity = float(np.mean(np.std(samples_np, axis=0)))
                    metrics['sample_diversity'] = sample_diversity
                    self.sample_diversity_history.append(sample_diversity)
                    
                    # Compute Energy Score if requested
                    if self.compute_energy_score:
                        energy_score = self._compute_energy_score(samples_np, targets_np)
                        metrics['energy_score'] = energy_score
                        
                    # Compute cross-variable correlation preservation
                    correlation_error = self._compute_correlation_error(samples_np, targets_np)
                    metrics['correlation_error'] = correlation_error
                    
                    # Compute mean forecast bias
                    mean_samples = np.mean(samples_np, axis=0)
                    forecast_bias = float(np.mean(mean_samples - targets_np))
                    metrics['forecast_bias'] = forecast_bias
                    
                else:
                    logger.warning("TACTiS model does not have sample method for multivariate metrics")
                    
        except Exception as e:
            logger.error(f"Error computing multivariate metrics: {e}")
            
        return metrics
    
    def _compute_energy_score(self, samples: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Energy Score for multivariate probabilistic forecasting.
        
        Energy Score = E[||X - Y||] - 0.5 * E[||X1 - X2||]
        where X is forecast, Y is observation, X1,X2 are independent forecasts
        
        Parameters:
        -----------
        samples : np.ndarray [num_samples, batch, series, pred_len]
        targets : np.ndarray [batch, series, pred_len]
        
        Returns:
        --------
        float : Energy Score (lower is better)
        """
        try:
            num_samples, batch_size, num_series, pred_len = samples.shape
            
            # Reshape for easier computation: [batch*pred_len, num_samples, num_series]
            samples_reshaped = samples.transpose(1, 3, 0, 2).reshape(-1, num_samples, num_series)
            targets_reshaped = targets.transpose(0, 2, 1).reshape(-1, num_series)
            
            energy_scores = []
            
            for i in range(min(samples_reshaped.shape[0], 100)):  # Limit for performance
                sample_set = samples_reshaped[i]  # [num_samples, num_series]
                target = targets_reshaped[i]      # [num_series]
                
                # E[||X - Y||] where X is forecast, Y is observation
                term1 = np.mean([np.linalg.norm(sample - target) for sample in sample_set])
                
                # 0.5 * E[||X1 - X2||] where X1, X2 are independent forecasts
                if num_samples > 1:
                    pairwise_distances = []
                    # Sample pairs to avoid O(n^2) computation for large sample sets
                    num_pairs = min(50, num_samples * (num_samples - 1) // 2)
                    indices = np.random.choice(num_samples, size=min(20, num_samples), replace=False)
                    
                    for j in range(len(indices)):
                        for k in range(j+1, len(indices)):
                            pairwise_distances.append(
                                np.linalg.norm(sample_set[indices[j]] - sample_set[indices[k]])
                            )
                            
                    term2 = 0.5 * np.mean(pairwise_distances) if pairwise_distances else 0.0
                else:
                    term2 = 0.0
                
                energy_scores.append(term1 - term2)
            
            return float(np.mean(energy_scores))
            
        except Exception as e:
            logger.error(f"Error computing Energy Score: {e}")
            return float('inf')
    
    def _compute_correlation_error(self, samples: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute how well samples preserve cross-variable correlations.
        
        Parameters:
        -----------
        samples : np.ndarray [num_samples, batch, series, pred_len]
        targets : np.ndarray [batch, series, pred_len]
        
        Returns:
        --------
        float : Mean squared error between sample and target correlations
        """
        try:
            # Compute correlations across series for each time step
            batch_size, num_series, pred_len = targets.shape
            
            if num_series < 2:
                return 0.0  # Need at least 2 series for correlation
            
            correlation_errors = []
            
            # Sample a few time steps to avoid expensive computation
            time_indices = np.random.choice(pred_len, size=min(5, pred_len), replace=False)
            
            for t in time_indices:
                # Get target correlations at time t across series
                target_slice = targets[:, :, t]  # [batch, series]
                if np.std(target_slice) > 1e-6:  # Avoid degenerate cases
                    target_corr = np.corrcoef(target_slice.T)  # [series, series]
                    
                    # Get sample correlations at time t
                    sample_slice = samples[:, :, :, t]  # [num_samples, batch, series]
                    sample_corrs = []
                    
                    for s in range(min(10, samples.shape[0])):  # Limit samples for performance
                        sample_data = sample_slice[s]  # [batch, series]
                        if np.std(sample_data) > 1e-6:
                            sample_corr = np.corrcoef(sample_data.T)
                            sample_corrs.append(sample_corr)
                    
                    if sample_corrs:
                        mean_sample_corr = np.mean(sample_corrs, axis=0)
                        correlation_error = np.mean((target_corr - mean_sample_corr)**2)
                        correlation_errors.append(correlation_error)
            
            return float(np.mean(correlation_errors)) if correlation_errors else 0.0
            
        except Exception as e:
            logger.error(f"Error computing correlation error: {e}")
            return 0.0