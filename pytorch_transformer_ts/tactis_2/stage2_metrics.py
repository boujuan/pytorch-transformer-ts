#!/usr/bin/env python3
"""
Stage 2 Metrics Utilities for TACTiS-2

This module provides utility functions for computing and managing
Stage 2 specific metrics, including:
- Stage-aware optimization metrics for Optuna
- Multivariate forecasting evaluation metrics
- Copula learning assessment functions

These utilities support proper evaluation of two-stage training
where marginal and copula components have different optimization
objectives and success criteria.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Any, Union
import logging

logger = logging.getLogger(__name__)


def compute_stage_aware_optimization_metric(
    current_epoch: int,
    stage2_start_epoch: int,
    total_loss: float,
    copula_loss: Optional[float] = None,
    fallback_to_total: bool = True
) -> float:
    """
    Compute the appropriate optimization metric based on training stage.
    
    This is the core function for stage-aware Optuna optimization:
    - Stage 1: Use total NLL (meaningful signal)
    - Stage 2: Use copula loss (true learning signal)
    
    Parameters:
    -----------
    current_epoch : int
        Current training epoch
    stage2_start_epoch : int
        Epoch when Stage 2 begins
    total_loss : float
        Total training loss (NLL)
    copula_loss : Optional[float]
        Copula loss component (only available in Stage 2)
    fallback_to_total : bool
        Whether to fallback to total_loss if copula_loss unavailable
        
    Returns:
    --------
    float : Optimization metric for Optuna
    """
    if current_epoch < stage2_start_epoch:
        # Stage 1: Total NLL is the meaningful optimization target
        return float(total_loss)
    else:
        # Stage 2: Copula loss is the true learning signal
        if copula_loss is not None:
            return float(copula_loss)
        elif fallback_to_total:
            logger.warning(f"Copula loss not available at epoch {current_epoch}, "
                          f"falling back to total loss")
            return float(total_loss)
        else:
            raise ValueError(f"Copula loss not available at epoch {current_epoch} "
                           f"and fallback disabled")


def validate_stage_aware_metric(
    metric_value: Union[float, torch.Tensor, np.ndarray],
    stage: int,
    epoch: int,
    metric_name: str = "stage_aware_metric"
) -> float:
    """
    Validate and convert stage-aware metric to float.
    
    Parameters:
    -----------
    metric_value : Union[float, torch.Tensor, np.ndarray]
        Raw metric value
    stage : int
        Current training stage (1 or 2)
    epoch : int
        Current epoch for logging
    metric_name : str
        Name of the metric for error reporting
        
    Returns:
    --------
    float : Validated metric value
        
    Raises:
    -------
    ValueError : If metric cannot be converted or is invalid
    """
    try:
        # Handle tensor/array values
        if hasattr(metric_value, 'item'):
            metric_value = metric_value.item()
        elif isinstance(metric_value, (np.ndarray, torch.Tensor)) and metric_value.size == 1:
            metric_value = metric_value.item()
        
        metric_value = float(metric_value)
        
        # Validate metric value
        if np.isnan(metric_value) or np.isinf(metric_value):
            raise ValueError(f"Invalid metric value: {metric_value}")
        
        logger.debug(f"Stage {stage}, Epoch {epoch}: {metric_name} = {metric_value}")
        return metric_value
        
    except (TypeError, ValueError) as e:
        error_msg = (f"Error validating {metric_name} at stage {stage}, epoch {epoch}: "
                    f"value={metric_value}, error={e}")
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def compute_copula_learning_indicators(
    copula_loss_history: List[float],
    window_size: int = 10,
    min_improvement_threshold: float = 1e-6
) -> Dict[str, float]:
    """
    Compute indicators of copula learning progress.
    
    Parameters:
    -----------
    copula_loss_history : List[float]
        History of copula loss values
    window_size : int
        Window size for trend analysis
    min_improvement_threshold : float
        Minimum improvement to consider as learning
        
    Returns:
    --------
    Dict[str, float] : Learning indicators
        - trend_slope: Linear trend slope (negative = improving)
        - recent_improvement: Improvement in recent window
        - total_improvement: Total improvement from start
        - volatility: Standard deviation of recent changes
    """
    if len(copula_loss_history) < 2:
        return {
            'trend_slope': 0.0,
            'recent_improvement': 0.0,
            'total_improvement': 0.0,
            'volatility': 0.0
        }
    
    history = np.array(copula_loss_history)
    
    # Compute linear trend slope
    x = np.arange(len(history))
    trend_slope = np.polyfit(x, history, 1)[0] if len(history) > 1 else 0.0
    
    # Recent improvement (comparing recent window to previous window)
    if len(history) >= 2 * window_size:
        recent_window = history[-window_size:]
        previous_window = history[-2*window_size:-window_size]
        recent_improvement = np.mean(previous_window) - np.mean(recent_window)
    else:
        recent_improvement = history[0] - history[-1]
    
    # Total improvement from start
    total_improvement = history[0] - history[-1]
    
    # Volatility (standard deviation of recent changes)
    if len(history) >= window_size:
        recent_changes = np.diff(history[-window_size:])
        volatility = np.std(recent_changes)
    else:
        volatility = np.std(np.diff(history)) if len(history) > 1 else 0.0
    
    return {
        'trend_slope': float(trend_slope),
        'recent_improvement': float(recent_improvement),
        'total_improvement': float(total_improvement),
        'volatility': float(volatility)
    }


def assess_copula_learning_health(
    copula_loss_history: List[float],
    energy_score_history: List[float] = None,
    sample_diversity_history: List[float] = None,
    min_epochs_for_assessment: int = 20
) -> Dict[str, Any]:
    """
    Assess the health of copula learning based on multiple indicators.
    
    Parameters:
    -----------
    copula_loss_history : List[float]
        History of copula loss values
    energy_score_history : List[float], optional
        History of Energy Score values
    sample_diversity_history : List[float], optional  
        History of sample diversity values
    min_epochs_for_assessment : int
        Minimum epochs needed for reliable assessment
        
    Returns:
    --------
    Dict[str, Any] : Health assessment
        - learning_status: "healthy", "concerning", or "failing"
        - primary_issues: List of identified issues
        - recommendations: List of recommendations
        - indicators: Computed learning indicators
    """
    if len(copula_loss_history) < min_epochs_for_assessment:
        return {
            'learning_status': 'insufficient_data',
            'primary_issues': ['Need more training epochs for assessment'],
            'recommendations': [f'Continue training for at least {min_epochs_for_assessment} Stage 2 epochs'],
            'indicators': {}
        }
    
    indicators = compute_copula_learning_indicators(copula_loss_history)
    issues = []
    recommendations = []
    
    # Check for learning progress
    if indicators['trend_slope'] > 0:
        issues.append('Copula loss is increasing (getting worse)')
        recommendations.append('Check Stage 2 learning rate - may be too high')
    elif abs(indicators['trend_slope']) < 1e-7:
        issues.append('Copula loss is completely flat - no learning detected')
        recommendations.append('Check if copula parameters are frozen or learning rate too low')
    elif indicators['recent_improvement'] < 1e-6:
        issues.append('No recent improvement in copula loss')
        recommendations.append('Consider learning rate scheduling or early stopping')
    
    # Check volatility
    if indicators['volatility'] > 0.1:
        issues.append('High volatility in copula loss - training may be unstable')
        recommendations.append('Reduce Stage 2 learning rate or increase gradient clipping')
    
    # Check sample diversity if available
    if sample_diversity_history and len(sample_diversity_history) > 5:
        diversity_trend = np.polyfit(np.arange(len(sample_diversity_history)), 
                                   sample_diversity_history, 1)[0]
        if diversity_trend < -1e-3:
            issues.append('Sample diversity is decreasing - model may be overconfident')
            recommendations.append('Check copula architecture or increase regularization')
    
    # Determine overall health status
    if len(issues) == 0:
        learning_status = 'healthy'
    elif len(issues) <= 2 and 'increasing' not in str(issues) and 'flat' not in str(issues):
        learning_status = 'concerning'
    else:
        learning_status = 'failing'
    
    return {
        'learning_status': learning_status,
        'primary_issues': issues,
        'recommendations': recommendations,
        'indicators': indicators
    }


def compute_multivariate_forecast_quality(
    samples: np.ndarray, 
    targets: np.ndarray,
    compute_energy_score: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive multivariate forecast quality metrics.
    
    Parameters:
    -----------
    samples : np.ndarray [num_samples, batch, series, pred_len]
        Forecast samples
    targets : np.ndarray [batch, series, pred_len]
        Ground truth targets
    compute_energy_score : bool
        Whether to compute Energy Score (expensive)
        
    Returns:
    --------
    Dict[str, float] : Quality metrics
        - energy_score: Energy Score (if computed)
        - sample_diversity: Mean standard deviation across samples
        - forecast_bias: Mean bias across all predictions
        - correlation_preservation: How well cross-variable correlations are preserved
        - coverage_90: 90% prediction interval coverage
    """
    metrics = {}
    
    try:
        num_samples, batch_size, num_series, pred_len = samples.shape
        
        # Sample diversity
        sample_diversity = float(np.mean(np.std(samples, axis=0)))
        metrics['sample_diversity'] = sample_diversity
        
        # Forecast bias
        mean_samples = np.mean(samples, axis=0)
        forecast_bias = float(np.mean(mean_samples - targets))
        metrics['forecast_bias'] = forecast_bias
        
        # 90% Prediction interval coverage
        pi_lower = np.percentile(samples, 5, axis=0)
        pi_upper = np.percentile(samples, 95, axis=0)
        coverage_90 = float(np.mean((targets >= pi_lower) & (targets <= pi_upper)))
        metrics['coverage_90'] = coverage_90
        
        # Correlation preservation (simplified)
        if num_series > 1:
            correlation_error = _compute_correlation_error_fast(samples, targets)
            metrics['correlation_preservation'] = 1.0 - min(correlation_error, 1.0)
        
        # Energy Score (expensive)
        if compute_energy_score:
            energy_score = _compute_energy_score_fast(samples, targets)
            metrics['energy_score'] = energy_score
            
    except Exception as e:
        logger.error(f"Error computing multivariate forecast quality: {e}")
        
    return metrics


def _compute_energy_score_fast(samples: np.ndarray, targets: np.ndarray) -> float:
    """Fast approximation of Energy Score using subsampling."""
    try:
        num_samples, batch_size, num_series, pred_len = samples.shape
        
        # Subsample for performance
        max_timesteps = min(50, batch_size * pred_len)
        max_samples_for_pairs = min(20, num_samples)
        
        # Flatten to [timesteps, num_samples, num_series]
        samples_flat = samples.transpose(1, 3, 0, 2).reshape(-1, num_samples, num_series)
        targets_flat = targets.transpose(0, 2, 1).reshape(-1, num_series)
        
        # Random subsample of timesteps
        timestep_indices = np.random.choice(len(samples_flat), 
                                          size=min(max_timesteps, len(samples_flat)), 
                                          replace=False)
        
        energy_scores = []
        
        for i in timestep_indices:
            sample_set = samples_flat[i][:max_samples_for_pairs]
            target = targets_flat[i]
            
            # E[||X - Y||]
            term1 = np.mean([np.linalg.norm(sample - target) for sample in sample_set])
            
            # 0.5 * E[||X1 - X2||]
            if len(sample_set) > 1:
                # Sample pairs for performance
                num_pairs = min(10, len(sample_set) * (len(sample_set) - 1) // 2)
                indices = np.random.choice(len(sample_set), size=min(6, len(sample_set)), 
                                         replace=False)
                
                pairwise_distances = []
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
        logger.error(f"Error computing fast Energy Score: {e}")
        return float('inf')


def _compute_correlation_error_fast(samples: np.ndarray, targets: np.ndarray) -> float:
    """Fast approximation of correlation error using subsampling."""
    try:
        batch_size, num_series, pred_len = targets.shape
        
        if num_series < 2:
            return 0.0
        
        # Sample a few time steps
        time_indices = np.random.choice(pred_len, size=min(3, pred_len), replace=False)
        correlation_errors = []
        
        for t in time_indices:
            target_slice = targets[:, :, t]
            if np.std(target_slice) > 1e-6:
                target_corr = np.corrcoef(target_slice.T)
                
                # Sample correlations
                sample_slice = samples[:, :, :, t]
                sample_corrs = []
                
                # Use subset of samples for performance
                sample_indices = np.random.choice(samples.shape[0], 
                                                size=min(5, samples.shape[0]), 
                                                replace=False)
                
                for s in sample_indices:
                    sample_data = sample_slice[s]
                    if np.std(sample_data) > 1e-6:
                        sample_corr = np.corrcoef(sample_data.T)
                        sample_corrs.append(sample_corr)
                
                if sample_corrs:
                    mean_sample_corr = np.mean(sample_corrs, axis=0)
                    correlation_error = np.mean((target_corr - mean_sample_corr)**2)
                    correlation_errors.append(correlation_error)
        
        return float(np.mean(correlation_errors)) if correlation_errors else 0.0
        
    except Exception as e:
        logger.error(f"Error computing fast correlation error: {e}")
        return 0.0