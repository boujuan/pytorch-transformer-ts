"""
TACTiS-2 (Temporal And Cross-Sectional Transformers in Sequence) Model.

TACTiS is a transformer-based model for probabilistic multivariate time series forecasting
that leverages normalizing flows and copulas to learn complex multivariate distributions.

This is an implementation of the TACTiS model integrated with the pytorch-transformer-ts 
framework, compatible with GluonTS.
"""

from .tactis2 import TACTiS
from .module import TACTiS2Model
from .lightning_module import TACTiS2LightningModule
from .estimator import TACTiS2Estimator

__all__ = [
    "TACTiS",
    "TACTiS2Model",
    "TACTiS2LightningModule",
    "TACTiS2Estimator",
]
