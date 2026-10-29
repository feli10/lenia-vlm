"""Normalized ridge-readout Reservoir epiplexity API."""

from .core import RCEpiplexity, RCEpiplexityOutput, ridge_least_squares
from .online import CovarianceRLSEpiplexity
from .reservoirs import (
    RCEpiplexity1D,
    RCEpiplexityMLP,
)

__all__ = [
    "CovarianceRLSEpiplexity",
    "RCEpiplexity",
    "RCEpiplexity1D",
    "RCEpiplexityMLP",
    "RCEpiplexityOutput",
    "ridge_least_squares",
]
