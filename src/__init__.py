"""
Emotion Network Model Package

A computational framework for modeling dynamics using network-based 
approaches with stochastic differential equations.
"""

__version__ = "0.1.0"
__author__ = "MeiqiSun333"

from src.models.network_model import EmotionNetworkModel
from src.data.preprocessing import EmotionDataProcessor
from src.visualization.plots import NetworkVisualizer

from src.models.optimization import run_estimation, JAXOptimizer

__all__ = [
    "EmotionNetworkModel",
    "EmotionDataProcessor",
    "NetworkVisualizer",
    "run_estimation",
    "JAXOptimizer",
]
