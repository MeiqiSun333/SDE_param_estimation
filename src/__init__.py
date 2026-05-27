"""
Emotion Network Model Package

A computational framework for modeling emotion dynamics using network-based 
approaches with stochastic differential equations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.models.network_model import EmotionNetworkModel, LegacyEmotionNetworkModel
from src.data.preprocessing import EmotionDataProcessor
from src.visualization.plots import NetworkVisualizer

__all__ = [
    "EmotionNetworkModel",
    "LegacyEmotionNetworkModel", 
    "EmotionDataProcessor",
    "NetworkVisualizer",
]
