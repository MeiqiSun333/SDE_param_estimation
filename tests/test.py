"""
Unit and integration tests for the Network Model.

Run these tests using:
    pytest test_model.py
"""

import pytest
import numpy as np
from src.models.network_model import EmotionNetworkModel


class TestEmotionNetworkModel:
    """
    Unit tests covering the core network simulation behaviors.
    """
    
    def test_initialization(self):
        """Test if the model initializes with correct default attributes."""

        model = EmotionNetworkModel(stochastic=True, random_seed=42)
        # assert model.num_states == 4
        # assert model.dt == 0.5
        assert model.stochastic is True
    
    def test_simulation_shape(self):
        """Test if the simulation output matrices have aligned dimensions."""

        model = EmotionNetworkModel(dt=0.5, random_seed=42)
        end_time = 10
        time, states = model.simulate(end_time=end_time)
        
        expected_steps = int(end_time / 0.5) + 1
        assert time.shape == (expected_steps,)
        assert states.shape == (4, expected_steps)
    
    def test_simulation_bounds(self):
        """Test that the bounded diffusion keeps all state values strictly within [0, 1]."""

        model = EmotionNetworkModel(stochastic=True, random_seed=42)
        _, states = model.simulate(end_time=30)
        
        assert np.all(states >= 0.0), "State violation: Values dropped below 0"
        assert np.all(states <= 1.0), "State violation: Values exceeded 1"
    
    def test_deterministic_simulation(self):
        """Test that turning off stochasticity yields identical deterministic trajectories."""

        model1 = EmotionNetworkModel(stochastic=False, random_seed=42)
        model2 = EmotionNetworkModel(stochastic=False, random_seed=999) # Different seed should not matter
        
        _, states1 = model1.simulate(end_time=15)
        _, states2 = model2.simulate(end_time=15)
        
        np.testing.assert_array_almost_equal(states1, states2, decimal=6)


class TestIntegration:
    """Integration tests evaluating data flow pipeline reproducibility."""
    
    def test_reproducibility(self):
        """Test that enforcing a random seed produces identical stochastic paths."""

        model1 = EmotionNetworkModel(stochastic=True, random_seed=42)
        model2 = EmotionNetworkModel(stochastic=True, random_seed=42)
        
        _, states1 = model1.simulate(end_time=20)
        _, states2 = model2.simulate(end_time=20)
        
        np.testing.assert_array_equal(states1, states2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])