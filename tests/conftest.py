import numpy as np
import pytest

from src.calibration._common import CalibrationData
from src.model.four_node import simulate_numpy


@pytest.fixture
def true_parameters():
    return np.array(
        [0.4, 0.6, 0.3, 0.5, 0.15, 0.20, 0.18, 0.25, 0.8, 0.5, -0.4, 0.7]
    )


@pytest.fixture
def trajectories(true_parameters):
    return simulate_numpy(
        true_parameters,
        n_trajectories=6,
        n_steps=30,
        delta_t=0.02,
        seed=7,
    )


@pytest.fixture
def calibration_data(trajectories):
    return CalibrationData.from_trajectories(trajectories, 0.02)
