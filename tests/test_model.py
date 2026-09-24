import numpy as np

from src.model.four_node import (
    diffusion_numpy,
    drift_numpy,
    simulate_numpy,
    targets_numpy,
)
from src.model.simulator import EmotionNetworkModel


def test_four_node_equations_have_expected_shape(true_parameters):
    states = np.array([0.2, 0.4, 0.6, 0.8])
    assert targets_numpy(states, true_parameters[8:]).shape == (4,)
    assert drift_numpy(states, true_parameters[:4], true_parameters[8:]).shape == (4,)
    assert diffusion_numpy(states, true_parameters[4:8]).shape == (4,)


def test_batch_simulation_is_bounded_and_reproducible(true_parameters):
    first = simulate_numpy(
        true_parameters, n_trajectories=3, n_steps=20, seed=42
    )
    second = simulate_numpy(
        true_parameters, n_trajectories=3, n_steps=20, seed=42
    )
    assert first.shape == (3, 20, 4)
    assert np.all((first > 0) & (first < 1))
    np.testing.assert_array_equal(first, second)


def test_user_facing_simulator_returns_time_by_state_arrays():
    model = EmotionNetworkModel(dt=0.3, stochastic=False)
    time, states = model.simulate(end_time=1.0)
    np.testing.assert_allclose(time, [0.0, 0.3, 0.6, 0.9, 1.0])
    assert states.shape == (4, 5)


def test_deterministic_simulation_does_not_depend_on_seed():
    first = EmotionNetworkModel(stochastic=False, random_seed=1)
    second = EmotionNetworkModel(stochastic=False, random_seed=2)
    np.testing.assert_allclose(
        first.simulate(end_time=2)[1],
        second.simulate(end_time=2)[1],
    )
