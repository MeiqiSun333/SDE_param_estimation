"""User-facing simulator for the fixed four-node model."""

import numpy as np

from src.model.four_node import simulate_numpy
from src.utils.config import load_config

_DEFAULT = object()


class EmotionNetworkModel:
    def __init__(
        self,
        parameters=None,
        dt=None,
        stochastic=None,
        random_seed=_DEFAULT,
        config_path=None,
    ):
        config = load_config(config_path)
        self.parameters = np.asarray(
            config["parameters"] if parameters is None else parameters,
            dtype=float,
        )
        if self.parameters.shape != (12,):
            raise ValueError("parameters must contain twelve values")
        self.dt = config["dt"] if dt is None else float(dt)
        self.stochastic = (
            config["stochastic"] if stochastic is None else bool(stochastic)
        )
        self.random_seed = (
            config["random_seed"] if random_seed is _DEFAULT else random_seed
        )
        self.end_time = config["end_time"]
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    def simulate(self, end_time=None, initial_states=None):
        end_time = self.end_time if end_time is None else float(end_time)
        if end_time < 0:
            raise ValueError("end_time must be non-negative")
        steps = int(np.ceil(end_time / self.dt))
        time = np.minimum(np.arange(steps + 1) * self.dt, end_time)
        initial = (
            np.full(4, 0.5)
            if initial_states is None
            else np.asarray(initial_states, dtype=float)
        )
        trajectories = simulate_numpy(
            self.parameters,
            n_trajectories=1,
            n_steps=len(time),
            delta_t=np.diff(time),
            initial_states=initial,
            seed=self.random_seed,
            stochastic=self.stochastic,
        )
        return time, trajectories[0].T
