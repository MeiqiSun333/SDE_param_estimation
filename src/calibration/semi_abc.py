"""Semi-automatic ABC with regression-based summary statistics."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.calibration._common import (
    BETA_LOWER,
    BETA_NAMES,
    BETA_UPPER,
    replace_betas,
    summary_statistics,
    validate_parameters,
)
from src.calibration.abc_smc import run_pyabc
from src.model.four_node import simulate_numpy


class SemiAutomaticABCCalibrator:
    def __init__(self, fixed_parameters):
        self.fixed_parameters = validate_parameters(fixed_parameters).copy()
        self.regressor = None
        self.summary_names = None
        self.pilot_r_squared = None

    def fit_pilot(
        self,
        n_pilot=2000,
        n_trajectories=20,
        n_steps=100,
        delta_t=0.01,
        ridge_alpha=1.0,
        seed=0,
    ):
        if n_pilot <= 0:
            raise ValueError("n_pilot must be positive")
        rng = np.random.default_rng(seed)
        betas = rng.uniform(BETA_LOWER, BETA_UPPER, size=(n_pilot, 4))
        summaries = []

        for index, beta in enumerate(betas):
            simulated = simulate_numpy(
                replace_betas(self.fixed_parameters, beta),
                n_trajectories=n_trajectories,
                n_steps=n_steps,
                delta_t=delta_t,
                seed=seed + index + 1,
            )
            statistics = summary_statistics(simulated)
            if self.summary_names is None:
                self.summary_names = tuple(sorted(statistics))
            summaries.append(
                [statistics[name] for name in self.summary_names]
            )

        features = np.asarray(summaries)
        self.regressor = make_pipeline(
            StandardScaler(),
            Ridge(alpha=ridge_alpha),
        ).fit(features, betas)
        prediction = self.regressor.predict(features)
        residual = ((betas - prediction) ** 2).sum(axis=0)
        total = ((betas - betas.mean(axis=0)) ** 2).sum(axis=0)
        self.pilot_r_squared = 1.0 - residual / total
        return self.pilot_r_squared.copy()

    def _learned_summary(self, trajectories):
        if self.regressor is None:
            raise RuntimeError("fit_pilot must be called before fit")
        statistics = summary_statistics(trajectories)
        vector = np.array(
            [[statistics[name] for name in self.summary_names]]
        )
        prediction = self.regressor.predict(vector)[0]
        return {
            f"beta{i + 1}_hat": float(value)
            for i, value in enumerate(prediction)
        }

    def fit(
        self,
        observed_trajectories,
        delta_t=0.01,
        population_size=100,
        max_populations=20,
        minimum_epsilon=0.05,
        minimum_acceptance_rate=5e-3,
        posterior_draws=2000,
        seed=0,
    ):
        observed = np.asarray(observed_trajectories, dtype=float)
        if observed.ndim != 3 or observed.shape[2] != 4:
            raise ValueError("observed_trajectories must have shape (n, time, 4)")
        observed_summary = self._learned_summary(observed)
        simulation_number = 0

        def model(parameter):
            nonlocal simulation_number
            betas = np.array([parameter[name] for name in BETA_NAMES])
            simulated = simulate_numpy(
                replace_betas(self.fixed_parameters, betas),
                n_trajectories=observed.shape[0],
                n_steps=observed.shape[1],
                delta_t=delta_t,
                seed=seed + simulation_number,
            )
            simulation_number += 1
            return self._learned_summary(simulated)

        return run_pyabc(
            model,
            observed_summary,
            "semi_abc",
            population_size,
            max_populations,
            minimum_epsilon,
            minimum_acceptance_rate,
            posterior_draws,
            seed,
        )
