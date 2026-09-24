"""ABC-SMC calibration for the four beta parameters."""

import tempfile
import logging
import uuid
from pathlib import Path

import numpy as np

from src.calibration._common import (
    BETA_LOWER,
    BETA_NAMES,
    BETA_UPPER,
    CalibrationResult,
    replace_betas,
    summary_statistics,
    validate_parameters,
)
from src.model.four_node import simulate_numpy


def _require_pyabc():
    try:
        from pyabc import ABCSMC, AdaptivePNormDistance, Distribution, RV
        from pyabc.sampler import SingleCoreSampler
    except ImportError as error:
        raise ImportError("ABC-SMC requires the pyabc package") from error
    for name in ("ABC", "ABC.History", "ABC.Population"):
        logging.getLogger(name).setLevel(logging.ERROR)
    return ABCSMC, AdaptivePNormDistance, Distribution, RV, SingleCoreSampler


def _posterior_result(history, method, posterior_draws, seed):
    dataframe, weights = history.get_distribution(m=0)
    values = dataframe.loc[:, BETA_NAMES].to_numpy(dtype=float)
    weights = np.array(weights, dtype=float, copy=True)
    weights /= weights.sum()
    estimate = np.average(values, axis=0, weights=weights)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=posterior_draws, p=weights)
    return CalibrationResult(
        method,
        BETA_NAMES,
        estimate,
        True,
        posterior_samples=values[indices],
    )


def run_pyabc(
    model,
    observed_summary,
    method,
    population_size,
    max_populations,
    minimum_epsilon,
    minimum_acceptance_rate,
    posterior_draws,
    seed,
):
    ABCSMC, AdaptivePNormDistance, Distribution, RV, SingleCoreSampler = (
        _require_pyabc()
    )
    prior = Distribution(
        **{
            name: RV("uniform", float(low), float(high - low))
            for name, low, high in zip(BETA_NAMES, BETA_LOWER, BETA_UPPER)
        }
    )
    database = Path(tempfile.gettempdir()) / f"emotion_abc_{uuid.uuid4().hex}.db"
    abc = ABCSMC(
        models=model,
        parameter_priors=prior,
        distance_function=AdaptivePNormDistance(p=2),
        population_size=population_size,
        sampler=SingleCoreSampler(),
    )
    abc.new(f"sqlite:///{database.as_posix()}", observed_summary)
    history = abc.run(
        minimum_epsilon=minimum_epsilon,
        max_nr_populations=max_populations,
        min_acceptance_rate=minimum_acceptance_rate,
    )
    return _posterior_result(history, method, posterior_draws, seed)


class ABCSMCCalibrator:
    def __init__(self, fixed_parameters):
        self.fixed_parameters = validate_parameters(fixed_parameters).copy()

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
        observed_summary = summary_statistics(observed)
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
            return summary_statistics(simulated)

        return run_pyabc(
            model,
            observed_summary,
            "abc_smc",
            population_size,
            max_populations,
            minimum_epsilon,
            minimum_acceptance_rate,
            posterior_draws,
            seed,
        )
