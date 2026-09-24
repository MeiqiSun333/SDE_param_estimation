import importlib.util

import jax
import numpy as np
import pytest

from src.calibration._common import PARAMETER_NAMES
from src.calibration.abc_smc import ABCSMCCalibrator
from src.calibration.ait_sahalia import AitSahaliaEstimator, ait_sahalia_logpdf
from src.calibration.em import EulerMaruyamaEstimator, euler_maruyama_logpdf
from src.calibration.milstein import MilsteinEstimator, milstein_logpdf
from src.calibration.sbi_npe import SBINPECalibrator
from src.calibration.semi_abc import SemiAutomaticABCCalibrator
from src.calibration.sml import SimulatedMaximumLikelihoodEstimator


@pytest.mark.parametrize(
    "logpdf",
    [euler_maruyama_logpdf, milstein_logpdf, ait_sahalia_logpdf],
)
def test_classical_likelihoods_are_finite(
    logpdf, calibration_data, true_parameters
):
    value, gradient = jax.value_and_grad(logpdf)(
        true_parameters,
        calibration_data.previous[0],
        calibration_data.current[0],
        calibration_data.delta_t[0],
    )
    assert np.isfinite(value)
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize(
    "estimator", [EulerMaruyamaEstimator(), MilsteinEstimator()]
)
def test_direct_likelihood_calibration_runs(estimator, calibration_data):
    result = estimator.fit(calibration_data, max_iterations=3)
    assert result.parameter_names == PARAMETER_NAMES
    assert result.estimate.shape == (12,)
    assert np.isfinite(result.estimate).all()


def test_simulated_maximum_likelihood_runs(calibration_data):
    result = SimulatedMaximumLikelihoodEstimator(
        substeps=2, paths=4, seed=1
    ).fit(calibration_data, max_iterations=2)
    assert result.estimate.shape == (12,)
    assert np.isfinite(result.estimate).all()


@pytest.mark.calibration
def test_ait_sahalia_calibration_runs(calibration_data):
    result = AitSahaliaEstimator().fit(calibration_data, max_iterations=1)
    assert result.estimate.shape == (12,)
    assert np.isfinite(result.estimate).all()


@pytest.mark.calibration
@pytest.mark.skipif(importlib.util.find_spec("pyabc") is None, reason="pyabc missing")
def test_abc_smc_calibrates_simulated_data(trajectories, true_parameters):
    result = ABCSMCCalibrator(true_parameters).fit(
        trajectories,
        delta_t=0.02,
        population_size=5,
        max_populations=1,
        minimum_epsilon=1e6,
        minimum_acceptance_rate=0.0,
        posterior_draws=10,
        seed=2,
    )
    assert result.estimate.shape == (4,)
    assert np.isfinite(result.estimate).all()


@pytest.mark.calibration
@pytest.mark.skipif(importlib.util.find_spec("pyabc") is None, reason="pyabc missing")
def test_semi_abc_calibrates_simulated_data(trajectories, true_parameters):
    calibrator = SemiAutomaticABCCalibrator(true_parameters)
    scores = calibrator.fit_pilot(
        n_pilot=12,
        n_trajectories=2,
        n_steps=30,
        delta_t=0.02,
        seed=3,
    )
    result = calibrator.fit(
        trajectories,
        delta_t=0.02,
        population_size=5,
        max_populations=1,
        minimum_epsilon=1e6,
        minimum_acceptance_rate=0.0,
        posterior_draws=10,
        seed=4,
    )
    assert scores.shape == (4,)
    assert result.estimate.shape == (4,)
    assert np.isfinite(result.estimate).all()


@pytest.mark.calibration
@pytest.mark.skipif(importlib.util.find_spec("sbi") is None, reason="sbi missing")
def test_sbi_calibrates_simulated_data(trajectories):
    calibrator = SBINPECalibrator(density_estimator="mdn")
    parameters, summaries = calibrator.build_training_set(
        n_simulations=20,
        n_trajectories=2,
        n_steps=30,
        delta_t=0.02,
        seed=5,
    )
    calibrator.train(
        parameters,
        summaries,
        seed=5,
        training_kwargs={
            "max_num_epochs": 1,
            "stop_after_epochs": 1,
            "training_batch_size": 10,
        },
    )
    result = calibrator.fit(trajectories, posterior_samples=10, seed=5)
    assert result.estimate.shape == (12,)
    assert np.isfinite(result.estimate).all()
