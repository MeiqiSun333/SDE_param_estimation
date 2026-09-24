"""Pedersen simulated maximum likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from src.calibration._common import (
    CalibrationResult,
    DEFAULT_PARAMETERS,
    FULL_BOUNDS,
    PARAMETER_NAMES,
    validate_parameters,
)
from src.model.four_node import diffusion_jax, drift_jax


def _pedersen_transition_logpdf(
    parameters, previous, current, delta_t, normal_draws, substeps, paths
):
    betas = parameters[:4]
    sigmas = parameters[4:8]
    weights = parameters[8:12]
    step_dt = delta_t / substeps
    states = jnp.broadcast_to(previous[:, None, :], (len(previous), paths, 4))

    for step in range(substeps - 1):
        drift = drift_jax(states, betas, weights)
        diffusion = diffusion_jax(states, sigmas)
        states = states + drift * step_dt[:, None, None]
        states += (
            diffusion
            * jnp.sqrt(step_dt[:, None, None])
            * normal_draws[:, :, step]
        )
        states = jnp.clip(states, 1e-5, 1.0 - 1e-5)

    means = states + drift_jax(states, betas, weights) * step_dt[:, None, None]
    variances = (
        diffusion_jax(states, sigmas) ** 2 * step_dt[:, None, None] + 1e-10
    )
    residual = current[:, None, :] - means
    component_logpdf = -0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * variances) + residual**2 / variances,
        axis=-1,
    )
    return jax.scipy.special.logsumexp(component_logpdf, axis=1) - jnp.log(paths)


def simulated_negative_log_likelihood(
    parameters,
    previous,
    current,
    delta_t,
    normal_draws,
    substeps,
    paths,
    weight_penalty=0.05,
):
    logpdf = _pedersen_transition_logpdf(
        parameters, previous, current, delta_t, normal_draws, substeps, paths
    )
    return -jnp.sum(logpdf) + weight_penalty * jnp.sum(parameters[8:12] ** 2)


class SimulatedMaximumLikelihoodEstimator:
    def __init__(self, substeps=4, paths=64, seed=0):
        if substeps < 1 or paths < 1:
            raise ValueError("substeps and paths must be positive")
        self.substeps = int(substeps)
        self.paths = int(paths)
        self.seed = int(seed)

    def fit(
        self,
        data,
        initial_parameters=None,
        bounds=None,
        max_iterations=200,
        weight_penalty=0.05,
    ):
        initial = (
            DEFAULT_PARAMETERS.copy()
            if initial_parameters is None
            else validate_parameters(initial_parameters)
        )
        parameter_bounds = FULL_BOUNDS if bounds is None else bounds
        rng = np.random.default_rng(self.seed)
        draws = rng.normal(
            size=(
                data.n_transitions,
                self.paths,
                max(self.substeps - 1, 0),
                4,
            )
        )
        previous = jnp.asarray(data.previous)
        current = jnp.asarray(data.current)
        delta_t = jnp.asarray(data.delta_t)
        normal_draws = jnp.asarray(draws)

        def objective(parameters):
            return simulated_negative_log_likelihood(
                parameters,
                previous,
                current,
                delta_t,
                normal_draws,
                self.substeps,
                self.paths,
                weight_penalty,
            )

        value_and_gradient = jax.jit(jax.value_and_grad(objective))

        def scipy_function(parameters):
            value, gradient = value_and_gradient(jnp.asarray(parameters))
            return float(value), np.asarray(gradient, dtype=float)

        result = minimize(
            scipy_function,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=parameter_bounds,
            options={"maxiter": max_iterations, "ftol": 1e-6},
        )
        return CalibrationResult(
            "sml",
            PARAMETER_NAMES,
            result.x,
            bool(result.success),
            float(result.fun),
        )
