"""Milstein pseudo-Gaussian transition likelihood."""

import jax.numpy as jnp

from src.calibration._common import fit_likelihood
from src.model.four_node import diffusion_jax, drift_jax


def milstein_logpdf(parameters, previous, current, delta_t):
    betas = parameters[:4]
    sigmas = parameters[4:8]
    weights = parameters[8:12]
    drift = drift_jax(previous, betas, weights)
    diffusion = diffusion_jax(previous, sigmas)
    diffusion_prime = sigmas * (1.0 - 2.0 * previous)
    mean = previous + drift * delta_t
    variance = (
        diffusion**2 * delta_t
        + 0.5 * (diffusion * diffusion_prime) ** 2 * delta_t**2
        + 1e-10
    )
    residual = current - mean
    return -0.5 * jnp.sum(
        jnp.log(2 * jnp.pi) + jnp.log(variance) + residual**2 / variance
    )


class MilsteinEstimator:
    def fit(self, data, **kwargs):
        return fit_likelihood("milstein", milstein_logpdf, data, **kwargs)
