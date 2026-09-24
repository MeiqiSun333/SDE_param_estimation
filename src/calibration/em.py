"""Euler-Maruyama Gaussian transition likelihood."""

import jax.numpy as jnp

from src.calibration._common import fit_likelihood
from src.model.four_node import diffusion_jax, drift_jax


def euler_maruyama_logpdf(parameters, previous, current, delta_t):
    betas = parameters[:4]
    sigmas = parameters[4:8]
    weights = parameters[8:12]
    drift = drift_jax(previous, betas, weights)
    diffusion = diffusion_jax(previous, sigmas)
    variance = diffusion**2 * delta_t + 1e-10
    residual = current - previous - drift * delta_t
    return -0.5 * jnp.sum(
        jnp.log(2 * jnp.pi) + jnp.log(variance) + residual**2 / variance
    )


class EulerMaruyamaEstimator:
    def fit(self, data, **kwargs):
        return fit_likelihood(
            "euler_maruyama", euler_maruyama_logpdf, data, **kwargs
        )
