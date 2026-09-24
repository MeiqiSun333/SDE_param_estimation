"""Ait-Sahalia transition-density approximation."""

import jax
import jax.numpy as jnp

from src.calibration._common import fit_likelihood
from src.model.four_node import targets_jax


def transformed_drift(z, parameters):
    betas = parameters[:4]
    sigmas = parameters[4:8]
    weights = parameters[8:12]
    states = 1.0 / (1.0 + jnp.exp(-sigmas * z))
    targets = targets_jax(states, weights)
    return (
        betas * (targets - states) / (sigmas * states * (1.0 - states))
        - sigmas * (1.0 - 2.0 * states) / 2.0
    )


def c0(z, z0, parameters):
    def integrand(u):
        point = z0 + u * (z - z0)
        return jnp.dot(transformed_drift(point, parameters), z - z0)

    grid = jnp.linspace(0.0, 1.0, 10)
    return jnp.trapezoid(jax.vmap(integrand)(grid), grid)


def c1(z, z0, parameters):
    gradient_c0 = jax.grad(c0, argnums=0)

    def correction(point):
        divergence = jnp.trace(
            jax.jacfwd(transformed_drift, argnums=0)(point, parameters)
        )
        gradient = gradient_c0(point, z0, parameters)
        laplacian = jnp.trace(
            jax.hessian(c0, argnums=0)(point, z0, parameters)
        )
        drift = transformed_drift(point, parameters)
        return (
            -divergence
            - jnp.dot(gradient, drift)
            + 0.5 * laplacian
            + 0.5 * jnp.sum(gradient**2)
        )

    grid = jnp.linspace(0.0, 1.0, 10)
    values = jax.vmap(lambda u: correction(z0 + u * (z - z0)))(grid)
    return jnp.trapezoid(values, grid)


def ait_sahalia_logpdf(parameters, previous, current, delta_t):
    sigmas = parameters[4:8]
    z_previous = jnp.log(previous / (1.0 - previous)) / sigmas
    z_current = jnp.log(current / (1.0 - current)) / sigmas
    delta_z = z_current - z_previous
    log_density = (
        -2.0 * jnp.log(2.0 * jnp.pi * delta_t)
        - 0.5 * jnp.sum(delta_z**2) / delta_t
        + c0(z_current, z_previous, parameters)
        + c1(z_current, z_previous, parameters) * delta_t
    )
    log_jacobian = jnp.sum(
        -jnp.log(sigmas) - jnp.log(current) - jnp.log(1.0 - current)
    )
    return log_density + log_jacobian


class AitSahaliaEstimator:
    def fit(self, data, **kwargs):
        return fit_likelihood(
            "ait_sahalia", ait_sahalia_logpdf, data, **kwargs
        )
