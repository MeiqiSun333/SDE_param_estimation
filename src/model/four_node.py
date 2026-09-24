"""Canonical equations and topology for the four-node emotion network.

Both the user-facing simulator and calibration code import from this module so
the normalized logistic function, network topology, and SDE terms cannot drift
apart.
"""

import numpy as np

STATE_NAMES = ("SAD", "RUM", "DIST", "EMOCOPE")
FOUR_NODE_INPUT_INDICES = ((), (0,), (0,), (1, 2))
MU_FIXED = 0.5
ALPHAS_FIXED = np.array([5.0, 5.0, 5.0])
TAUS_FIXED = np.array([0.3, 0.5, 0.4])


def alogistic_numpy(impact, alpha, tau):
    """Normalized logistic response with a zero-valued response at zero input."""
    impact = np.asarray(impact)
    numerator = 1.0 / (1.0 + np.exp(-alpha * (impact - tau)))
    denominator = 1.0 / (1.0 + np.exp(alpha * tau))
    return (numerator - denominator) * (1.0 + np.exp(-alpha * tau))


def targets_numpy(states, weights):
    """Return the four target states for one state or a batch of states."""
    states = np.asarray(states, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if states.shape[-1] != 4 or weights.shape != (4,):
        raise ValueError("Expected states (..., 4) and four weights")
    y1, y2, y3 = states[..., 0], states[..., 1], states[..., 2]
    return np.stack(
        [
            np.full_like(y1, MU_FIXED),
            alogistic_numpy(weights[0] * y1, ALPHAS_FIXED[0], TAUS_FIXED[0]),
            alogistic_numpy(weights[1] * y1, ALPHAS_FIXED[1], TAUS_FIXED[1]),
            alogistic_numpy(
                weights[2] * y2 + weights[3] * y3,
                ALPHAS_FIXED[2],
                TAUS_FIXED[2],
            ),
        ],
        axis=-1,
    )


def drift_numpy(states, betas, weights):
    states = np.asarray(states, dtype=float)
    return drift_from_targets_numpy(states, targets_numpy(states, weights), betas)


def drift_from_targets_numpy(states, targets, betas):
    """Return the relaxation drift for already-computed target states."""
    states = np.asarray(states, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return np.asarray(betas, dtype=float) * (targets - states)


def diffusion_numpy(states, sigmas):
    states = np.asarray(states, dtype=float)
    return np.asarray(sigmas) * states * (1.0 - states)


def alogistic_jax(impact, alpha, tau):
    """JAX form of the same canonical normalized logistic response."""
    import jax.numpy as jnp

    numerator = 1.0 / (1.0 + jnp.exp(-alpha * (impact - tau)))
    denominator = 1.0 / (1.0 + jnp.exp(alpha * tau))
    return (numerator - denominator) * (1.0 + jnp.exp(-alpha * tau))


def targets_jax(states, weights):
    """JAX form of the canonical four target states."""
    import jax.numpy as jnp

    y1, y2, y3 = states[..., 0], states[..., 1], states[..., 2]
    return jnp.stack(
        [
            jnp.full_like(y1, MU_FIXED),
            alogistic_jax(weights[0] * y1, ALPHAS_FIXED[0], TAUS_FIXED[0]),
            alogistic_jax(weights[1] * y1, ALPHAS_FIXED[1], TAUS_FIXED[1]),
            alogistic_jax(
                weights[2] * y2 + weights[3] * y3,
                ALPHAS_FIXED[2],
                TAUS_FIXED[2],
            ),
        ],
        axis=-1,
    )


def drift_jax(states, betas, weights):
    return drift_from_targets_jax(states, targets_jax(states, weights), betas)


def drift_from_targets_jax(states, targets, betas):
    """JAX form of the relaxation drift for precomputed targets."""
    return betas * (targets - states)


def diffusion_jax(states, sigmas):
    return sigmas * states * (1.0 - states)


def simulate_numpy(
    parameters,
    n_trajectories=20,
    n_steps=100,
    delta_t=0.01,
    initial_states=None,
    seed=None,
    stochastic=True,
):
    """Simulate batches of the canonical four-node model."""
    params = np.asarray(parameters, dtype=float)
    if params.shape != (12,):
        raise ValueError("The simulator requires twelve parameters")
    if n_trajectories <= 0 or n_steps < 2:
        raise ValueError("n_trajectories must be positive and n_steps at least two")
    schedule = np.asarray(delta_t, dtype=float)
    if schedule.ndim == 0:
        schedule = np.full(n_steps - 1, float(schedule))
    if schedule.shape != (n_steps - 1,) or np.any(schedule <= 0):
        raise ValueError("delta_t must be positive with length n_steps - 1")

    rng = np.random.default_rng(seed)
    paths = np.empty((n_trajectories, n_steps, 4), dtype=float)
    if initial_states is None:
        paths[:, 0] = rng.uniform(0.1, 0.9, size=(n_trajectories, 4))
    else:
        initial = np.asarray(initial_states, dtype=float)
        if initial.shape == (4,):
            initial = np.broadcast_to(initial, (n_trajectories, 4))
        if initial.shape != (n_trajectories, 4):
            raise ValueError("initial_states must have shape (4,) or (n, 4)")
        paths[:, 0] = initial

    betas, sigmas, weights = params[:4], params[4:8], params[8:12]
    for step, dt in enumerate(schedule):
        state = paths[:, step]
        increment = drift_numpy(state, betas, weights) * dt
        if stochastic:
            increment += diffusion_numpy(state, sigmas) * np.sqrt(dt) * rng.normal(
                size=state.shape
            )
        paths[:, step + 1] = np.clip(state + increment, 1e-5, 1.0 - 1e-5)
    return paths
