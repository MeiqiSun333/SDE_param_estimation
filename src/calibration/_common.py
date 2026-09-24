"""Small shared helpers used by the seven calibration methods."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

PARAMETER_NAMES = (
    "beta1", "beta2", "beta3", "beta4",
    "sigma1", "sigma2", "sigma3", "sigma4",
    "w21", "w31", "w42", "w43",
)
BETA_NAMES = PARAMETER_NAMES[:4]

FULL_LOWER = np.array([0.001] * 4 + [0.01] * 4 + [-5.0] * 4)
FULL_UPPER = np.array([5.0] * 12)
BETA_LOWER = np.array([0.01] * 4)
BETA_UPPER = np.array([3.0] * 4)
FULL_BOUNDS = list(zip(FULL_LOWER, FULL_UPPER))
DEFAULT_PARAMETERS = np.array([0.5] * 4 + [0.2] * 4 + [0.1] * 4)


@dataclass
class CalibrationData:
    """Observed transitions used by the likelihood-based methods."""

    previous: np.ndarray
    current: np.ndarray
    delta_t: np.ndarray

    def __post_init__(self):
        self.previous = np.asarray(self.previous, dtype=float)
        self.current = np.asarray(self.current, dtype=float)
        self.delta_t = np.asarray(self.delta_t, dtype=float)
        if self.previous.ndim != 2 or self.previous.shape[1] != 4:
            raise ValueError("previous must have shape (n_transitions, 4)")
        if self.current.shape != self.previous.shape:
            raise ValueError("current must have the same shape as previous")
        if self.delta_t.shape != (len(self.previous),):
            raise ValueError("delta_t must contain one value per transition")
        if len(self.previous) == 0:
            raise ValueError("At least one transition is required")
        if not all(
            np.all(np.isfinite(values))
            for values in (self.previous, self.current, self.delta_t)
        ):
            raise ValueError("Calibration data must be finite")
        if np.any(self.delta_t <= 0):
            raise ValueError("All time intervals must be positive")
        if np.any(self.previous <= 0) or np.any(self.previous >= 1):
            raise ValueError("Previous states must lie strictly inside (0, 1)")
        if np.any(self.current <= 0) or np.any(self.current >= 1):
            raise ValueError("Current states must lie strictly inside (0, 1)")

    @property
    def n_transitions(self):
        return len(self.previous)

    @classmethod
    def from_trajectories(cls, trajectories, delta_t):
        values = np.asarray(trajectories, dtype=float)
        if values.ndim != 3 or values.shape[2] != 4 or values.shape[1] < 2:
            raise ValueError("trajectories must have shape (n, time, 4)")
        n_trajectories, n_steps, _ = values.shape
        schedule = np.asarray(delta_t, dtype=float)
        if schedule.ndim == 0:
            schedule = np.full(n_steps - 1, float(schedule))
        if schedule.shape != (n_steps - 1,):
            raise ValueError("delta_t schedule must have length n_steps - 1")
        return cls(
            values[:, :-1].reshape(-1, 4),
            values[:, 1:].reshape(-1, 4),
            np.tile(schedule, n_trajectories),
        )


@dataclass
class CalibrationResult:
    """Result returned by every calibration method."""

    method: str
    parameter_names: tuple[str, ...]
    estimate: np.ndarray
    success: bool
    objective: float | None = None
    posterior_samples: np.ndarray | None = None

    def __post_init__(self):
        self.estimate = np.asarray(self.estimate, dtype=float)
        if self.posterior_samples is not None:
            self.posterior_samples = np.asarray(self.posterior_samples, dtype=float)


def validate_parameters(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain twelve finite values")
    return values


def replace_betas(parameters, betas):
    parameters = validate_parameters(parameters).copy()
    betas = np.asarray(betas, dtype=float)
    if betas.shape != (4,):
        raise ValueError("betas must contain four values")
    parameters[:4] = betas
    return parameters


def fit_likelihood(
    method,
    logpdf,
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
    previous = jnp.asarray(data.previous)
    current = jnp.asarray(data.current)
    delta_t = jnp.asarray(data.delta_t)
    batch_logpdf = jax.vmap(logpdf, in_axes=(None, 0, 0, 0))

    def objective(parameters):
        likelihood = batch_logpdf(parameters, previous, current, delta_t)
        penalty = weight_penalty * jnp.sum(parameters[8:12] ** 2)
        return -jnp.sum(likelihood) + penalty

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
        method,
        PARAMETER_NAMES,
        result.x,
        bool(result.success),
        float(result.fun),
    )


def summary_statistics(
    trajectories,
    lags=(1, 3, 10),
    pairs=((0, 1), (0, 2), (1, 3), (2, 3)),
):
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("trajectories must have shape (n, time, 4)")
    if max(lags) >= values.shape[1]:
        raise ValueError("Every lag must be shorter than the trajectory")

    centered = values - values.mean(axis=1, keepdims=True)
    variances_per_trajectory = (centered**2).mean(axis=1) + 1e-12
    means = values.mean(axis=(0, 1))
    variances = values.var(axis=(0, 1))
    increments = np.diff(values, axis=1)
    half = values.shape[1] // 2
    shifts = (
        values[:, half:].mean(axis=(0, 1))
        - values[:, :half].mean(axis=(0, 1))
    )

    output = {}
    for node in range(4):
        output[f"mean_{node}"] = float(means[node])
        output[f"var_{node}"] = float(variances[node])
        output[f"incr_abs_{node}"] = float(np.abs(increments[..., node]).mean())
        output[f"drift_{node}"] = float(increments[..., node].mean())
        output[f"shift_{node}"] = float(shifts[node])

    for lag in lags:
        numerator = (centered[:, :-lag] * centered[:, lag:]).mean(axis=1)
        autocorrelation = (numerator / variances_per_trajectory).mean(axis=0)
        for node in range(4):
            output[f"ac{lag}_{node}"] = float(autocorrelation[node])
        for parent, child in pairs:
            numerator = (
                centered[:, :-lag, parent] * centered[:, lag:, child]
            ).mean(axis=1)
            denominator = np.sqrt(
                variances_per_trajectory[:, parent]
                * variances_per_trajectory[:, child]
            )
            output[f"xc{lag}_{parent}{child}"] = float(
                (numerator / denominator).mean()
            )
    return output


def summary_vector(trajectories):
    statistics = summary_statistics(trajectories)
    names = tuple(sorted(statistics))
    vector = np.array([statistics[name] for name in names], dtype=np.float32)
    return vector, names
