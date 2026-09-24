"""Plots used by the model and calibration examples."""

import matplotlib.pyplot as plt
import numpy as np

from src.model.four_node import STATE_NAMES


def plot_trajectories(time, states, title="Simulated emotion network"):
    """Plot the four state trajectories."""
    values = np.asarray(states)
    if values.ndim == 3:
        values = values[0].T
    if values.shape[0] != 4:
        raise ValueError("states must have shape (4, time) or (n, time, 4)")

    figure, axis = plt.subplots(figsize=(10, 5))
    for values_for_state, name in zip(values, STATE_NAMES):
        axis.plot(time, values_for_state, label=name)
    axis.set(xlabel="Time", ylabel="State", title=title, ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_parameter_comparison(
    true_parameters,
    estimates,
    parameter_names,
    title="Calibration results",
):
    """Compare true parameters with estimates from one or more methods."""
    truth = np.asarray(true_parameters, dtype=float)
    names = tuple(parameter_names)
    x = np.arange(len(names))
    width = 0.8 / (len(estimates) + 1)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(x - 0.4 + width / 2, truth[: len(names)], width, label="True")
    for index, (method, estimate) in enumerate(estimates.items(), start=1):
        values = np.asarray(estimate, dtype=float)
        axis.bar(
            x - 0.4 + width * (index + 0.5),
            values[: len(names)],
            width,
            label=method,
        )
    axis.set_xticks(x, names, rotation=45, ha="right")
    axis.set(title=title, ylabel="Parameter value")
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_posterior(samples, parameter_names, title="Posterior samples"):
    """Plot marginal posterior histograms."""
    values = np.asarray(samples, dtype=float)
    names = tuple(parameter_names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("samples and parameter_names do not match")

    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 2.7 * rows))
    axes = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes):
        if index < len(names):
            axis.hist(values[:, index], bins=25)
            axis.set_title(names[index])
        else:
            axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes
