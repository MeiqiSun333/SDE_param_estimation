"""Simulate data, calibrate it with Milstein and SBI, and plot the results."""

import matplotlib.pyplot as plt
import numpy as np

from src.calibration._common import CalibrationData, PARAMETER_NAMES
from src.calibration.milstein import MilsteinEstimator
from src.calibration.sbi_npe import SBINPECalibrator
from src.model.four_node import simulate_numpy
from src.visualization import (
    plot_parameter_comparison,
    plot_posterior,
    plot_trajectories,
)

TRUE_PARAMETERS = np.array(
    [0.4, 0.6, 0.3, 0.5, 0.15, 0.20, 0.18, 0.25, 0.8, 0.5, -0.4, 0.7]
)
DELTA_T = 0.05
N_TRAJECTORIES = 12
N_STEPS = 80


def main():
    trajectories = simulate_numpy(
        TRUE_PARAMETERS,
        n_trajectories=N_TRAJECTORIES,
        n_steps=N_STEPS,
        delta_t=DELTA_T,
        seed=42,
    )
    data = CalibrationData.from_trajectories(trajectories, DELTA_T)

    milstein = MilsteinEstimator().fit(data, max_iterations=150)

    sbi = SBINPECalibrator()
    parameters, summaries = sbi.build_training_set(
        n_simulations=300,
        n_trajectories=8,
        n_steps=N_STEPS,
        delta_t=DELTA_T,
        seed=42,
    )
    sbi.train(
        parameters,
        summaries,
        seed=42,
        training_kwargs={
            "training_batch_size": 50,
            "max_num_epochs": 50,
            "stop_after_epochs": 10,
        },
    )
    sbi_result = sbi.fit(trajectories, posterior_samples=2000, seed=42)

    time = np.arange(N_STEPS) * DELTA_T
    plot_trajectories(time, trajectories, title="Synthetic four-node data")
    plot_parameter_comparison(
        TRUE_PARAMETERS,
        {
            "Milstein": milstein.estimate,
            "SBI": sbi_result.estimate,
        },
        PARAMETER_NAMES,
    )
    plot_posterior(
        sbi_result.posterior_samples,
        PARAMETER_NAMES,
        title="SBI posterior",
    )
    plt.show()


if __name__ == "__main__":
    main()
