"""Neural posterior estimation for all twelve model parameters."""

import numpy as np

from src.calibration._common import (
    CalibrationResult,
    FULL_LOWER,
    FULL_UPPER,
    PARAMETER_NAMES,
    summary_vector,
)
from src.model.four_node import simulate_numpy


def _require_sbi():
    try:
        import torch
        from sbi.inference import NPE
        from sbi.utils import BoxUniform
    except ImportError as error:
        raise ImportError("SBI-NPE requires torch and sbi") from error
    return torch, NPE, BoxUniform


class SBINPECalibrator:
    def __init__(self, density_estimator="nsf"):
        self.density_estimator = density_estimator
        self.posterior = None
        self.summary_names = None

    def build_training_set(
        self,
        n_simulations=5000,
        n_trajectories=20,
        n_steps=100,
        delta_t=0.01,
        seed=0,
    ):
        torch, _, BoxUniform = _require_sbi()
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        torch.manual_seed(seed)
        prior = BoxUniform(
            low=torch.as_tensor(FULL_LOWER, dtype=torch.float32),
            high=torch.as_tensor(FULL_UPPER, dtype=torch.float32),
        )
        parameters = prior.sample((n_simulations,)).numpy().astype(np.float32)
        summaries = []

        for index, parameter in enumerate(parameters):
            simulated = simulate_numpy(
                parameter,
                n_trajectories=n_trajectories,
                n_steps=n_steps,
                delta_t=delta_t,
                seed=seed + index + 1,
            )
            vector, names = summary_vector(simulated)
            if self.summary_names is None:
                self.summary_names = names
            summaries.append(vector)

        return (
            torch.as_tensor(parameters),
            torch.as_tensor(np.stack(summaries)),
        )

    def train(self, parameters, summaries, seed=0, training_kwargs=None):
        torch, NPE, BoxUniform = _require_sbi()
        torch.manual_seed(seed)
        prior = BoxUniform(
            low=torch.as_tensor(FULL_LOWER, dtype=torch.float32),
            high=torch.as_tensor(FULL_UPPER, dtype=torch.float32),
        )
        inference = NPE(
            prior=prior,
            density_estimator=self.density_estimator,
            logging_level="ERROR",
            show_progress_bars=False,
        )
        inference.append_simulations(parameters, summaries)
        options = dict(training_kwargs or {})
        options.setdefault("show_train_summary", False)
        density = inference.train(**options)
        self.posterior = inference.build_posterior(density)
        return self.posterior

    def fit(self, observed_trajectories, posterior_samples=10000, seed=0):
        if self.posterior is None:
            raise RuntimeError("train must be called before fit")
        torch, _, _ = _require_sbi()
        vector, names = summary_vector(observed_trajectories)
        if names != self.summary_names:
            raise RuntimeError("Observed and training summaries do not match")
        torch.manual_seed(seed)
        samples = (
            self.posterior.sample(
                (posterior_samples,),
                x=torch.as_tensor(vector, dtype=torch.float32),
                show_progress_bars=False,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return CalibrationResult(
            "sbi_npe",
            PARAMETER_NAMES,
            samples.mean(axis=0),
            True,
            posterior_samples=samples,
        )
