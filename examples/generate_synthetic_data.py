"""Generate an artificial processed-like longitudinal emotion dataset.

This generator does not use the four-node SDE from src/model. It combines
person-specific baselines, irregular observation times, circadian variation,
cross-lag effects, and correlated noise. The output is useful for demonstrating
the data-loading and calibration pipeline, not for parameter-recovery tests.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = ("SAD", "STR", "SITMOD", "DIST", "REAP", "RUM", "EMOCOPE")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "example"
    / "synthetic_processed_data.csv"
)


def _logit(probability):
    return np.log(probability / (1.0 - probability))


def _scores(latent, rng):
    values = 100.0 / (1.0 + np.exp(-latent))
    values += rng.normal(0.0, 2.5, size=len(VARIABLES))
    return np.round(np.clip(values, 0.0, 100.0), 1)


def generate_synthetic_processed_data(
    n_subjects=40,
    min_observations=48,
    max_observations=72,
    seed=2026,
):
    """Return a reproducible processed-like dataset with no real records."""
    if n_subjects < 1:
        raise ValueError("n_subjects must be positive")
    if min_observations < 2 or max_observations < min_observations:
        raise ValueError("observation limits must satisfy 2 <= min <= max")

    rng = np.random.default_rng(seed)
    population_scores = np.array([35, 42, 47, 51, 56, 38, 59]) / 100.0
    population_baseline = _logit(population_scores)

    distress_loading = np.array([1.0, 0.9, 0.35, -0.25, -0.30, 0.85, -0.60])
    regulation_loading = np.array([-0.20, -0.15, 0.20, 0.60, 0.65, -0.20, 0.70])
    circadian_loading = np.array([0.10, 0.08, 0.03, -0.02, 0.02, 0.08, -0.04])
    rows = []

    for subject_index in range(1, n_subjects + 1):
        n_observations = int(
            rng.integers(min_observations, max_observations + 1)
        )
        intervals = np.clip(
            rng.lognormal(mean=np.log(3.2), sigma=0.45, size=n_observations - 1),
            0.75,
            9.0,
        )
        times = np.concatenate(([0.0], np.cumsum(intervals)))

        person_distress = rng.normal(0.0, 0.42)
        person_regulation = rng.normal(0.0, 0.36)
        baseline = (
            population_baseline
            + person_distress * distress_loading
            + person_regulation * regulation_loading
            + rng.normal(0.0, 0.16, size=len(VARIABLES))
        )
        latent = baseline + rng.normal(0.0, 0.12, size=len(VARIABLES))

        for observation_index, time_hours in enumerate(times):
            if observation_index:
                step = min(intervals[observation_index - 1] / 3.2, 2.5)
                centered = latent - baseline
                cross_lag = np.array(
                    [
                        0.06 * centered[5] - 0.05 * centered[4],
                        0.07 * centered[0],
                        0.05 * centered[1],
                        0.05 * centered[0] - 0.04 * centered[5],
                        0.04 * centered[2],
                        0.08 * centered[0],
                        (
                            0.06 * centered[3]
                            + 0.07 * centered[4]
                            - 0.08 * centered[0]
                            - 0.07 * centered[5]
                        ),
                    ]
                )
                circadian = np.sin(
                    2.0 * np.pi * ((time_hours % 24.0) - 15.0) / 24.0
                )
                distress_shock = rng.normal()
                regulation_shock = rng.normal()
                innovation = (
                    rng.normal(0.0, 0.10, size=len(VARIABLES))
                    + 0.12 * distress_shock * distress_loading
                    + 0.10 * regulation_shock * regulation_loading
                )
                latent += (
                    step
                    * (
                        0.20 * (baseline - latent)
                        + cross_lag
                        + circadian * circadian_loading
                    )
                    + np.sqrt(step) * innovation
                )
                latent = np.clip(latent, -3.5, 3.5)

            row = {
                "UUID": f"SYNTH_{subject_index:04d}",
                **dict(zip(VARIABLES, _scores(latent, rng))),
                "time_hours": round(float(time_hours), 3),
            }
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["delta_t"] = frame.groupby("UUID")["time_hours"].diff().round(3)
    return frame[["UUID", *VARIABLES, "time_hours", "delta_t"]]


def main():
    parser = argparse.ArgumentParser(
        description="Generate artificial processed-like longitudinal data"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", type=int, default=40)
    parser.add_argument("--min-observations", type=int, default=48)
    parser.add_argument("--max-observations", type=int, default=72)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    frame = generate_synthetic_processed_data(
        n_subjects=args.subjects,
        min_observations=args.min_observations,
        max_observations=args.max_observations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"Wrote {len(frame)} artificial observations "
        f"for {frame['UUID'].nunique()} subjects to {args.output}"
    )


if __name__ == "__main__":
    main()
