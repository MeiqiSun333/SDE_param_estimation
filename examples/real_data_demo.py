"""Calibrate processed-format data with Milstein."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration._common import CalibrationData
from src.calibration.milstein import MilsteinEstimator

STATE_COLUMNS = ["SAD", "RUM", "DIST", "EMOCOPE"]
DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "example"
    / "synthetic_processed_data.csv"
)


def load_transitions(path):
    frame = pd.read_csv(path)
    required = ["UUID", "time_hours", *STATE_COLUMNS]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    previous, current, intervals = [], [], []
    for _, subject in frame.dropna(subset=required).groupby("UUID"):
        subject = subject.sort_values("time_hours")
        if len(subject) < 2:
            continue
        states = subject[STATE_COLUMNS].to_numpy(dtype=float) / 100.0
        states = np.clip(states, 1e-5, 1.0 - 1e-5)
        delta_t = np.diff(subject["time_hours"].to_numpy(dtype=float))
        valid = delta_t > 0
        previous.append(states[:-1][valid])
        current.append(states[1:][valid])
        intervals.append(delta_t[valid])

    if not previous:
        raise ValueError("No usable transitions found")
    return CalibrationData(
        np.vstack(previous),
        np.vstack(current),
        np.concatenate(intervals),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    result = MilsteinEstimator().fit(load_transitions(args.data))
    for name, estimate in zip(result.parameter_names, result.estimate):
        print(f"{name}: {estimate:.4f}")


if __name__ == "__main__":
    main()
