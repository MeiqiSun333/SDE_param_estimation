"""Read the small project configuration file."""

import configparser
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default_config.ini"


def _numbers(value):
    return [float(item.strip()) for item in value.split(",")]


def load_config(path=None):
    parser = configparser.ConfigParser()
    config_path = DEFAULT_CONFIG if path is None else Path(path)
    if not parser.read(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    seed_text = parser["simulation"].get("random_seed", "42").strip().lower()
    random_seed = None if seed_text in {"", "none", "null"} else int(seed_text)

    return {
        "dt": parser["simulation"].getfloat("dt", 0.05),
        "end_time": parser["simulation"].getfloat("end_time", 20.0),
        "stochastic": parser["simulation"].getboolean("stochastic", True),
        "random_seed": random_seed,
        "parameters": _numbers(
            parser["model"].get(
                "parameters",
                "0.5,0.5,0.5,0.5,0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1",
            )
        ),
        "figsize": tuple(
            int(value)
            for value in parser["visualization"].get("figsize", "10,6").split(",")
        ),
        "dpi": parser["visualization"].getint("dpi", 150),
    }
