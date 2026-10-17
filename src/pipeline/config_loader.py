"""Utility functions to load configuration files."""

from pathlib import Path
from typing import Any

import yaml


def load_params(config_path: str | Path = "configs/params.yaml") -> dict[str, Any]:
    """Load yaml configuration file with pipeline parameters.

    Args:
        config_path: Path to the yaml configuration file.

    Returns:
        Dictionary containing loaded configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params
