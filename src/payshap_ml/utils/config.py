"""Shared configuration loading utilities.

All pipeline stages and the benchmark harness read their configuration
from a single params.yaml file so that DVC can track parameter changes
and invalidate the correct downstream stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = "params.yaml"


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load and return the full params.yaml contents as a dict.

    Raises FileNotFoundError with a clear message if the file is missing,
    since every pipeline stage depends on it being present.
    """
    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(
            f"Could not find params file at '{params_path}'. "
            "Run pipeline stages from the repository root."
        )
    with params_path.open("r") as f:
        return yaml.safe_load(f)


def get_params_arg_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser with the shared --params flag.

    Every pipeline entrypoint accepts --params so a caller (or DVC stage)
    can point at an alternate config, e.g. for a parameter sweep.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--params",
        type=str,
        default=DEFAULT_PARAMS_PATH,
        help="Path to params.yaml (default: %(default)s)",
    )
    return parser


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for a given output path if missing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
