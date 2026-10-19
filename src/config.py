"""Configuration helpers for the image classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import read_yaml


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths used across pipeline stages."""

    root: Path
    dataset_path: Path
    processed_path: Path
    artifacts_path: Path
    processed_artifacts_path: Path
    features_path: Path
    model_path: Path
    metrics_path: Path


def load_params() -> dict[str, Any]:
    """Load params.yaml from the project root.

    Returns:
        Parsed parameter values.
    """

    return read_yaml(Path("params.yaml"))


def resolve_paths(params: dict[str, Any]) -> ProjectPaths:
    """Resolve configured paths relative to the project root.

    Args:
        params: Parsed project parameters.

    Returns:
        Resolved project path collection.
    """

    root = Path(".").resolve()
    return ProjectPaths(
        root=root,
        dataset_path=root / params["dataset_path"],
        processed_path=root / params["processed_path"],
        artifacts_path=root / params["artifacts_path"],
        processed_artifacts_path=root / params["processed_artifacts_path"],
        features_path=root / params["features_path"],
        model_path=root / params["model_path"],
        metrics_path=root / params["metrics_path"],
    )
