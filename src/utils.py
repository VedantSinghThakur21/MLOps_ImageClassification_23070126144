"""Utility helpers shared across pipeline stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml
from tensorflow.keras.models import load_model as keras_load_model


LOGGER = logging.getLogger(__name__)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML content.
    """

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Persist a dictionary as formatted JSON.

    Args:
        path: Destination file path.
        data: Serializable mapping.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def create_directories(paths: list[Path] | tuple[Path, ...]) -> None:
    """Create a collection of directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, path: Path) -> None:
    """Save a TensorFlow/Keras model to disk.

    Args:
        model: Trained Keras model.
        path: Destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: Path) -> Any:
    """Load a TensorFlow/Keras model from disk."""

    return keras_load_model(path)


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Plot training history metrics and save the figure.

    Args:
        history: Keras training history dictionary.
        output_path: File path to write the chart to.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for metric_name, values in history.items():
        if metric_name.startswith("val_"):
            continue
        plt.plot(values, label=metric_name)
        validation_name = f"val_{metric_name}"
        if validation_name in history:
            plt.plot(history[validation_name], linestyle="--", label=validation_name)

    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
