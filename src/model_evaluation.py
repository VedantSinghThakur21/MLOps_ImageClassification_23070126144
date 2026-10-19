"""Evaluate the trained image classification model."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import load_params, resolve_paths
from .utils import create_directories, load_model, save_json


LOGGER = logging.getLogger(__name__)


def _load_test_generator(processed_path: Path, params: dict[str, object]):
    """Create the test generator."""

    image_size = tuple(int(value) for value in params["image_size"])
    batch_size = int(params["batch_size"])
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        processed_path / "test",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def _plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """Plot and save a confusion matrix figure."""

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=45, ha="right")
    plt.yticks(tick_positions, class_names)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index, row_values in enumerate(matrix):
        for col_index, value in enumerate(row_values):
            plt.text(
                col_index,
                row_index,
                f"{int(value)}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Evaluate the model on the test split and persist metrics."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    paths = resolve_paths(params)
    create_directories([paths.metrics_path])

    model_path = paths.model_path / f"{params['model_name']}.keras"
    model = load_model(model_path)
    test_generator = _load_test_generator(paths.processed_path, params)

    probabilities = model.predict(test_generator, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    truth = test_generator.classes

    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(truth, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
    }
    save_json(paths.metrics_path / "metrics.json", metrics)

    matrix = confusion_matrix(truth, predictions)
    class_names = list(test_generator.class_indices.keys())
    _plot_confusion_matrix(matrix, class_names, paths.metrics_path / "confusion_matrix.png")
    LOGGER.info("Evaluation metrics saved to %s", paths.metrics_path)


if __name__ == "__main__":
    main()
