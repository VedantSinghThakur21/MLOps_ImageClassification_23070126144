"""Build and export the MobileNetV2 feature extractor."""

from __future__ import annotations

import logging
import ssl
from pathlib import Path

import certifi
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input

from .config import load_params, resolve_paths
from .utils import create_directories, save_model


LOGGER = logging.getLogger(__name__)


ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


def build_feature_extractor(image_size: tuple[int, int], freeze_base_model: bool, fine_tune_at: int) -> Model:
    """Create a MobileNetV2-based feature extractor model.

    Args:
        image_size: Input image size.
        freeze_base_model: Whether to freeze the base convolutional layers.
        fine_tune_at: Layer index from which fine-tuning may start later.

    Returns:
        Keras model with pooled feature output.
    """

    inputs = Input(shape=(*image_size, 3), name="image_input")
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base_model.trainable = not freeze_base_model
    if not freeze_base_model:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    return Model(inputs=inputs, outputs=x, name="mobilenetv2_feature_extractor")


def main() -> None:
    """Create and persist the feature extraction model."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    paths = resolve_paths(params)
    create_directories([paths.features_path])

    feature_model = build_feature_extractor(
        image_size=tuple(int(value) for value in params["image_size"]),
        freeze_base_model=bool(params["freeze_base_model"]),
        fine_tune_at=int(params["fine_tune_at"]),
    )
    save_model(feature_model, paths.features_path / f"{params['model_name']}_feature_extractor.keras")
    LOGGER.info("Feature extractor saved to %s", paths.features_path)


if __name__ == "__main__":
    main()
