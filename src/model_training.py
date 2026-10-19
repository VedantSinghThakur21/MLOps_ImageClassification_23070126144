"""Train the image classification model."""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import load_params, resolve_paths
from .utils import create_directories, plot_history, save_json, save_model


LOGGER = logging.getLogger(__name__)


def _build_classification_model(feature_model_path: Path, dense_units: int, dropout_rate: float, num_classes: int, learning_rate: float, optimizer_name: str, loss_function: str, metrics: list[str]) -> Model:
    """Build and compile the classification head on top of the feature extractor."""

    feature_extractor = load_model(feature_model_path)
    feature_extractor.trainable = False

    input_shape = tuple(feature_extractor.input_shape[1:])
    inputs = Input(shape=input_shape, name="image_input")
    x = feature_extractor(inputs, training=False)
    x = Dense(dense_units, activation="relu", name="dense_layer")(x)
    x = Dropout(dropout_rate, name="dropout_layer")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs, name="mobilenetv2_classifier")

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.get(optimizer_name)
        if hasattr(optimizer, "learning_rate"):
            optimizer.learning_rate = learning_rate
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    return model


def _build_generators(processed_path: Path, params: dict[str, object]) -> tuple[object, object]:
    """Create train and validation generators from processed images."""

    image_size = tuple(int(value) for value in params["image_size"])
    batch_size = int(params["batch_size"])
    seed = int(params["seed"])

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        processed_path / "train",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )
    validation_generator = eval_datagen.flow_from_directory(
        processed_path / "validation",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, validation_generator


def main() -> None:
    """Train the classification model and persist artifacts."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    paths = resolve_paths(params)
    create_directories([paths.model_path])

    feature_model_path = paths.features_path / f"{params['model_name']}_feature_extractor.keras"
    train_generator, validation_generator = _build_generators(paths.processed_path, params)
    model = _build_classification_model(
        feature_model_path=feature_model_path,
        dense_units=int(params["dense_units"]),
        dropout_rate=float(params["dropout_rate"]),
        num_classes=train_generator.num_classes,
        learning_rate=float(params["learning_rate"]),
        optimizer_name=str(params["optimizer"]),
        loss_function=str(params["loss_function"]),
        metrics=list(params["metrics"]),
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=paths.model_path / f"{params['model_name']}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=int(params["epochs"]),
        callbacks=callbacks,
        verbose=1,
    )

    save_model(model, paths.model_path / f"{params['model_name']}_final.keras")
    save_json(paths.model_path / "training_history.json", history.history)
    plot_history(history.history, paths.model_path / "training_history.png")
    LOGGER.info("Model training complete. Artifacts saved to %s", paths.model_path)


if __name__ == "__main__":
    main()
