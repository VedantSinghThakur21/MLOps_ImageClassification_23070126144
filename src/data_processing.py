"""Process raw images into train, validation, and test datasets."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from .config import load_params, resolve_paths
from .utils import create_directories, save_json


LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_dataset_root(raw_path: Path) -> Path:
    """Find the directory containing class subfolders."""

    if any((raw_path / class_name).is_dir() for class_name in ("buildings", "forest", "glacier", "mountain", "sea", "street")):
        return raw_path

    for child in raw_path.iterdir():
        if child.is_dir() and any(item.is_dir() for item in child.iterdir()):
            return child

    return raw_path


def _collect_image_paths(dataset_root: Path) -> list[tuple[Path, str]]:
    """Collect image files and derive labels from parent directories."""

    items: list[tuple[Path, str]] = []
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((path, path.parent.name))
    return items


def _split_items(items: list[tuple[Path, str]], validation_split: float, test_split: float, seed: int) -> dict[str, list[tuple[Path, str]]]:
    """Split items into train, validation, and test subsets."""

    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    shuffled = [items[index] for index in indices]
    test_size = max(1, int(len(shuffled) * test_split))
    val_size = max(1, int(len(shuffled) * validation_split))
    test_items = shuffled[:test_size]
    val_items = shuffled[test_size:test_size + val_size]
    train_items = shuffled[test_size + val_size:]
    if not train_items:
        train_items = shuffled
    return {"train": train_items, "validation": val_items, "test": test_items}


def _write_split(split_items: list[tuple[Path, str]], split_dir: Path, image_size: tuple[int, int]) -> None:
    """Resize and copy images into a split directory."""

    for image_path, class_name in split_items:
        destination = split_dir / class_name
        destination.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Skipping unreadable image: %s", image_path)
            continue
        resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(destination / image_path.name), resized)


def _build_generators(processed_path: Path, params: dict[str, object]) -> dict[str, ImageDataGenerator]:
    """Create Keras generators for augmentation metadata and reproducibility."""

    image_size = tuple(params["image_size"])  # type: ignore[arg-type]
    batch_size = int(params["batch_size"])
    seed = int(params["seed"])
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=int(params["rotation_range"]),
        zoom_range=float(params["zoom_range"]),
        horizontal_flip=bool(params["horizontal_flip"]),
        brightness_range=tuple(params["brightness_range"]),  # type: ignore[arg-type]
    )
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
    test_generator = eval_datagen.flow_from_directory(
        processed_path / "test",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return {
        "train_generator": train_generator,
        "validation_generator": validation_generator,
        "test_generator": test_generator,
    }


def main() -> None:
    """Process the raw dataset into structured train/validation/test splits."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    paths = resolve_paths(params)
    create_directories([paths.processed_path, paths.processed_artifacts_path])

    dataset_root = _find_dataset_root(paths.dataset_path)
    items = _collect_image_paths(dataset_root)
    if not items:
        raise FileNotFoundError(f"No images found under {dataset_root}")

    splits = _split_items(items, float(params["validation_split"]), float(params["test_split"]), int(params["seed"]))
    image_size = tuple(int(value) for value in params["image_size"])  # type: ignore[arg-type]

    for split_name in ("train", "validation", "test"):
        split_dir = paths.processed_path / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_split(splits[split_name], split_dir, image_size)

    generators = _build_generators(paths.processed_path, params)
    metadata = {
        "train_samples": int(generators["train_generator"].samples),
        "validation_samples": int(generators["validation_generator"].samples),
        "test_samples": int(generators["test_generator"].samples),
        "class_indices": generators["train_generator"].class_indices,
    }
    save_json(paths.processed_artifacts_path / "dataset_metadata.json", metadata)
    LOGGER.info("Processed dataset saved to %s", paths.processed_path)


if __name__ == "__main__":
    main()
