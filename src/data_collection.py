"""Download or collect the Intel image classification dataset."""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .config import load_params, resolve_paths
from .utils import create_directories


LOGGER = logging.getLogger(__name__)


def _dataset_already_present(dataset_path: Path) -> bool:
    """Check whether the raw dataset is already available locally."""

    return dataset_path.exists() and any(dataset_path.rglob("*.jpg"))


def _try_kaggle_download(target_path: Path, dataset_url: str) -> bool:
    """Download a public Kaggle dataset through the Kaggle download API.

    Args:
        target_path: Destination directory for the extracted dataset.
        dataset_url: Kaggle dataset URL.

    Returns:
        True if the download and extraction succeeded, otherwise False.
    """

    url_parts = dataset_url.rstrip("/").split("/")
    dataset_slug = "/".join(url_parts[-2:]) if len(url_parts) >= 2 else dataset_url
    archive_path = target_path / "dataset.zip"

    try:
        urllib.request.urlretrieve(f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}", archive_path)
        with ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(target_path)
        archive_path.unlink(missing_ok=True)
        return True
    except (OSError, BadZipFile) as exc:
        LOGGER.warning("Kaggle download unavailable: %s", exc)
        return False


def _copy_from_local_source(source_path: Path, destination_path: Path) -> bool:
    """Copy a locally provided dataset into the raw data directory."""

    if not source_path.exists():
        return False

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with ZipFile(source_path, "r") as zip_file:
            zip_file.extractall(destination_path)
        return True

    if source_path.is_dir():
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        return True

    return False


def main() -> None:
    """Collect the dataset and store it inside `data/raw`."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    paths = resolve_paths(params)
    create_directories([paths.dataset_path, paths.artifacts_path])

    if _dataset_already_present(paths.dataset_path):
        LOGGER.info("Raw dataset already present at %s", paths.dataset_path)
        return

    dataset_url = str(params.get("dataset_url", ""))
    if dataset_url and _try_kaggle_download(paths.dataset_path, dataset_url):
        LOGGER.info("Dataset downloaded and extracted to %s", paths.dataset_path)
        return

    local_source = params.get("dataset_local_path")
    if local_source and _copy_from_local_source(Path(local_source), paths.dataset_path):
        LOGGER.info("Dataset copied from local source to %s", paths.dataset_path)
        return

    raise FileNotFoundError(
        "Intel dataset is not available locally and Kaggle download could not be completed. "
        "Place the dataset under data/raw."
    )


if __name__ == "__main__":
    main()
