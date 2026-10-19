"""Convenience entrypoint for local end-to-end execution.

This file is optional for the DVC pipeline, but useful for manual runs.
"""

from __future__ import annotations

import logging


def main() -> None:
    """Run the ML pipeline stages sequentially.

    DVC remains the primary orchestration mechanism. This entrypoint exists
    for users who want to run the stages manually during development.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Use 'dvc repro' for the full automated pipeline.")


if __name__ == "__main__":
    main()
