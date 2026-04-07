"""Data loader for traffic CSV files and cached graph data."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_traffic_csv(path: str | Path) -> pd.DataFrame:
    """Load a traffic CSV file and validate required columns.

    Expected schema::

        u, v, key, hour, day_of_week, rain_mm, observed_speed_kph

    Args:
        path: Path to the CSV file.

    Returns:
        A validated :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Traffic CSV not found: {path}")

    logger.info("Loading traffic CSV from %s", path)
    df = pd.read_csv(path)

    required = {"u", "v", "key", "hour", "day_of_week", "rain_mm", "observed_speed_kph"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in traffic CSV: {missing}")

    logger.info("Loaded %d traffic records", len(df))
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved DataFrame (%d rows) to %s", len(df), path)


def get_cache_path(cache_dir: str | Path, filename: str) -> Path:
    """Return the full cache path, creating the directory if needed.

    Args:
        cache_dir: Directory for cached files.
        filename: Name of the file.

    Returns:
        Full :class:`Path` to the cached file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def cache_exists(cache_dir: str | Path, filename: str) -> bool:
    """Check whether a cached file exists.

    Args:
        cache_dir: Directory for cached files.
        filename: Name of the file.

    Returns:
        ``True`` if the file exists.
    """
    return get_cache_path(cache_dir, filename).exists()
