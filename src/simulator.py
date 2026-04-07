"""Synthetic traffic data generator for Bengaluru roads."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.feature_engineering import compute_target_congestion

logger = logging.getLogger(__name__)


def _peak_factor(hour: int) -> float:
    """Return a traffic-volume multiplier based on hour of day.

    Peak hours (8–10, 17–19) get a high factor; night hours get a low one.
    """
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        return 1.0
    if 6 <= hour <= 7 or 11 <= hour <= 16 or 20 <= hour <= 21:
        return 0.6
    return 0.2


def generate_synthetic_traffic(
    G: nx.MultiDiGraph,
    sample_edges: int | None = None,
    hours: list[int] | None = None,
    days: list[int] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Generate synthetic feature/target pairs for training.

    For each sampled edge × hour × day combination the function produces:

    * a feature vector (via :func:`src.feature_engineering.edge_features`)
    * a congestion target in [0, 1]

    Args:
        G: Enriched road network graph.
        sample_edges: Number of edges to sample.  ``None`` = all.
        hours: Hours to simulate.  Defaults to a representative set.
        days: Days of week to simulate.  Defaults to ``[0, 2, 4, 5]``.
        seed: Random seed.

    Returns:
        ``(X, y, metadata)`` where *X* has shape ``(n, 9)``, *y* has shape
        ``(n,)`` and *metadata* is a list of dicts with edge/hour/day info.
    """
    from src.feature_engineering import edge_features

    rng = np.random.default_rng(seed)

    if hours is None:
        hours = [6, 8, 9, 12, 15, 17, 18, 21, 23]
    if days is None:
        days = [0, 2, 4, 5]

    all_edges = list(G.edges(keys=True))
    if sample_edges is not None and sample_edges < len(all_edges):
        indices = rng.choice(len(all_edges), size=sample_edges, replace=False)
        edges = [all_edges[i] for i in indices]
    else:
        edges = all_edges

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta: list[dict[str, Any]] = []

    for u, v, k in edges:
        data = G.edges[u, v, k]
        free_flow = data.get("speed_kph", 25.0)

        for hour in hours:
            for day in days:
                peak = _peak_factor(hour)
                rain_mm = float(rng.exponential(1.0)) if rng.random() < 0.3 else 0.0
                rain_mm = round(rain_mm, 1)

                # Weekend has less traffic
                weekend_factor = 0.7 if day >= 5 else 1.0

                # Simulate observed speed
                noise = rng.normal(0, 0.05)
                speed_ratio = 1.0 - peak * weekend_factor * 0.6 + noise
                speed_ratio -= rain_mm * 0.02  # rain slows traffic
                speed_ratio = np.clip(speed_ratio, 0.05, 1.0)
                observed_speed = free_flow * speed_ratio

                # Random incidents (5 % chance during peak)
                if peak > 0.8 and rng.random() < 0.05:
                    observed_speed *= rng.uniform(0.1, 0.4)

                load = 1.0 - speed_ratio  # synthetic load proxy

                feat = edge_features(
                    G, u, v, k,
                    hour=hour, day_of_week=day,
                    rain_mm=rain_mm, load=load,
                )
                target = compute_target_congestion(observed_speed, free_flow)

                X_list.append(feat)
                y_list.append(target)
                meta.append({
                    "u": u, "v": v, "key": k,
                    "hour": hour, "day_of_week": day,
                    "rain_mm": rain_mm,
                    "observed_speed_kph": round(observed_speed, 2),
                    "congestion": round(target, 4),
                })

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    logger.info("Generated %d synthetic samples from %d edges", len(y), len(edges))
    return X, y, meta


def synthetic_to_dataframe(meta: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert metadata from :func:`generate_synthetic_traffic` to a DataFrame.

    Args:
        meta: Metadata list produced by the synthetic generator.

    Returns:
        A :class:`pandas.DataFrame` with the standard traffic CSV schema.
    """
    return pd.DataFrame(meta)
