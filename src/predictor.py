"""Predictor: inference wrapper around a trained NEAT genome."""

from __future__ import annotations

import logging
from typing import Any

import neat
import networkx as nx
import numpy as np

from src.feature_engineering import NUM_FEATURES, batch_edge_features, edge_features

logger = logging.getLogger(__name__)


class CongestionPredictor:
    """Thin wrapper for predicting edge congestion with a NEAT network.

    Args:
        genome: A trained NEAT genome.
        config: The matching NEAT configuration.
    """

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config) -> None:
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        logger.info("CongestionPredictor initialised (fitness=%.4f)", genome.fitness or 0.0)

    # ------------------------------------------------------------------
    # Single-edge prediction
    # ------------------------------------------------------------------

    def predict_single(self, features: np.ndarray) -> float:
        """Predict the congestion score for a single feature vector.

        Args:
            features: 1-D array of shape ``(NUM_FEATURES,)``.

        Returns:
            Congestion score in [0, 1].
        """
        output = self.net.activate(features.tolist())
        return max(0.0, min(1.0, output[0]))

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict congestion for a batch of feature vectors.

        Args:
            X: 2-D array of shape ``(n, NUM_FEATURES)``.

        Returns:
            1-D array of congestion scores of shape ``(n,)``.
        """
        preds: list[float] = []
        for row in X:
            preds.append(self.predict_single(row))
        return np.array(preds, dtype=np.float64)

    # ------------------------------------------------------------------
    # Graph-level helpers
    # ------------------------------------------------------------------

    def predict_edge(
        self,
        G: nx.MultiDiGraph,
        u: Any,
        v: Any,
        key: int = 0,
        hour: int = 12,
        day_of_week: int = 2,
        rain_mm: float = 0.0,
        load: float = 0.0,
    ) -> float:
        """Predict congestion for a specific graph edge.

        Args:
            G: Road network graph.
            u: Source node.
            v: Target node.
            key: Edge key.
            hour: Hour of day.
            day_of_week: Day of week.
            rain_mm: Rainfall.
            load: Current traffic load.

        Returns:
            Congestion score in [0, 1].
        """
        feat = edge_features(G, u, v, key, hour, day_of_week, rain_mm, load)
        return self.predict_single(feat)

    def predict_all_edges(
        self,
        G: nx.MultiDiGraph,
        hour: int = 12,
        day_of_week: int = 2,
        rain_mm: float = 0.0,
        loads: dict[tuple[Any, Any, int], float] | None = None,
    ) -> dict[tuple[Any, Any, int], float]:
        """Predict congestion for every edge in the graph.

        Args:
            G: Road network graph.
            hour: Hour of day.
            day_of_week: Day of week.
            rain_mm: Rainfall.
            loads: Optional per-edge loads.

        Returns:
            Mapping ``(u, v, key) → congestion_score``.
        """
        edge_list, X = batch_edge_features(
            G, hour=hour, day_of_week=day_of_week,
            rain_mm=rain_mm, loads=loads,
        )
        preds = self.predict_batch(X)
        return {e: float(p) for e, p in zip(edge_list, preds)}
