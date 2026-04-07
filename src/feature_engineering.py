"""Feature engineering for road-edge congestion prediction."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Number of features produced per edge
NUM_FEATURES = 9


def _betweenness_cache(G: nx.MultiDiGraph) -> dict[Any, float]:
    """Compute edge-betweenness centrality (cached on the graph object).

    The result is stored on the graph as ``graph["_edge_betweenness"]`` so that
    repeated calls do not recompute.
    """
    if "_edge_betweenness" not in G.graph:
        logger.info("Computing edge-betweenness centrality …")
        simple = nx.DiGraph(G)  # collapse multi-edges for centrality
        bc = nx.edge_betweenness_centrality(simple, normalized=True)
        # Map (u, v) → centrality; for multi-edges we reuse the same value
        G.graph["_edge_betweenness"] = bc
    return G.graph["_edge_betweenness"]


def edge_features(G: nx.MultiDiGraph,
                  u: Any,
                  v: Any,
                  key: int,
                  hour: int = 12,
                  day_of_week: int = 2,
                  rain_mm: float = 0.0,
                  load: float = 0.0) -> np.ndarray:
    """Build a feature vector for a single edge.

    Features (in order):

    0. road length (metres, clipped to [0, 5000] then /5000)
    1. baseline speed (km/h, /80)
    2. hour of day (/23)
    3. day of week (/6)
    4. rain level (mm, clipped to [0, 50] then /50)
    5. road type code (/13)
    6. historical/synthetic traffic load (clipped to [0, 1])
    7. edge-betweenness centrality (already in [0, 1])
    8. current simulated occupancy/load (clipped to [0, 1])

    All values are normalised roughly to [0, 1].

    Args:
        G: The road network graph.
        u: Source node.
        v: Target node.
        key: Edge key in the multigraph.
        hour: Hour of day (0–23).
        day_of_week: Day of week (0=Monday … 6=Sunday).
        rain_mm: Rainfall in mm.
        load: Current traffic load / occupancy (0–1).

    Returns:
        A 1-D NumPy array of shape ``(NUM_FEATURES,)``.
    """
    data = G.edges[u, v, key]
    bc_map = _betweenness_cache(G)
    bc_val = bc_map.get((u, v), 0.0)

    feat = np.array([
        min(data.get("length", 100.0), 5000.0) / 5000.0,
        min(data.get("speed_kph", 25.0), 80.0) / 80.0,
        hour / 23.0,
        day_of_week / 6.0,
        min(max(rain_mm, 0.0), 50.0) / 50.0,
        data.get("road_type_code", 12) / 13.0,
        np.clip(load, 0.0, 1.0),
        np.clip(bc_val, 0.0, 1.0),
        np.clip(load, 0.0, 1.0),
    ], dtype=np.float64)
    return feat


def batch_edge_features(G: nx.MultiDiGraph,
                        edges: list[tuple[Any, Any, int]] | None = None,
                        hour: int = 12,
                        day_of_week: int = 2,
                        rain_mm: float = 0.0,
                        loads: dict[tuple[Any, Any, int], float] | None = None,
                        ) -> tuple[list[tuple[Any, Any, int]], np.ndarray]:
    """Compute features for multiple edges at once.

    Args:
        G: The road network graph.
        edges: List of ``(u, v, key)`` tuples.  Defaults to all edges.
        hour: Hour of day.
        day_of_week: Day of week.
        rain_mm: Rainfall.
        loads: Optional mapping of edge → current load.

    Returns:
        A tuple ``(edge_list, feature_matrix)`` where *feature_matrix*
        has shape ``(len(edge_list), NUM_FEATURES)``.
    """
    if edges is None:
        edges = [(u, v, k) for u, v, k in G.edges(keys=True)]
    if loads is None:
        loads = {}

    features = []
    for u, v, k in edges:
        load = loads.get((u, v, k), 0.0)
        features.append(edge_features(G, u, v, k, hour, day_of_week, rain_mm, load))

    return edges, np.array(features, dtype=np.float64)


def compute_target_congestion(speed_kph: float, free_flow_speed: float) -> float:
    """Compute a congestion score in [0, 1] from observed vs free-flow speed.

    0 = free flow, 1 = complete standstill.

    Args:
        speed_kph: Observed speed.
        free_flow_speed: Free-flow (max) speed for that edge.

    Returns:
        Congestion score.
    """
    if free_flow_speed <= 0:
        return 1.0
    ratio = max(0.0, min(speed_kph, free_flow_speed)) / free_flow_speed
    return 1.0 - ratio
