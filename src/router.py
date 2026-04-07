"""Route optimiser that uses predicted congestion to reduce overall traffic."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from src.predictor import CongestionPredictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge-cost computation
# ---------------------------------------------------------------------------

def predict_edge_costs(
    G: nx.MultiDiGraph,
    predictor: CongestionPredictor,
    hour: int = 12,
    day_of_week: int = 2,
    rain_mm: float = 0.0,
    loads: dict[tuple[Any, Any, int], float] | None = None,
    w_distance: float = 0.3,
    w_time: float = 0.3,
    w_congestion: float = 0.2,
    w_overload: float = 0.2,
    trip_counts: dict[tuple[Any, Any, int], int] | None = None,
) -> dict[tuple[Any, Any, int], float]:
    """Compute a composite cost for every edge.

    Cost = w_distance * norm_length
         + w_time     * norm_travel_time
         + w_congestion * congestion_score
         + w_overload * overload_penalty

    Args:
        G: Road network graph.
        predictor: Trained :class:`CongestionPredictor`.
        hour: Hour of day.
        day_of_week: Day of week.
        rain_mm: Rainfall.
        loads: Per-edge traffic loads.
        w_distance: Weight for normalised distance.
        w_time: Weight for normalised travel time.
        w_congestion: Weight for predicted congestion.
        w_overload: Weight for overload penalty.
        trip_counts: How many trips have already been assigned to each edge.

    Returns:
        Mapping ``(u, v, key) → composite_cost``.
    """
    if trip_counts is None:
        trip_counts = {}

    congestion = predictor.predict_all_edges(
        G, hour=hour, day_of_week=day_of_week,
        rain_mm=rain_mm, loads=loads,
    )

    costs: dict[tuple[Any, Any, int], float] = {}
    max_len = max(
        (d.get("length", 100.0) for _, _, _, d in G.edges(keys=True, data=True)),
        default=1.0,
    )
    max_tt = max(
        (d.get("travel_time_s", 10.0) for _, _, _, d in G.edges(keys=True, data=True)),
        default=1.0,
    )

    for u, v, k, data in G.edges(keys=True, data=True):
        edge_key = (u, v, k)
        norm_len = data.get("length", 100.0) / max_len
        norm_tt = data.get("travel_time_s", 10.0) / max_tt
        cong = congestion.get(edge_key, 0.5)
        tc = trip_counts.get(edge_key, 0)
        overload = min(tc / 10.0, 1.0)  # saturate at 10 trips

        cost = (
            w_distance * norm_len
            + w_time * norm_tt
            + w_congestion * cong
            + w_overload * overload
        )
        costs[edge_key] = cost

    return costs


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _set_edge_weights(G: nx.MultiDiGraph,
                      costs: dict[tuple[Any, Any, int], float],
                      attr: str = "route_cost") -> None:
    """Write computed costs back as an edge attribute."""
    for (u, v, k), c in costs.items():
        if G.has_edge(u, v, k):
            G.edges[u, v, k][attr] = c


def find_route(
    G: nx.MultiDiGraph,
    source: Any,
    target: Any,
    weight: str = "route_cost",
) -> list[Any] | None:
    """Find the shortest path using the given weight attribute.

    Args:
        G: Road network graph (must already have *weight* attribute set).
        source: Source node.
        target: Target node.
        weight: Edge attribute used as cost.

    Returns:
        List of nodes on the path, or ``None`` if no path exists.
    """
    try:
        return nx.shortest_path(G, source, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        logger.warning("No path from %s to %s", source, target)
        return None


def path_edges(path: list[Any]) -> list[tuple[Any, Any, int]]:
    """Convert a node-level path to a list of edge keys (using key=0)."""
    return [(path[i], path[i + 1], 0) for i in range(len(path) - 1)]


def assign_trips(
    G: nx.MultiDiGraph,
    trips: list[tuple[Any, Any]],
    predictor: CongestionPredictor,
    hour: int = 12,
    day_of_week: int = 2,
    rain_mm: float = 0.0,
    **cost_kwargs: Any,
) -> tuple[list[list[Any] | None], dict[tuple[Any, Any, int], int]]:
    """Sequentially assign trips, updating edge loads after each one.

    This implements the traffic-reduction logic: later trips are penalised
    on edges already used by earlier trips, spreading traffic across the
    network.

    Args:
        G: Road network graph.
        trips: List of ``(source, target)`` pairs.
        predictor: Trained congestion predictor.
        hour: Hour of day.
        day_of_week: Day of week.
        rain_mm: Rainfall.
        **cost_kwargs: Extra keyword arguments forwarded to
            :func:`predict_edge_costs`.

    Returns:
        A tuple ``(routes, trip_counts)`` where *routes* is a list of
        node-level paths (or ``None``) and *trip_counts* tracks how many
        trips traverse each edge.
    """
    trip_counts: dict[tuple[Any, Any, int], int] = {}
    routes: list[list[Any] | None] = []

    for idx, (src, tgt) in enumerate(trips):
        logger.debug("Assigning trip %d/%d: %s → %s", idx + 1, len(trips), src, tgt)
        costs = predict_edge_costs(
            G, predictor,
            hour=hour, day_of_week=day_of_week,
            rain_mm=rain_mm, trip_counts=trip_counts,
            **cost_kwargs,
        )
        _set_edge_weights(G, costs, attr="route_cost")
        route = find_route(G, src, tgt, weight="route_cost")
        routes.append(route)

        if route is not None:
            for edge in path_edges(route):
                trip_counts[edge] = trip_counts.get(edge, 0) + 1

    logger.info("Assigned %d trips (%d found paths)",
                len(trips), sum(1 for r in routes if r is not None))
    return routes, trip_counts


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def baseline_routes(
    G: nx.MultiDiGraph,
    trips: list[tuple[Any, Any]],
    weight: str = "length",
) -> tuple[list[list[Any] | None], dict[tuple[Any, Any, int], int]]:
    """Compute shortest-distance routes without congestion awareness.

    Args:
        G: Road network graph.
        trips: List of ``(source, target)`` pairs.
        weight: Edge attribute used as cost (default ``"length"``).

    Returns:
        Same format as :func:`assign_trips`.
    """
    trip_counts: dict[tuple[Any, Any, int], int] = {}
    routes: list[list[Any] | None] = []

    for src, tgt in trips:
        try:
            route = nx.shortest_path(G, src, tgt, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            route = None
        routes.append(route)
        if route is not None:
            for edge in path_edges(route):
                trip_counts[edge] = trip_counts.get(edge, 0) + 1

    return routes, trip_counts


def compare_baseline_vs_optimized(
    baseline_counts: dict[tuple[Any, Any, int], int],
    optimized_counts: dict[tuple[Any, Any, int], int],
) -> dict[str, float]:
    """Compute comparison metrics between baseline and optimised routing.

    Metrics:

    * ``max_load_baseline`` / ``max_load_optimized``
    * ``mean_load_baseline`` / ``mean_load_optimized``
    * ``edges_used_baseline`` / ``edges_used_optimized``
    * ``load_std_baseline`` / ``load_std_optimized``

    Args:
        baseline_counts: Edge trip counts from baseline routing.
        optimized_counts: Edge trip counts from optimised routing.

    Returns:
        Dictionary of metric names to values.
    """

    def _stats(counts: dict[tuple[Any, Any, int], int]) -> dict[str, float]:
        vals = list(counts.values()) if counts else [0]
        return {
            "max_load": float(max(vals)),
            "mean_load": float(np.mean(vals)),
            "edges_used": float(len(vals)),
            "load_std": float(np.std(vals)),
        }

    b = _stats(baseline_counts)
    o = _stats(optimized_counts)

    return {
        "max_load_baseline": b["max_load"],
        "max_load_optimized": o["max_load"],
        "mean_load_baseline": b["mean_load"],
        "mean_load_optimized": o["mean_load"],
        "edges_used_baseline": b["edges_used"],
        "edges_used_optimized": o["edges_used"],
        "load_std_baseline": b["load_std"],
        "load_std_optimized": o["load_std"],
    }
