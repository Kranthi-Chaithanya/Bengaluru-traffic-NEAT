"""Build and cache the Bengaluru road network graph."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Default speed limits (km/h) by OSM highway tag
_DEFAULT_SPEEDS: dict[str, float] = {
    "motorway": 80.0,
    "motorway_link": 60.0,
    "trunk": 60.0,
    "trunk_link": 50.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 40.0,
    "secondary_link": 35.0,
    "tertiary": 30.0,
    "tertiary_link": 25.0,
    "residential": 20.0,
    "living_street": 15.0,
    "unclassified": 25.0,
    "service": 15.0,
}

# Numeric encoding for road types
ROAD_TYPE_ENCODING: dict[str, int] = {
    "motorway": 0,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 3,
    "primary": 4,
    "primary_link": 5,
    "secondary": 6,
    "secondary_link": 7,
    "tertiary": 8,
    "tertiary_link": 9,
    "residential": 10,
    "living_street": 11,
    "unclassified": 12,
    "service": 13,
}


def _resolve_highway(highway: Any) -> str:
    """Resolve the ``highway`` attribute to a single string.

    OSMnx may return a list when a segment has multiple highway tags.
    """
    if isinstance(highway, list):
        highway = highway[0]
    return str(highway) if highway else "unclassified"


def _estimate_speed(highway_tag: str, maxspeed: Any) -> float:
    """Return an estimated speed in km/h for an edge.

    Uses *maxspeed* if available, otherwise falls back to defaults.
    """
    if maxspeed is not None:
        try:
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            return float(str(maxspeed).split()[0])
        except (ValueError, IndexError):
            pass
    return _DEFAULT_SPEEDS.get(highway_tag, 25.0)


def _estimate_lanes(lanes: Any) -> int:
    """Return the number of lanes, defaulting to 1."""
    if lanes is None:
        return 1
    try:
        if isinstance(lanes, list):
            lanes = lanes[0]
        return max(1, int(float(str(lanes))))
    except (ValueError, TypeError):
        return 1


def download_graph(place: str = "Bengaluru, India",
                   network_type: str = "drive") -> nx.MultiDiGraph:
    """Download the drivable road network via OSMnx.

    Args:
        place: Place query for OSMnx.
        network_type: Network type (``"drive"``).

    Returns:
        A :class:`networkx.MultiDiGraph` with enriched edge attributes.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "osmnx is required. Install it with: pip install osmnx"
        ) from exc

    logger.info("Downloading road network for '%s' …", place)
    G = ox.graph_from_place(place, network_type=network_type)
    logger.info("Downloaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def enrich_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add derived attributes to every edge in the graph.

    Attributes added or normalised:

    * ``highway_str`` – single-string road type
    * ``road_type_code`` – numeric road-type encoding
    * ``speed_kph`` – estimated speed
    * ``lanes`` – estimated lane count
    * ``length`` – edge length in metres (kept from OSMnx)
    * ``travel_time_s`` – estimated travel time in seconds

    Args:
        G: A road-network graph (typically from OSMnx).

    Returns:
        The same graph, mutated in place.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = _resolve_highway(data.get("highway"))
        data["highway_str"] = hw
        data["road_type_code"] = ROAD_TYPE_ENCODING.get(hw, 12)

        speed = _estimate_speed(hw, data.get("maxspeed"))
        data["speed_kph"] = speed

        data["lanes"] = _estimate_lanes(data.get("lanes"))

        length_m = data.get("length", 100.0)
        data["length"] = float(length_m)

        if speed > 0:
            data["travel_time_s"] = (float(length_m) / 1000.0) / speed * 3600.0
        else:
            data["travel_time_s"] = float("inf")

    logger.info("Graph enriched with speed, lanes, road-type, travel-time attributes")
    return G


def save_graph(G: nx.MultiDiGraph, path: str | Path) -> None:
    """Serialize the graph to a pickle file.

    Args:
        G: The graph to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Graph saved to %s", path)


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    """Load a graph from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialised :class:`networkx.MultiDiGraph`.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as fh:
        G = pickle.load(fh)  # noqa: S301 – trusted local cache
    logger.info("Graph loaded from %s (%d nodes, %d edges)",
                path, G.number_of_nodes(), G.number_of_edges())
    return G


def build_bengaluru_graph(cache_dir: str | Path = "data",
                          place: str = "Bengaluru, India") -> nx.MultiDiGraph:
    """High-level helper: load from cache or download & enrich.

    Args:
        cache_dir: Directory used for caching the graph pickle.
        place: Place query forwarded to :func:`download_graph`.

    Returns:
        An enriched :class:`networkx.MultiDiGraph`.
    """
    cache_path = Path(cache_dir) / "bengaluru_graph.pkl"
    if cache_path.exists():
        logger.info("Using cached graph at %s", cache_path)
        return load_graph(cache_path)

    G = download_graph(place=place)
    G = enrich_graph(G)
    save_graph(G, cache_path)
    return G


def build_sample_graph(num_nodes: int = 50, seed: int = 42) -> nx.MultiDiGraph:
    """Create a small synthetic directed graph for testing.

    The graph is a random geometric graph converted to a directed multigraph
    with synthetic road attributes attached to every edge.

    Args:
        num_nodes: Number of nodes.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`networkx.MultiDiGraph` with enriched attributes.
    """
    rng = np.random.default_rng(seed)
    # Create a connected random graph
    G_simple = nx.watts_strogatz_graph(num_nodes, k=4, p=0.3, seed=int(seed))
    G = nx.MultiDiGraph()

    road_types = list(ROAD_TYPE_ENCODING.keys())

    for u, v in G_simple.edges():
        hw = rng.choice(road_types)
        length = rng.uniform(100, 2000)
        speed = _DEFAULT_SPEEDS.get(hw, 25.0)
        lanes = rng.choice([1, 2, 3, 4])
        G.add_edge(u, v, 0,
                   highway_str=hw,
                   road_type_code=ROAD_TYPE_ENCODING[hw],
                   speed_kph=speed,
                   lanes=int(lanes),
                   length=float(length),
                   travel_time_s=(length / 1000.0) / speed * 3600.0)
        # Add reverse edge with probability 0.7
        if rng.random() < 0.7:
            G.add_edge(v, u, 0,
                       highway_str=hw,
                       road_type_code=ROAD_TYPE_ENCODING[hw],
                       speed_kph=speed,
                       lanes=int(lanes),
                       length=float(length),
                       travel_time_s=(length / 1000.0) / speed * 3600.0)

    logger.info("Built sample graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G
