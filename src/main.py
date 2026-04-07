"""CLI entry-point for the Bengaluru NEAT Traffic project.

Usage example::

    python -m src.main --generations 30 --trip-count 20 --sample-edges 1500 --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.graph_builder import build_bengaluru_graph, build_sample_graph, enrich_graph
from src.simulator import generate_synthetic_traffic
from src.neat_model import load_neat_config, evolve, save_model, load_model
from src.predictor import CongestionPredictor
from src.router import (
    assign_trips,
    baseline_routes,
    compare_baseline_vs_optimized,
)
from src.visualize import plot_graph_congestion, plot_route_comparison, plot_metrics

logger = logging.getLogger("bengaluru_neat_traffic")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Bengaluru NEAT Traffic – AI-powered congestion routing",
    )
    parser.add_argument(
        "--generations", type=int, default=30,
        help="Number of NEAT generations (default: 30)",
    )
    parser.add_argument(
        "--trip-count", type=int, default=20,
        help="Number of trips to route (default: 20)",
    )
    parser.add_argument(
        "--sample-edges", type=int, default=1500,
        help="Number of edges to sample for training data (default: 1500)",
    )
    parser.add_argument(
        "--use-real-data", action="store_true",
        help="Use real Bengaluru graph (requires internet for first run)",
    )
    parser.add_argument(
        "--traffic-csv", type=str, default=None,
        help="Path to a real traffic CSV file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for output artefacts (default: outputs)",
    )
    parser.add_argument(
        "--neat-config", type=str, default=None,
        help="Path to NEAT config INI file",
    )
    parser.add_argument(
        "--graph-nodes", type=int, default=50,
        help="Number of nodes in the sample graph (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    """Configure the root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _random_trips(G: Any, count: int, seed: int) -> list[tuple[Any, Any]]:
    """Pick random OD pairs that actually have paths in the graph."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    trips: list[tuple[Any, Any]] = []
    attempts = 0
    max_attempts = count * 20
    while len(trips) < count and attempts < max_attempts:
        s, t = rng.choice(nodes, size=2, replace=False)
        if s != t and _has_path(G, s, t):
            trips.append((s, t))
        attempts += 1
    return trips


def _has_path(G: Any, source: Any, target: Any) -> bool:
    """Check reachability without raising."""
    import networkx as nx
    try:
        return nx.has_path(G, source, target)
    except nx.NodeNotFound:
        return False


def main(argv: list[str] | None = None) -> None:
    """Run the full pipeline: build graph → train → route → visualise."""
    args = parse_args(argv)
    _setup_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build graph -------------------------------------------------------
    if args.use_real_data:
        logger.info("Building real Bengaluru graph …")
        G = build_bengaluru_graph(cache_dir="data")
    else:
        logger.info("Building sample graph with %d nodes …", args.graph_nodes)
        G = build_sample_graph(num_nodes=args.graph_nodes, seed=args.seed)

    logger.info("Graph ready: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # 2. Generate training data --------------------------------------------
    logger.info("Generating synthetic training data …")
    X, y, meta = generate_synthetic_traffic(
        G, sample_edges=args.sample_edges, seed=args.seed,
    )
    logger.info("Training data: X=%s  y=%s", X.shape, y.shape)

    # 3. Train NEAT model --------------------------------------------------
    neat_cfg = load_neat_config(args.neat_config)
    winner, _ = evolve(X, y, neat_cfg, generations=args.generations)
    save_model(winner, output_dir / "winner_genome.pkl")

    # 4. Predict congestion ------------------------------------------------
    predictor = CongestionPredictor(winner, neat_cfg)
    congestion = predictor.predict_all_edges(G)
    logger.info(
        "Congestion prediction: min=%.3f  mean=%.3f  max=%.3f",
        min(congestion.values()),
        np.mean(list(congestion.values())),
        max(congestion.values()),
    )

    # 5. Route trips -------------------------------------------------------
    trips = _random_trips(G, count=args.trip_count, seed=args.seed)
    logger.info("Generated %d random trips", len(trips))

    baseline_paths, baseline_counts = baseline_routes(G, trips)
    optimized_paths, optimized_counts = assign_trips(
        G, trips, predictor, hour=8, day_of_week=0,
    )

    # 6. Compare ------------------------------------------------------------
    metrics = compare_baseline_vs_optimized(baseline_counts, optimized_counts)
    logger.info("Comparison metrics:\n%s", json.dumps(metrics, indent=2))

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # 7. Visualise ----------------------------------------------------------
    plot_graph_congestion(G, congestion, output_path=output_dir / "congestion_map.png")
    plot_route_comparison(
        G, baseline_counts, optimized_counts,
        output_path=output_dir / "route_comparison.png",
    )
    plot_metrics(metrics, output_path=output_dir / "metrics_chart.png")

    logger.info("Done! All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
