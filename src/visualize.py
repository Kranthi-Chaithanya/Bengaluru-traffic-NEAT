"""Visualisation utilities for the Bengaluru traffic project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def plot_graph_congestion(
    G: nx.MultiDiGraph,
    congestion: dict[tuple[Any, Any, int], float],
    title: str = "Predicted Edge Congestion",
    output_path: str | Path | None = None,
) -> None:
    """Draw the graph with edges coloured by congestion score.

    Args:
        G: Road network graph.
        congestion: Mapping ``(u, v, key) → congestion`` in [0, 1].
        title: Plot title.
        output_path: If given, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Use spring layout for sample graphs; real graphs may have lat/lon
    pos = _get_positions(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=5, node_color="black", alpha=0.4)

    # Draw edges with colour mapped to congestion
    edge_list = list(G.edges(keys=True))
    colors = [congestion.get(e, 0.0) for e in edge_list]
    edges_no_key = [(u, v) for u, v, _ in edge_list]

    nx.draw_networkx_edges(
        G, pos, edgelist=edges_no_key, ax=ax,
        edge_color=colors, edge_cmap=plt.cm.RdYlGn_r,
        edge_vmin=0, edge_vmax=1, width=1.0, alpha=0.7,
        arrows=False,
    )

    sm = cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Congestion (0=free, 1=jam)")

    ax.set_title(title)
    ax.axis("off")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved congestion plot to %s", output_path)
    plt.close(fig)


def plot_route_comparison(
    G: nx.MultiDiGraph,
    baseline_counts: dict[tuple[Any, Any, int], int],
    optimized_counts: dict[tuple[Any, Any, int], int],
    title: str = "Baseline vs Optimised Route Loads",
    output_path: str | Path | None = None,
) -> None:
    """Side-by-side comparison of edge loads for baseline and optimised routing.

    Args:
        G: Road network graph.
        baseline_counts: Edge trip counts from baseline routing.
        optimized_counts: Edge trip counts from optimised routing.
        title: Plot title.
        output_path: If given, saves the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    pos = _get_positions(G)

    for ax, counts, label in [
        (axes[0], baseline_counts, "Baseline"),
        (axes[1], optimized_counts, "NEAT-Optimised"),
    ]:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3, node_color="grey", alpha=0.3)
        edge_list = list(G.edges(keys=True))
        loads = [counts.get(e, 0) for e in edge_list]
        max_load = max(loads) if loads and max(loads) > 0 else 1
        norm_loads = [l / max_load for l in loads]
        edges_no_key = [(u, v) for u, v, _ in edge_list]

        nx.draw_networkx_edges(
            G, pos, edgelist=edges_no_key, ax=ax,
            edge_color=norm_loads, edge_cmap=plt.cm.hot_r,
            edge_vmin=0, edge_vmax=1, width=1.5, alpha=0.8,
            arrows=False,
        )
        ax.set_title(f"{label} (max load={max(loads)})")
        ax.axis("off")

    fig.suptitle(title, fontsize=14)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved route comparison plot to %s", output_path)
    plt.close(fig)


def plot_metrics(
    metrics: dict[str, float],
    output_path: str | Path | None = None,
) -> None:
    """Bar chart comparing baseline and optimised routing metrics.

    Args:
        metrics: Output of :func:`src.router.compare_baseline_vs_optimized`.
        output_path: If given, saves the figure.
    """
    labels = ["Max Load", "Mean Load", "Edges Used", "Load Std"]
    baseline_vals = [
        metrics["max_load_baseline"],
        metrics["mean_load_baseline"],
        metrics["edges_used_baseline"],
        metrics["load_std_baseline"],
    ]
    optimized_vals = [
        metrics["max_load_optimized"],
        metrics["mean_load_optimized"],
        metrics["edges_used_optimized"],
        metrics["load_std_optimized"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="salmon")
    ax.bar(x + width / 2, optimized_vals, width, label="NEAT-Optimised", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Routing Comparison Metrics")
    ax.set_ylabel("Value")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved metrics plot to %s", output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_positions(G: nx.MultiDiGraph) -> dict[Any, tuple[float, float]]:
    """Extract node positions from the graph.

    If nodes have ``x`` and ``y`` attributes (from OSMnx) those are used;
    otherwise a spring layout is computed.
    """
    sample_node = next(iter(G.nodes()))
    node_data = G.nodes[sample_node]
    if "x" in node_data and "y" in node_data:
        return {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    return nx.spring_layout(G, seed=42)
