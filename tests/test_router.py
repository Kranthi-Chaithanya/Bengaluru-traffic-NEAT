"""Tests for src.router."""

from __future__ import annotations

import numpy as np
import pytest

from src.graph_builder import build_sample_graph
from src.simulator import generate_synthetic_traffic
from src.neat_model import load_neat_config, evolve
from src.predictor import CongestionPredictor
from src.router import (
    assign_trips,
    baseline_routes,
    compare_baseline_vs_optimized,
    find_route,
    path_edges,
    predict_edge_costs,
)


@pytest.fixture(scope="module")
def graph_and_predictor():
    """Build a small graph and train a predictor."""
    G = build_sample_graph(num_nodes=20, seed=55)
    X, y, _ = generate_synthetic_traffic(G, sample_edges=15, seed=55)
    cfg = load_neat_config()
    winner, _ = evolve(X, y, cfg, generations=5)
    predictor = CongestionPredictor(winner, cfg)
    return G, predictor


class TestPredictEdgeCosts:
    """Tests for predict_edge_costs."""

    def test_returns_dict(self, graph_and_predictor) -> None:
        G, predictor = graph_and_predictor
        costs = predict_edge_costs(G, predictor)
        assert isinstance(costs, dict)
        assert len(costs) == G.number_of_edges()

    def test_costs_are_positive(self, graph_and_predictor) -> None:
        G, predictor = graph_and_predictor
        costs = predict_edge_costs(G, predictor)
        for c in costs.values():
            assert c >= 0


class TestFindRoute:
    """Tests for find_route."""

    def test_finds_existing_path(self, graph_and_predictor) -> None:
        G, predictor = graph_and_predictor
        costs = predict_edge_costs(G, predictor)
        from src.router import _set_edge_weights
        _set_edge_weights(G, costs)
        # Find two connected nodes
        nodes = list(G.nodes())
        import networkx as nx
        for s in nodes:
            for t in nodes:
                if s != t:
                    try:
                        if nx.has_path(G, s, t):
                            route = find_route(G, s, t)
                            assert route is not None
                            assert route[0] == s
                            assert route[-1] == t
                            return
                    except nx.NodeNotFound:
                        continue
        pytest.skip("No connected pair found in sample graph")

    def test_returns_none_for_disconnected(self, graph_and_predictor) -> None:
        G, _ = graph_and_predictor
        route = find_route(G, "nonexistent_a", "nonexistent_b")
        assert route is None


class TestPathEdges:
    """Tests for path_edges."""

    def test_converts_node_path(self) -> None:
        edges = path_edges([1, 2, 3, 4])
        assert edges == [(1, 2, 0), (2, 3, 0), (3, 4, 0)]

    def test_single_edge(self) -> None:
        assert path_edges([10, 20]) == [(10, 20, 0)]


class TestAssignTrips:
    """Tests for assign_trips."""

    def test_returns_correct_structure(self, graph_and_predictor) -> None:
        G, predictor = graph_and_predictor
        import networkx as nx
        nodes = list(G.nodes())
        rng = np.random.default_rng(42)
        trips = []
        for _ in range(50):
            s, t = rng.choice(nodes, size=2, replace=False)
            if s != t:
                try:
                    if nx.has_path(G, s, t):
                        trips.append((s, t))
                        if len(trips) == 5:
                            break
                except nx.NodeNotFound:
                    continue

        if not trips:
            pytest.skip("No connected pairs found")

        routes, counts = assign_trips(G, trips, predictor)
        assert len(routes) == len(trips)
        assert isinstance(counts, dict)


class TestBaselineRoutes:
    """Tests for baseline_routes."""

    def test_returns_routes(self, graph_and_predictor) -> None:
        G, _ = graph_and_predictor
        import networkx as nx
        nodes = list(G.nodes())
        rng = np.random.default_rng(99)
        trips = []
        for _ in range(50):
            s, t = rng.choice(nodes, size=2, replace=False)
            if s != t:
                try:
                    if nx.has_path(G, s, t):
                        trips.append((s, t))
                        if len(trips) == 3:
                            break
                except nx.NodeNotFound:
                    continue

        if not trips:
            pytest.skip("No connected pairs found")

        routes, counts = baseline_routes(G, trips)
        assert len(routes) == len(trips)


class TestCompareBaseline:
    """Tests for compare_baseline_vs_optimized."""

    def test_returns_metrics(self) -> None:
        b = {(0, 1, 0): 3, (1, 2, 0): 5}
        o = {(0, 1, 0): 2, (1, 2, 0): 2, (2, 3, 0): 2}
        metrics = compare_baseline_vs_optimized(b, o)
        assert "max_load_baseline" in metrics
        assert "max_load_optimized" in metrics
        assert metrics["max_load_baseline"] == 5
        assert metrics["max_load_optimized"] == 2
        assert metrics["edges_used_optimized"] > metrics["edges_used_baseline"]
