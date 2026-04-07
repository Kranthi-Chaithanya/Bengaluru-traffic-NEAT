"""Tests for src.graph_builder."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.graph_builder import (
    ROAD_TYPE_ENCODING,
    build_sample_graph,
    enrich_graph,
    load_graph,
    save_graph,
    _estimate_lanes,
    _estimate_speed,
    _resolve_highway,
)


class TestResolveHighway:
    """Tests for _resolve_highway."""

    def test_string_passthrough(self) -> None:
        assert _resolve_highway("primary") == "primary"

    def test_list_takes_first(self) -> None:
        assert _resolve_highway(["secondary", "tertiary"]) == "secondary"

    def test_none_defaults_to_unclassified(self) -> None:
        assert _resolve_highway(None) == "unclassified"


class TestEstimateSpeed:
    """Tests for _estimate_speed."""

    def test_uses_maxspeed_when_available(self) -> None:
        assert _estimate_speed("residential", "40") == 40.0

    def test_falls_back_to_default(self) -> None:
        speed = _estimate_speed("motorway", None)
        assert speed == 80.0

    def test_handles_list_maxspeed(self) -> None:
        assert _estimate_speed("primary", ["60 kph", "50"]) == 60.0

    def test_invalid_maxspeed_falls_back(self) -> None:
        speed = _estimate_speed("tertiary", "unknown")
        assert speed == 30.0


class TestEstimateLanes:
    """Tests for _estimate_lanes."""

    def test_none_defaults_to_one(self) -> None:
        assert _estimate_lanes(None) == 1

    def test_int_passthrough(self) -> None:
        assert _estimate_lanes(3) == 3

    def test_list_takes_first(self) -> None:
        assert _estimate_lanes(["2", "3"]) == 2

    def test_invalid_defaults_to_one(self) -> None:
        assert _estimate_lanes("many") == 1


class TestBuildSampleGraph:
    """Tests for build_sample_graph."""

    def test_creates_multidigraph(self) -> None:
        G = build_sample_graph(num_nodes=10, seed=1)
        assert isinstance(G, nx.MultiDiGraph)

    def test_node_count(self) -> None:
        G = build_sample_graph(num_nodes=20, seed=2)
        assert G.number_of_nodes() == 20

    def test_edges_have_attributes(self) -> None:
        G = build_sample_graph(num_nodes=10, seed=3)
        for u, v, k, data in G.edges(keys=True, data=True):
            assert "highway_str" in data
            assert "speed_kph" in data
            assert "length" in data
            assert "travel_time_s" in data
            assert "road_type_code" in data
            assert "lanes" in data

    def test_deterministic(self) -> None:
        G1 = build_sample_graph(num_nodes=10, seed=99)
        G2 = build_sample_graph(num_nodes=10, seed=99)
        assert G1.number_of_edges() == G2.number_of_edges()


class TestEnrichGraph:
    """Tests for enrich_graph on a minimal hand-built graph."""

    def test_enriches_edges(self) -> None:
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, 0, highway="primary", length=500.0)
        G = enrich_graph(G)
        data = G.edges[0, 1, 0]
        assert data["highway_str"] == "primary"
        assert data["road_type_code"] == ROAD_TYPE_ENCODING["primary"]
        assert data["speed_kph"] == 50.0
        assert data["travel_time_s"] > 0


class TestSaveLoadGraph:
    """Tests for graph serialisation round-trip."""

    def test_round_trip(self) -> None:
        G = build_sample_graph(num_nodes=8, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.pkl"
            save_graph(G, path)
            assert path.exists()
            G2 = load_graph(path)
            assert G2.number_of_nodes() == G.number_of_nodes()
            assert G2.number_of_edges() == G.number_of_edges()

    def test_load_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_graph("/nonexistent/path.pkl")
