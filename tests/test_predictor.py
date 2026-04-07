"""Tests for src.predictor."""

from __future__ import annotations

import numpy as np
import pytest

from src.graph_builder import build_sample_graph
from src.simulator import generate_synthetic_traffic
from src.neat_model import load_neat_config, evolve
from src.predictor import CongestionPredictor


@pytest.fixture(scope="module")
def trained_predictor() -> CongestionPredictor:
    """Train a minimal NEAT model and return a CongestionPredictor."""
    G = build_sample_graph(num_nodes=15, seed=10)
    X, y, _ = generate_synthetic_traffic(G, sample_edges=10, seed=10)
    cfg = load_neat_config()
    winner, _ = evolve(X, y, cfg, generations=5)
    return CongestionPredictor(winner, cfg)


@pytest.fixture(scope="module")
def sample_graph():
    """Return a small sample graph."""
    return build_sample_graph(num_nodes=15, seed=10)


class TestCongestionPredictor:
    """Tests for the CongestionPredictor wrapper."""

    def test_predict_single_returns_float(self, trained_predictor: CongestionPredictor) -> None:
        features = np.random.default_rng(42).random(9)
        result = trained_predictor.predict_single(features)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_batch_shape(self, trained_predictor: CongestionPredictor) -> None:
        X = np.random.default_rng(42).random((5, 9))
        preds = trained_predictor.predict_batch(X)
        assert preds.shape == (5,)
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_predict_edge(
        self,
        trained_predictor: CongestionPredictor,
        sample_graph,
    ) -> None:
        u, v, k = next(iter(sample_graph.edges(keys=True)))
        score = trained_predictor.predict_edge(sample_graph, u, v, k)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_all_edges(
        self,
        trained_predictor: CongestionPredictor,
        sample_graph,
    ) -> None:
        result = trained_predictor.predict_all_edges(sample_graph)
        assert isinstance(result, dict)
        assert len(result) == sample_graph.number_of_edges()
        for score in result.values():
            assert 0.0 <= score <= 1.0
