"""NEAT model: evolution, training, saving, and loading."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable

import neat  # neat-python
import numpy as np

logger = logging.getLogger(__name__)


def _default_config_path() -> str:
    """Return the default path to ``neat_config.ini``."""
    return os.path.join(os.path.dirname(__file__), os.pardir, "config", "neat_config.ini")


def load_neat_config(config_path: str | Path | None = None) -> neat.Config:
    """Load NEAT configuration from an INI file.

    Args:
        config_path: Path to the NEAT config file.
            Defaults to ``config/neat_config.ini``.

    Returns:
        A :class:`neat.Config` instance.
    """
    if config_path is None:
        config_path = _default_config_path()
    config_path = str(Path(config_path).resolve())
    logger.info("Loading NEAT config from %s", config_path)
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def make_eval_function(
    X: np.ndarray,
    y: np.ndarray,
    config: neat.Config,
) -> Callable[[list[tuple[int, neat.DefaultGenome]], neat.Config], None]:
    """Create the NEAT genome evaluation (fitness) function.

    Fitness is ``1 - mean_absolute_error``, so higher is better (max 1.0).

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target array of shape ``(n_samples,)`` with values in [0, 1].
        config: NEAT configuration.

    Returns:
        An evaluation callable suitable for :meth:`neat.Population.run`.
    """

    def eval_genomes(
        genomes: list[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ) -> None:
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            errors: list[float] = []
            for xi, yi in zip(X, y):
                output = net.activate(xi.tolist())
                pred = max(0.0, min(1.0, output[0]))
                errors.append(abs(pred - yi))
            mae = float(np.mean(errors))
            genome.fitness = 1.0 - mae

    return eval_genomes


def evolve(
    X: np.ndarray,
    y: np.ndarray,
    config: neat.Config,
    generations: int = 50,
) -> tuple[neat.DefaultGenome, neat.Population]:
    """Run NEAT evolution and return the best genome.

    Args:
        X: Feature matrix.
        y: Target array (congestion scores in [0, 1]).
        config: NEAT configuration.
        generations: Number of generations to evolve.

    Returns:
        A tuple ``(winner_genome, population)``.
    """
    logger.info("Starting NEAT evolution for %d generations …", generations)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    eval_fn = make_eval_function(X, y, config)
    winner = pop.run(eval_fn, generations)

    logger.info("Best genome fitness: %.4f", winner.fitness)
    return winner, pop


def save_model(genome: neat.DefaultGenome, path: str | Path) -> None:
    """Serialize a NEAT genome to a pickle file.

    Args:
        genome: The genome to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(genome, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> neat.DefaultGenome:
    """Load a serialized NEAT genome.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialised genome.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        genome = pickle.load(fh)  # noqa: S301
    logger.info("Model loaded from %s", path)
    return genome
