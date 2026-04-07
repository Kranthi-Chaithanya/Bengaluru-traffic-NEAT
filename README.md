# Bengaluru NEAT Traffic

AI-powered traffic congestion prediction and route optimisation for Bengaluru using **NEAT (NeuroEvolution of Augmenting Topologies)**.

## Overview

This project builds a traffic-routing system that:

1. **Models Bengaluru's road network** as a directed graph using OSMnx / NetworkX.
2. **Predicts congestion** on individual road segments with a NEAT-evolved neural network.
3. **Optimises routes** between origin–destination pairs by penalising congested and overloaded edges.
4. **Reduces overall network traffic** by spreading trips across alternate corridors.

The pipeline works end-to-end with **synthetic data** out of the box and can also ingest real traffic CSV files.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Road Network│────▶│ Feature Engineer │────▶│  NEAT Model  │
│  (OSMnx)    │     │  (per-edge)      │     │  (evolve /   │
└─────────────┘     └──────────────────┘     │   predict)   │
                                              └──────┬───────┘
                                                     │
                    ┌──────────────────┐              │
                    │ Route Optimiser  │◀─────────────┘
                    │  (assign trips,  │
                    │   reduce load)   │
                    └──────┬───────────┘
                           │
                    ┌──────▼───────────┐
                    │  Visualisation   │
                    │  (maps, charts)  │
                    └──────────────────┘
```

### Module overview

| Module | Purpose |
|---|---|
| `src/data_loader.py` | Load traffic CSVs, caching helpers |
| `src/graph_builder.py` | Download / build / enrich the road graph |
| `src/feature_engineering.py` | Per-edge feature vectors for the NN |
| `src/neat_model.py` | NEAT config loading, evolution loop, save/load |
| `src/predictor.py` | Inference wrapper around a trained genome |
| `src/router.py` | Congestion-aware routing & trip assignment |
| `src/simulator.py` | Synthetic traffic data generator |
| `src/visualize.py` | Congestion maps, route comparisons, metric charts |
| `src/main.py` | CLI entry-point |

---

## Installation

```bash
# Clone
git clone https://github.com/<you>/bengaluru-neat-traffic.git
cd bengaluru-neat-traffic

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Python 3.11+** is required.

---

## Usage

### Quick start (synthetic data)

```bash
python -m src.main --generations 30 --trip-count 20 --sample-edges 1500 --output-dir outputs
```

### Use the real Bengaluru graph

```bash
python -m src.main --use-real-data --generations 50 --trip-count 50 --output-dir outputs
```

The first run downloads the graph via OSMnx and caches it as `data/bengaluru_graph.pkl`.

### Supply a real traffic CSV

```bash
python -m src.main --traffic-csv data/my_traffic.csv --generations 40
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--generations` | 30 | NEAT evolution generations |
| `--trip-count` | 20 | Number of trips to route |
| `--sample-edges` | 1500 | Edges sampled for training |
| `--use-real-data` | off | Download real Bengaluru graph |
| `--traffic-csv` | – | Path to real traffic CSV |
| `--output-dir` | `outputs` | Output directory |
| `--neat-config` | `config/neat_config.ini` | NEAT config path |
| `--graph-nodes` | 50 | Nodes in sample graph |
| `--seed` | 42 | Random seed |
| `--verbose` | off | DEBUG-level logging |

---

## Input data format

The traffic CSV must contain these columns:

```csv
u,v,key,hour,day_of_week,rain_mm,observed_speed_kph
```

* `u`, `v`, `key` – OSM node IDs & edge key identifying the road segment.
* `hour` – hour of day (0–23).
* `day_of_week` – 0 = Monday … 6 = Sunday.
* `rain_mm` – rainfall in mm.
* `observed_speed_kph` – measured speed on that segment.

A sample file is included at `data/sample_traffic.csv`.

---

## How NEAT is used

[NEAT](https://neat-python.readthedocs.io/) evolves both the topology and the weights of a feed-forward neural network:

1. **Inputs (9 features per edge):** normalised road length, baseline speed, hour, day, rainfall, road-type code, historical load, edge-betweenness centrality, current occupancy.
2. **Output:** a single congestion score in [0, 1] (0 = free flow, 1 = gridlock).
3. **Fitness:** `1 − MAE` over the training set, so networks that predict congestion more accurately score higher.
4. **Evolution:** the population of genomes is evolved for *N* generations; the best genome is serialised and used for inference.

The NEAT config lives in `config/neat_config.ini` and can be tuned freely.

---

## How routing reduces congestion

The route optimiser goes beyond simple shortest-path:

1. **Edge cost** is a weighted sum of normalised distance, estimated travel time, *predicted* congestion, and an **overload penalty**.
2. Trips are assigned **sequentially**. After each trip is routed, the edges it uses receive an incremented trip count.
3. Later trips see a higher penalty on already-loaded edges, naturally **spreading traffic** across alternate corridors.
4. The `compare_baseline_vs_optimized` function quantifies improvement in max-load, mean-load, edge utilisation, and load standard deviation.

---

## Outputs

After a run, the `outputs/` directory contains:

* `winner_genome.pkl` – serialised NEAT winner genome
* `congestion_map.png` – graph coloured by predicted congestion
* `route_comparison.png` – side-by-side baseline vs optimised edge loads
* `metrics_chart.png` – bar chart of comparison metrics
* `metrics.json` – raw comparison numbers

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Limitations

* The synthetic data generator produces plausible but not calibrated traffic patterns.
* NEAT evolution on the full Bengaluru graph can be slow; sampling edges keeps training tractable.
* Edge-betweenness centrality is expensive on very large graphs.
* The overload-penalty model is a simple linear saturating function; real capacity models are more complex.
* Visualisations use spring layout for sample graphs and lat/lon for real OSMnx graphs.

---

## Future improvements

* Integrate live traffic feeds (Google Maps API, Ola/Uber data).
* Use time-dependent routing with evolving congestion predictions.
* Add multi-objective NEAT to balance travel time, distance, and emissions.
* Support multi-modal routing (bus, metro).
* Deploy as a REST API with FastAPI.
* Add interactive Folium / Leaflet maps.
* Benchmark against other ML baselines (GBM, GNN).

---

## License

MIT