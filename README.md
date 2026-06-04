# GBNN Coverage Path Planning on 3D Hulls

Graph-based Glasius Bio-Inspired Neural Network (GBNN) coverage path planning for a sampled 3D ship hull surface.

The planner loads an STL/OBJ mesh, samples surface points, builds a local radius-neighborhood graph, detects obstacle-like outliers, updates neural activity across the graph, and generates an explicit coverage path. Animation and CSV export are built on top of that generated path.

## Files

- `mapping.py` loads the hull mesh, samples surface points, and assigns initial KDTree neighbors.
- `point.py` defines each graph node/neuron: position, activity, type, and neighbors.
- `gbnn.py` contains the full planner:
  - radius-neighborhood graph construction
  - local plane-fit obstacle detection
  - GBNN activity propagation
  - next-node scoring using neural activity and turn preference
  - dead-zone traversal to the nearest remaining uncovered node
  - coverage summary, animation, and CSV export
- `Gbnn_Individual_testing/` contains prototype notebooks used while developing the components.

## Install

```bash
pip install numpy scipy matplotlib trimesh
```

If STL loading fails on your machine, also try:

```bash
pip install numpy-stl
```

## Run

From this directory:

```bash
python gbnn.py
```

From the repository root:

```bash
python GBNN-path-planning-algoritm/gbnn.py
```

Useful non-interactive run:

```bash
python GBNN-path-planning-algoritm/gbnn.py --no-animation --export GBNN-path-planning-algoritm/coverage_path.csv
```

The command prints a summary such as:

```text
Coverage: 120/120 (100.0%), path nodes: 124, completed: True
```

## Main Parameters

- `--mesh`: STL/OBJ path. Defaults to `hull_only.stl` next to `gbnn.py`.
- `--num-points`: number of hull surface samples.
- `--k-neighbors`: initial KDTree neighbor count.
- `--radius`: physical neighborhood radius for valid graph edges.
- `--max-neighbors`: cap on local GBNN neighbors. Defaults to `--k-neighbors`.
- `--obstacle-threshold`: local plane-distance threshold for obstacle classification. Use `0` to disable.
- `--max-steps`: maximum allowed path steps.
- `--activity-iterations`: neural relaxations per planning step.
- `--start-index`: explicit start node.
- `--random-start --seed 1`: reproducible random start.
- `--allow-reposition`: permits a jump to a disconnected uncovered component.
- `--map-preview`: shows sampled hull points before planning.
- `--no-animation`: runs planning without opening a matplotlib window.
- `--export path.csv`: writes step, node index, and XYZ coordinates.

## Algorithm Notes

The planner treats every sampled point as a neuron. Uncovered nodes receive positive external input, obstacles receive inhibitory input, and cleaned nodes receive zero external input. Positive neighbor activity propagates through weighted graph edges. At each normal move, the robot chooses among uncovered neighboring nodes using neural activity plus a turn-smoothness preference.

When no uncovered neighbor is available, the planner performs a graph BFS to the nearest remaining uncovered node and traverses that route step by step. This is dead-zone handling, not teleportation; every transit node is part of the returned path.

