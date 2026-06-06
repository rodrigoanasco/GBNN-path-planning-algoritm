# GBNN Coverage Path Planning for 3D Hulls

This project implements coverage path planning on sampled 3D hull surfaces using a Graph-Based Glasius Bio-Inspired Neural Network (GBNN).

The planner loads an STL or OBJ mesh, samples the surface into graph nodes, filters local neighborhoods by distance, identifies obstacle-like surface outliers, propagates neural activity across the graph, and generates an explicit coverage path. The resulting path can be visualized with Matplotlib or exported as a CSV file for downstream analysis.

## Features

- STL/OBJ mesh loading and surface sampling with `trimesh`.
- KDTree-based neighborhood initialization and radius-filtered graph construction.
- Local plane-fit obstacle detection for irregular surface points.
- GBNN activity propagation for coverage-oriented node selection.
- Turn-aware next-node scoring to encourage smoother path transitions.
- Dead-zone recovery using graph search to reach the nearest remaining uncovered node.
- Optional 3D animation and CSV export of the generated path.

## Project Structure

| File | Description |
| --- | --- |
| `gbnn.py` | Main planner, command-line interface, animation, path export, and coverage summary. |
| `mapping.py` | Mesh loading, surface sampling, point conversion, and initial KDTree neighbor assignment. |
| `point.py` | Graph node representation, including coordinates, activity, type, and neighbors. |
| `hull_only.stl`, `Cargoship.stl` | Example hull meshes used for planning experiments. |
| `Gbnn_Individual_testing/` | Prototype notebooks used during component development and testing. |

## Installation

Install the required Python dependencies:

```bash
pip install numpy scipy matplotlib trimesh
```

If STL loading is not available in your environment, install the optional STL dependency as well:

```bash
pip install numpy-stl
```

## Usage

Run the planner from the project directory:

```bash
python gbnn.py
```

The default mesh is `hull_only.stl`, located next to `gbnn.py`.

To run from the repository root:

```bash
python GBNN-path-planning-algoritm/gbnn.py
```

For a non-interactive run that exports the generated path:

```bash
python GBNN-path-planning-algoritm/gbnn.py --no-animation --export GBNN-path-planning-algoritm/coverage_path.csv
```

Example console output:

```text
Coverage: 120/120 (100.0%), path nodes: 124, completed: True
```

## Command-Line Options

| Option | Description |
| --- | --- |
| `--mesh` | STL/OBJ mesh path. Defaults to `hull_only.stl`. |
| `--num-points` | Number of sampled hull surface points. |
| `--k-neighbors` | Initial KDTree neighbor count for each sampled point. |
| `--radius` | Physical radius used to keep valid graph edges. |
| `--max-neighbors` | Maximum number of local GBNN neighbors. Defaults to `--k-neighbors`. |
| `--obstacle-threshold` | Plane-distance threshold for obstacle classification. Use `0` to disable obstacle detection. |
| `--max-steps` | Maximum number of path-planning steps before stopping. |
| `--activity-iterations` | Number of neural activity relaxation iterations per planning step. |
| `--start-index` | Explicit start node index. |
| `--random-start` | Selects a random non-obstacle start node. |
| `--seed` | Random seed used with `--random-start`. |
| `--allow-reposition` | Allows repositioning to a disconnected uncovered component if necessary. |
| `--map-preview` | Displays the sampled hull points before planning. |
| `--no-animation` | Runs the planner without opening the Matplotlib animation window. |
| `--export` | Writes the generated path to a CSV file with step, node index, and XYZ coordinates. |

## Algorithm Overview

Each sampled surface point is treated as a graph node and GBNN neuron. Uncovered nodes receive positive external input, obstacle nodes receive inhibitory input, and cleaned nodes receive zero external input. Positive activity propagates through weighted graph edges, creating an activity landscape that guides coverage decisions.

During normal traversal, the planner selects among uncovered neighboring nodes using a score that combines neural activity, turn smoothness, uncovered-node priority, and edge distance. When the current node has no uncovered neighbors, the planner performs a breadth-first search to the nearest remaining uncovered node and traverses that route step by step. This dead-zone handling keeps every transit node in the returned path instead of teleporting between disconnected decisions.

## Outputs

- Console summary with coverage ratio, path length, completion status, and stopping condition.
- Optional Matplotlib animation showing the robot marker and covered nodes over time.
- Optional CSV export containing `step`, `node_index`, `x`, `y`, and `z` columns.

## Notes

The graph radius, obstacle threshold, and number of sampled points should be tuned to the scale and density of the selected hull mesh. Very sparse samples or overly small radius values may create disconnected graph components, while overly large radius values may produce unrealistic shortcuts across the surface.

