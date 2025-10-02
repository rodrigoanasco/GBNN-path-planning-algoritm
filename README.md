# GBNN Coverage Path Planning on 3D Hulls

Neural-inspired coverage path planning for a ship hull using a **Graph‑Based Neural Network (GBNN)**.  
Given a 3D hull mesh (STL/OBJ), the pipeline samples surface points, builds a neighbor graph (via KD‑Tree with distance filtering), detects obstacles by plane fitting, and simulates coverage with an animated walker guided by neural activities.

https://github.com/ (add repo link when published)

---

## ✨ What’s Inside

- **`mapping.py`** – Loads a hull mesh with `trimesh`, samples surface points, builds KD‑Tree neighbors, and (optionally) 3D‑visualizes samples.
- **`point.py`** – Lightweight `Point` structure holding coordinates, activity, type (`E`, `NEG_E`, `CLEANED`), and dynamic neighbors.
- **`gbnn.py`** – Core pipeline:
  - Neighbor filtering by radius (`is_neighbour`)  
  - Obstacle detection via least‑squares plane fit and point‑to‑plane distance  
  - Neural activity update rule (weights ∝ 1/distance, external inputs by point type)  
  - Greedy next‑step with **dead‑zone BFS escape** and **matplotlib** animation
- **Notebooks** – Focused experiments for each component:
  - `Neural_Activity.ipynb` – activity update & obstacle logic
  - `is_neighbour_func.ipynb` – KD‑Tree vs. filtered neighbors & visualization
  - `complete_implementation.ipynb` – end‑to‑end test harness

> The notebooks were used to validate features before integrating the full script pipeline.

---

## 🧰 Requirements

- Python 3.10+
- Recommended: a virtual environment (venv or conda)

Install dependencies:
```bash
pip install numpy scipy matplotlib trimesh geomdl
```
> On some systems `trimesh` may fetch extra readers (e.g., for OBJ). If STL loading fails, install: `pip install numpy-stl`.

---

## 📁 Project Layout

```
.
├─ gbnn.py
├─ mapping.py
├─ point.py
├─ Neural_Activity.ipynb
├─ is_neighbour_func.ipynb
├─ complete_implementation.ipynb
└─ hull_only.stl              # <- provide your hull mesh here
```

---

## 🚀 Quick Start

1) Place your hull mesh at the project root (default name expected by the code is **`hull_only.stl`**).  
2) Run the end‑to‑end simulation:
```bash
python gbnn.py
```
This will:
- Load and sample the hull surface (default 500 points)
- Build KD‑Tree neighbors and **filter by radius** (default `R=300` in code units)
- Detect obstacles (red) using **least‑squares plane fitting** and a distance threshold
- Animate coverage: the agent starts on a valid node, cleans as it moves (green), avoids obstacles (red), and uses BFS to escape dead‑zones.

> If you want to preview just the sampled points, run:
```bash
python mapping.py
```

---

## ⚙️ Key Parameters (edit in `gbnn.py` / `mapping.py`)

- Sampling density: `sample_hull_points(..., num_points=500)`
- Neighbor pre‑selection (KD‑Tree): `k=20`
- Radius filter: `is_neighbour(..., R=300)`
- Obstacle distance threshold: `detect_obstacles(..., d_threshold=10.0)`
- Neural constants: `E=100.0`, `NEG_E=-100.0`, `CLEANED=0.0`, `MU=0.5`, `C=1.0`
- Animation steps: `run_animation(..., max_steps=5000)`

> Units: Keep `R` and `d_threshold` consistent with your STL/OBJ units (e.g., millimeters, inches, meters).

---

## 🧠 Method Highlights

- **Graph construction**: KD‑Tree k‑NN → radius‑filtered adjacency for physical feasibility.  
- **Obstacle detection**: Fit plane to local neighbors; classify a point as obstacle if its **point‑to‑plane distance** exceeds a threshold.  
- **Neural update**: Activity at node *i* is driven by neighbors’ positive activity (ReLU) and an external input based on the node type (uncleaned/obstacle/cleaned). Weights decrease with Euclidean distance.  
- **Path selection**: Greedy move to neighbor with highest activity; if oscillating or stuck, perform **BFS jump** to nearest uncleaned node and continue.

---

## 📊 Visualizations

- 3D scatter of sampled points (`mapping.visualize_discretized`)  
- Live **matplotlib** animation of coverage in `gbnn.run_animation`  
- In notebooks: comparisons of KD‑Tree vs filtered graph, activity distributions, and path demos.

---

## 🧪 Notebooks (Optional)

Run the notebooks to experiment with individual components:
- Tune radius thresholds and observe neighbor graph changes.
- Vary `MU`, `E`, and `C` to study stability and coverage speed.
- Validate plane‑fit obstacle detection on synthetic data before real hulls.

---

## 🧩 Troubleshooting

- **Blank or slow plots**: Use a non‑interactive backend or reduce `num_points`.  
- **STL fails to load**: Confirm a watertight or valid mesh; try `pip install numpy-stl`.  
- **Scale issues**: If the agent “sees” everything as neighbors, your `R` is too large for the mesh scale. If it sees nothing, `R` is too small.  
- **No progress / stuck**: Lower `d_threshold` (more obstacles) or increase BFS escape frequency.

---

## ✅ Roadmap

- Adaptive radius per region (density‑aware neighbors)  
- Multi‑agent coverage & collision avoidance  
- Export visited sequence as waypoints (CSV/ROS msg)  
- Mesh decimation & uniform surface sampling

---

## 📄 License

MIT (add LICENSE file if publishing).

---

## 🙌 Acknowledgements

- Built with: **NumPy**, **SciPy**, **trimesh**, **matplotlib**, **geomdl (NURBS)**, and **KDTree**.  
- Inspired by neural coverage ideas (GBNN) for robotics on complex 3D surfaces.
