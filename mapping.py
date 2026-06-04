from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import KDTree

from point import Point

def load_hull_model(file_path):
    """Loads an STL or OBJ mesh using trimesh."""
    hull_mesh = trimesh.load(file_path)
    return hull_mesh

def sample_hull_points(hull_mesh, num_points=500):
    """
    Samples num_points from the hull surface using trimesh's built-in method.
    Returns a (num_points x 3) numpy array.
    """
    # Ensure the mesh is watertight or workable
    # If the STL is huge, you might want to do hull_mesh.simplify() or so, if needed
    discretized_points = hull_mesh.sample(num_points)
    return discretized_points

def convert_to_point_objects(discretized_points, k=20):
    """
    Converts Nx3 points into a list of Point objects,
    building a KDTree to assign k-nearest neighbors.
    """
    points = [Point(x, y, z) for x, y, z in discretized_points]
    coords = np.array([[p.x, p.y, p.z] for p in points])
    tree = KDTree(coords)

    # Assign up to k neighbors for each point, excluding the point itself.
    for i, p in enumerate(points):
        query_count = min(k + 1, len(points))
        _, idxs = tree.query(coords[i], k=query_count)
        idxs = np.atleast_1d(idxs)
        p.neighbors = [int(idx) for idx in idxs if int(idx) != i][:k]
    return points

def visualize_discretized(points):
    """
    Plots the raw discretized points in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coords = np.array([[p.x, p.y, p.z] for p in points])
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='b', marker='o')
    ax.set_title('Sampled Hull Surface Points')
    plt.show()

def main(file_path=None, num_points=500, k=20, visualize=True):
    """
    1. Loads the STL hull
    2. Samples num_points from the mesh
    3. Converts them to Point objects
    4. Returns the final list of Point objects
    """
    if file_path is None:
        file_path = Path(__file__).with_name("hull_only.stl")
    hull_mesh = load_hull_model(file_path)

    discretized_points = sample_hull_points(hull_mesh, num_points=num_points)
    # Convert array -> GBNN-compatible Point objects
    point_objects = convert_to_point_objects(discretized_points, k=k)

    if visualize:
        visualize_discretized(point_objects)

    return point_objects

if __name__ == "__main__":
    main()
