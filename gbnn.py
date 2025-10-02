import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from collections import deque
from scipy.spatial import KDTree

from mapping import main as mapping_main
from point import Point

# GBNN Constants
E = 100.0
NEG_E = -100.0
CLEANED = 0.0
MU = 0.5
C = 1.0

def distance(point1, point2):
    return np.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2 + (point2.z - point1.z)**2)

def is_neighbour(tree_nodes, R=300):
    """
    Instead of BFS, we simply check each node's precomputed neighbors,
    filtering out those beyond distance R.
    """
    filtered_map = {}
    for i, node in enumerate(tree_nodes):
        valid_neighbors = []
        for nbr_idx in node.neighbors:
            if distance(node, tree_nodes[nbr_idx]) <= R:
                valid_neighbors.append(nbr_idx)
        filtered_map[i] = valid_neighbors
    return filtered_map

def fit_plane_least_squares(points):
    A = np.array([[p.x, p.y, 1] for p in points])
    b = np.array([-p.z for p in points])
    normal_vector, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    n = np.array([normal_vector[0], normal_vector[1], -1])
    n /= np.linalg.norm(n)
    return n

def point_plane_distance(point, normal_vector):
    numerator = abs(np.dot(normal_vector, np.array([point.x, point.y, point.z])) + 1)
    denominator = np.linalg.norm(normal_vector)
    return numerator / denominator

def detect_obstacles(tree_nodes, filtered_neighbors, d_threshold=10.0):
    obstacles = []
    for i, node in enumerate(tree_nodes):
        neigh = filtered_neighbors[i]
        if len(neigh) < 3:
            continue
        neighbor_points = [tree_nodes[idx] for idx in neigh]
        normal_vector = fit_plane_least_squares(neighbor_points)
        d_i = point_plane_distance(node, normal_vector)
        if d_i > d_threshold:
            obstacles.append(i)
    return obstacles

def activation_function(x):
    if x < -1:
        return -1
    elif x > 1:
        return 1
    return x

def compute_weight(point_i, point_j, mu=0.5):
    dist = distance(point_i, point_j)
    return mu / dist if dist > 0 else 0

def compute_external_input(point, E=100.0):
    if point.type == CLEANED:
        return 0
    elif point.type == NEG_E:
        return -E
    return E

def update_activities(tree_nodes, filtered_neighbors, mu=0.5, E=100.0):
    new_activities = {}
    for i, p in enumerate(tree_nodes):
        sum_term = 0
        for j in filtered_neighbors[i]:
            neighbor = tree_nodes[j]
            w_ij = compute_weight(p, neighbor, mu)
            sum_term += w_ij * max(neighbor.activity, 0)
        I_i = compute_external_input(p, E)
        new_activities[i] = activation_function(sum_term + I_i)
    for i, val in new_activities.items():
        tree_nodes[i].activity = val

def get_next_position(robot_idx, tree_nodes, filtered_neighbors):
    neighbors = filtered_neighbors[robot_idx]
    if not neighbors:
        return robot_idx
    return max(neighbors, key=lambda idx: tree_nodes[idx].activity)

# --------------------
# Dead-Zone BFS Escape
# --------------------
def bfs_path(start, goal, filtered_neighbors):
    """
    Returns a path [start -> ... -> goal] if one exists, else [].
    Simple BFS over the adjacency from filtered_neighbors.
    """
    queue = deque([[start]])
    visited = set([start])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for nbr in filtered_neighbors.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                new_path = list(path)
                new_path.append(nbr)
                queue.append(new_path)
    return []

def find_nearest_uncleaned(robot_idx, tree_nodes, filtered_neighbors):
    """Find the nearest uncleaned (blue) point to the robot."""
    uncleaned = [i for i, p in enumerate(tree_nodes) if p.type == E]
    if not uncleaned:
        return None
    # Pick the physically closest uncleaned
    coords = np.array([[p.x, p.y, p.z] for p in tree_nodes])
    dists = [np.linalg.norm(coords[robot_idx] - coords[u]) for u in uncleaned]
    return uncleaned[np.argmin(dists)]

# --------------------
# Animation with Escape
# --------------------
def update_animation(frame, scatter, tree_nodes):
    colors = []
    for p in tree_nodes:
        if p.type == NEG_E:
            colors.append("red")
        elif p.type == 0.0:  # CLEANED
            colors.append("green")
        else:
            val = (p.activity + 1)/2
            colors.append((0,0,val)) 
    scatter.set_color(colors)
    return scatter,

def run_animation(tree_nodes, filtered_neighbors, max_steps=5000):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    coords = np.array([[p.x, p.y, p.z] for p in tree_nodes])
    scatter = ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=50)

    # Pick random start
    available_points = [i for i,p in enumerate(tree_nodes) if p.type != NEG_E]
    robot_idx = random.choice(available_points)

    stuck_counter = 0
    prev_idx = None

    def attempt_dead_zone_escape(current_idx):
        """Try BFS jump to nearest uncleaned point."""
        nearest = find_nearest_uncleaned(current_idx, tree_nodes, filtered_neighbors)
        if nearest is None:
            return current_idx  # No uncleaned points left
        path = bfs_path(current_idx, nearest, filtered_neighbors)
        if len(path) > 1:
            # Mark the path as cleaned, and jump to end
            for node_i in path[1:]:
                tree_nodes[node_i].type = CLEANED
            return path[-1]
        return current_idx

    def update(frame):
        nonlocal robot_idx, stuck_counter, prev_idx
        # Mark current node as cleaned
        tree_nodes[robot_idx].type = CLEANED
        
        # Update activities
        update_activities(tree_nodes, filtered_neighbors)
        
        # GBNN decides next move
        next_idx = get_next_position(robot_idx, tree_nodes, filtered_neighbors)
        
        # Check if stuck or toggling
        if next_idx == robot_idx or (prev_idx is not None and next_idx == prev_idx):
            stuck_counter += 1
        else:
            stuck_counter = 0

        # Dead-zone BFS escape if stuck too long
        if stuck_counter > 5:
            next_idx = attempt_dead_zone_escape(robot_idx)
            stuck_counter = 0

        prev_idx = robot_idx
        robot_idx = next_idx
        return update_animation(frame, scatter, tree_nodes)

    ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=500, repeat=False)
    plt.show()
    return ani

# --------------
# Main Execution
# --------------
if __name__ == "__main__":
    # 1) Read & sample from your hull STL in mapping_main()
    point_objects = mapping_main()

    # 2) Filter neighbors with distance check
    filtered_neighbors = is_neighbour(point_objects, R=300)

    # 3) Detect obstacles
    obstacle_indices = detect_obstacles(point_objects, filtered_neighbors, d_threshold=10.0)
    for idx in obstacle_indices:
        point_objects[idx].type = NEG_E

    # 4) Animate coverage with dead-zone BFS escape
    run_animation(point_objects, filtered_neighbors, max_steps=5000)
