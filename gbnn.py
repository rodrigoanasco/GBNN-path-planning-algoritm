import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from mapping import main as mapping_main
from point import Point

# Point state values.
E = 100.0
NEG_E = -100.0
CLEANED = 0.0

# GBNN tuning constants.
MU = 0.5
C = 1.0
EXCITATORY_INPUT = 0.65
INHIBITORY_INPUT = -1.0


@dataclass
class PlanningResult:
    path: list[int]
    covered_count: int
    coverable_count: int
    remaining_uncovered: list[int]
    completed: bool
    max_steps_hit: bool

    @property
    def coverage_ratio(self):
        if self.coverable_count == 0:
            return 1.0
        return self.covered_count / self.coverable_count


def clone_points(points):
    clones = [Point(p.x, p.y, p.z, activity=p.activity, point_type=p.type) for p in points]
    for clone, original in zip(clones, points):
        clone.neighbors = list(original.neighbors)
    return clones


def point_coords(point):
    return np.array([point.x, point.y, point.z], dtype=float)


def distance(point1, point2):
    return np.linalg.norm(point_coords(point2) - point_coords(point1))


def is_obstacle(point):
    return point.type == NEG_E


def is_uncovered(point):
    return point.type == E


def is_coverable(point):
    return not is_obstacle(point)


def mark_cleaned(point):
    if is_uncovered(point):
        point.type = CLEANED


def is_neighbour(tree_nodes, R=300, max_neighbors=None):
    """
    Build an undirected radius-neighborhood graph for the GBNN.

    A KDTree is used for radius queries. The resulting graph excludes self
    connections and is made symmetric so path traversal is physically coherent.
    """
    if not tree_nodes:
        return {}

    coords = np.array([point_coords(p) for p in tree_nodes])
    tree = KDTree(coords)
    neighbor_sets = {i: set() for i in range(len(tree_nodes))}

    for i, coord in enumerate(coords):
        candidates = [idx for idx in tree.query_ball_point(coord, r=R) if idx != i]
        candidates.sort(key=lambda idx: np.linalg.norm(coords[i] - coords[idx]))
        if max_neighbors is not None:
            candidates = candidates[:max_neighbors]

        for idx in candidates:
            neighbor_sets[i].add(int(idx))
            neighbor_sets[int(idx)].add(i)

    return {
        i: sorted(neighbors, key=lambda idx: np.linalg.norm(coords[i] - coords[idx]))
        for i, neighbors in neighbor_sets.items()
    }


def fit_plane_least_squares(points):
    """
    Fit a 3D plane by PCA and return (normal, offset).

    The plane is normal . x + offset = 0. PCA avoids assuming the hull can be
    represented as z = f(x, y), which is fragile for vertical hull regions.
    """
    coords = np.array([point_coords(p) for p in points], dtype=float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0.0, 0.0, 1.0]), -centroid[2]
    normal = normal / norm
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def point_plane_distance(point, normal_vector, offset=0.0):
    numerator = abs(np.dot(normal_vector, point_coords(point)) + offset)
    denominator = np.linalg.norm(normal_vector)
    return numerator / denominator if denominator > 0 else 0.0


def detect_obstacles(tree_nodes, filtered_neighbors, d_threshold=10.0):
    obstacles = []
    for i, node in enumerate(tree_nodes):
        neigh = filtered_neighbors.get(i, [])
        if len(neigh) < 3:
            continue

        neighbor_points = [tree_nodes[idx] for idx in neigh]
        normal_vector, offset = fit_plane_least_squares(neighbor_points)
        d_i = point_plane_distance(node, normal_vector, offset)
        if d_i > d_threshold:
            obstacles.append(i)

    return obstacles


def activation_function(x):
    if x < -1:
        return -1
    if x > 1:
        return 1
    return x


def compute_weight(point_i, point_j, mu=MU, radius=None, gamma=1.0):
    dist = distance(point_i, point_j)
    return compute_weight_from_distance(dist, mu=mu, radius=radius, gamma=gamma)


def compute_weight_from_distance(dist, mu=MU, radius=None, gamma=1.0):
    if dist <= 0:
        return 0.0
    if radius is not None and radius > 0:
        return mu * np.exp(-gamma * dist / radius)
    return mu / (1.0 + dist)


def build_edge_cache(tree_nodes, filtered_neighbors, radius=None, mu=MU):
    coords = np.array([point_coords(p) for p in tree_nodes])
    weighted_neighbors = {}
    edge_distances = {}

    for i, neighbors in filtered_neighbors.items():
        weighted_neighbors[i] = []
        for j in neighbors:
            dist = float(np.linalg.norm(coords[j] - coords[i]))
            edge_distances[(i, j)] = dist
            weighted_neighbors[i].append(
                (j, compute_weight_from_distance(dist, mu=mu, radius=radius))
            )

    return weighted_neighbors, edge_distances


def compute_external_input(
    point,
    excitatory_input=EXCITATORY_INPUT,
    inhibitory_input=INHIBITORY_INPUT,
):
    if point.type == CLEANED:
        return 0.0
    if point.type == NEG_E:
        return inhibitory_input
    return excitatory_input


def update_activities(
    tree_nodes,
    filtered_neighbors,
    mu=MU,
    excitatory_input=EXCITATORY_INPUT,
    inhibitory_input=INHIBITORY_INPUT,
    radius=None,
    weighted_neighbors=None,
):
    new_activities = {}

    for i, point in enumerate(tree_nodes):
        if is_obstacle(point):
            new_activities[i] = -1.0
            continue

        sum_term = 0.0
        if weighted_neighbors is None:
            neighbors = [
                (j, compute_weight(point, tree_nodes[j], mu=mu, radius=radius))
                for j in filtered_neighbors.get(i, [])
            ]
        else:
            neighbors = weighted_neighbors.get(i, [])

        for j, w_ij in neighbors:
            neighbor = tree_nodes[j]
            sum_term += w_ij * max(neighbor.activity, 0.0)

        I_i = compute_external_input(
            point,
            excitatory_input=excitatory_input,
            inhibitory_input=inhibitory_input,
        )
        new_activities[i] = activation_function(sum_term + I_i)

    for i, val in new_activities.items():
        tree_nodes[i].activity = val


def relax_activities(tree_nodes, filtered_neighbors, iterations=3, **kwargs):
    for _ in range(max(1, iterations)):
        update_activities(tree_nodes, filtered_neighbors, **kwargs)


def valid_neighbors(node_idx, tree_nodes, filtered_neighbors):
    return [
        idx
        for idx in filtered_neighbors.get(node_idx, [])
        if idx != node_idx and not is_obstacle(tree_nodes[idx])
    ]


def compute_turn_score(candidate_idx, current_idx, prev_idx, tree_nodes):
    if prev_idx is None:
        return 1.0

    prev_pos = point_coords(tree_nodes[prev_idx])
    current_pos = point_coords(tree_nodes[current_idx])
    candidate_pos = point_coords(tree_nodes[candidate_idx])

    incoming = current_pos - prev_pos
    outgoing = candidate_pos - current_pos
    incoming_norm = np.linalg.norm(incoming)
    outgoing_norm = np.linalg.norm(outgoing)

    if incoming_norm == 0 or outgoing_norm == 0:
        return 0.0

    cos_theta = np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return 1.0 - (angle / np.pi)


def choose_next_location(
    current_idx,
    prev_idx,
    tree_nodes,
    filtered_neighbors,
    candidates=None,
    turn_weight=C,
    uncovered_bonus=0.35,
    cleaned_penalty=0.25,
    backtrack_penalty=0.25,
    distance_weight=0.05,
    radius=None,
    edge_distances=None,
):
    if candidates is None:
        candidates = valid_neighbors(current_idx, tree_nodes, filtered_neighbors)
    else:
        candidates = [
            idx for idx in candidates if idx != current_idx and not is_obstacle(tree_nodes[idx])
        ]

    if not candidates:
        return None

    current_point = tree_nodes[current_idx]
    scored_candidates = []
    for idx in candidates:
        candidate = tree_nodes[idx]
        candidate_distance = (
            edge_distances.get((current_idx, idx))
            if edge_distances is not None
            else None
        )
        if candidate_distance is None:
            candidate_distance = distance(current_point, candidate)

        score = candidate.activity
        score += turn_weight * compute_turn_score(idx, current_idx, prev_idx, tree_nodes)

        if is_uncovered(candidate):
            score += uncovered_bonus
        elif candidate.type == CLEANED:
            score -= cleaned_penalty

        if prev_idx is not None and idx == prev_idx and len(candidates) > 1:
            score -= backtrack_penalty

        if radius is not None and radius > 0:
            score -= distance_weight * (candidate_distance / radius)

        scored_candidates.append((score, is_uncovered(candidate), -candidate_distance, idx))

    return max(scored_candidates)[-1]


def bfs_path(start, goal, filtered_neighbors, tree_nodes=None):
    """
    Return a topological path [start, ..., goal], avoiding obstacles when points
    are provided. Returns [] if no path exists.
    """
    queue = deque([start])
    parents = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            break

        if tree_nodes is None:
            neighbors = filtered_neighbors.get(node, [])
        else:
            neighbors = valid_neighbors(node, tree_nodes, filtered_neighbors)

        for neighbor in neighbors:
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)

    if goal not in parents:
        return []

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parents[node]
    return list(reversed(path))


def bfs_path_to_nearest_uncovered(start, tree_nodes, filtered_neighbors):
    queue = deque([start])
    parents = {start: None}
    target = None

    while queue:
        node = queue.popleft()
        if node != start and is_uncovered(tree_nodes[node]):
            target = node
            break

        for neighbor in valid_neighbors(node, tree_nodes, filtered_neighbors):
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)

    if target is None:
        return []

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = parents[node]
    return list(reversed(path))


def uncovered_indices(tree_nodes):
    return [i for i, point in enumerate(tree_nodes) if is_uncovered(point)]


def coverage_counts(tree_nodes):
    coverable = [point for point in tree_nodes if is_coverable(point)]
    covered = [point for point in coverable if point.type == CLEANED]
    return len(covered), len(coverable)


def plan_coverage_path(
    tree_nodes,
    filtered_neighbors,
    start_idx=None,
    max_steps=None,
    radius=None,
    activity_iterations=3,
    random_start=False,
    seed=None,
    allow_reposition=False,
):
    """
    Generate a complete coverage path over the reachable non-obstacle graph.

    Normal moves use the GBNN activity landscape and direction preference. When
    the robot reaches a dead zone, BFS supplies a transit path to the nearest
    remaining uncovered node without teleporting.
    """
    coverable_indices = [i for i, point in enumerate(tree_nodes) if is_coverable(point)]
    if not coverable_indices:
        return PlanningResult([], 0, 0, [], True, False)

    if seed is not None:
        random.seed(seed)

    if start_idx is None:
        start_idx = random.choice(coverable_indices) if random_start else coverable_indices[0]

    if start_idx not in coverable_indices:
        raise ValueError("start_idx must point to a non-obstacle node")

    if max_steps is None:
        max_steps = max(1, len(coverable_indices) * 20)

    for point in tree_nodes:
        point.activity = 0.0

    weighted_neighbors, edge_distances = build_edge_cache(
        tree_nodes,
        filtered_neighbors,
        radius=radius,
    )

    current_idx = start_idx
    prev_idx = None
    path = [current_idx]
    mark_cleaned(tree_nodes[current_idx])
    max_steps_hit = False

    while uncovered_indices(tree_nodes):
        if len(path) - 1 >= max_steps:
            max_steps_hit = True
            break

        relax_activities(
            tree_nodes,
            filtered_neighbors,
            iterations=activity_iterations,
            radius=radius,
            weighted_neighbors=weighted_neighbors,
        )

        neighbors = valid_neighbors(current_idx, tree_nodes, filtered_neighbors)
        uncovered_neighbors = [idx for idx in neighbors if is_uncovered(tree_nodes[idx])]

        if uncovered_neighbors:
            next_idx = choose_next_location(
                current_idx,
                prev_idx,
                tree_nodes,
                filtered_neighbors,
                candidates=uncovered_neighbors,
                radius=radius,
                edge_distances=edge_distances,
            )
        else:
            transit_path = bfs_path_to_nearest_uncovered(
                current_idx,
                tree_nodes,
                filtered_neighbors,
            )

            if len(transit_path) > 1:
                for transit_idx in transit_path[1:]:
                    if len(path) - 1 >= max_steps:
                        max_steps_hit = True
                        break
                    prev_idx, current_idx = current_idx, transit_idx
                    path.append(current_idx)
                    mark_cleaned(tree_nodes[current_idx])
                continue

            if not allow_reposition:
                break

            remaining = uncovered_indices(tree_nodes)
            if not remaining:
                break
            next_idx = min(
                remaining,
                key=lambda idx: distance(tree_nodes[current_idx], tree_nodes[idx]),
            )
            prev_idx = None

        if next_idx is None:
            break

        prev_idx, current_idx = current_idx, next_idx
        path.append(current_idx)
        mark_cleaned(tree_nodes[current_idx])

    covered_count, coverable_count = coverage_counts(tree_nodes)
    remaining_uncovered = uncovered_indices(tree_nodes)
    return PlanningResult(
        path=path,
        covered_count=covered_count,
        coverable_count=coverable_count,
        remaining_uncovered=remaining_uncovered,
        completed=len(remaining_uncovered) == 0,
        max_steps_hit=max_steps_hit,
    )


def colors_for_nodes(tree_nodes):
    colors = []
    for point in tree_nodes:
        if point.type == NEG_E:
            colors.append("red")
        elif point.type == CLEANED:
            colors.append("green")
        else:
            val = (point.activity + 1.0) / 2.0
            colors.append((0.0, 0.0, val))
    return colors


def update_animation(frame, scatter, robot_marker, path_line, display_nodes, path):
    frame = min(frame, len(path) - 1)
    current_idx = path[frame]

    for idx in path[: frame + 1]:
        mark_cleaned(display_nodes[idx])

    coords = np.array([point_coords(p) for p in display_nodes])
    scatter.set_color(colors_for_nodes(display_nodes))

    current = coords[current_idx]
    robot_marker._offsets3d = ([current[0]], [current[1]], [current[2]])

    path_coords = coords[path[: frame + 1]]
    path_line.set_data(path_coords[:, 0], path_coords[:, 1])
    path_line.set_3d_properties(path_coords[:, 2])
    return scatter, robot_marker, path_line


def animate_path(tree_nodes, path, interval=250):
    display_nodes = clone_points(tree_nodes)
    coords = np.array([point_coords(p) for p in display_nodes])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=35)

    start = coords[path[0]] if path else coords[0]
    robot_marker = ax.scatter(
        [start[0]],
        [start[1]],
        [start[2]],
        s=120,
        c="black",
        marker="o",
        depthshade=False,
    )
    path_line, = ax.plot([], [], [], color="black", linewidth=2, alpha=0.8)

    ax.set_title("GBNN Coverage Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    scatter.set_color(colors_for_nodes(display_nodes))

    frames = max(1, len(path))
    ani = animation.FuncAnimation(
        fig,
        update_animation,
        frames=frames,
        fargs=(scatter, robot_marker, path_line, display_nodes, path),
        interval=interval,
        repeat=False,
    )
    plt.show()
    return ani


def run_animation(
    tree_nodes,
    filtered_neighbors,
    max_steps=5000,
    start_idx=None,
    radius=None,
    activity_iterations=3,
    random_start=False,
    seed=None,
):
    planning_nodes = clone_points(tree_nodes)
    result = plan_coverage_path(
        planning_nodes,
        filtered_neighbors,
        start_idx=start_idx,
        max_steps=max_steps,
        radius=radius,
        activity_iterations=activity_iterations,
        random_start=random_start,
        seed=seed,
    )
    ani = animate_path(tree_nodes, result.path)
    return result, ani


def export_path_csv(file_path, tree_nodes, path):
    file_path = Path(file_path)
    if file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "node_index", "x", "y", "z"])
        for step, node_idx in enumerate(path):
            point = tree_nodes[node_idx]
            writer.writerow([step, node_idx, point.x, point.y, point.z])


def parse_args():
    parser = argparse.ArgumentParser(description="GBNN coverage path planning on a hull mesh.")
    parser.add_argument("--mesh", default=str(Path(__file__).with_name("hull_only.stl")))
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--radius", type=float, default=300.0)
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help="Caps local GBNN neighbors; defaults to --k-neighbors.",
    )
    parser.add_argument("--obstacle-threshold", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--activity-iterations", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--map-preview", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--allow-reposition", action="store_true")
    parser.add_argument("--export", default=None)
    return parser.parse_args()


def print_summary(result):
    print(
        "Coverage: "
        f"{result.covered_count}/{result.coverable_count} "
        f"({result.coverage_ratio:.1%}), "
        f"path nodes: {len(result.path)}, "
        f"completed: {result.completed}"
    )
    if result.max_steps_hit:
        print("Stopped because max_steps was reached.")
    if result.remaining_uncovered:
        print(f"Remaining uncovered nodes: {len(result.remaining_uncovered)}")


def main():
    args = parse_args()

    point_objects = mapping_main(
        file_path=args.mesh,
        num_points=args.num_points,
        k=args.k_neighbors,
        visualize=args.map_preview,
    )

    max_neighbors = args.max_neighbors if args.max_neighbors is not None else args.k_neighbors
    filtered_neighbors = is_neighbour(
        point_objects,
        R=args.radius,
        max_neighbors=max_neighbors,
    )

    if args.obstacle_threshold > 0:
        obstacle_indices = detect_obstacles(
            point_objects,
            filtered_neighbors,
            d_threshold=args.obstacle_threshold,
        )
        for idx in obstacle_indices:
            point_objects[idx].type = NEG_E

    planning_nodes = clone_points(point_objects)
    result = plan_coverage_path(
        planning_nodes,
        filtered_neighbors,
        start_idx=args.start_index,
        max_steps=args.max_steps,
        radius=args.radius,
        activity_iterations=args.activity_iterations,
        random_start=args.random_start,
        seed=args.seed,
        allow_reposition=args.allow_reposition,
    )

    print_summary(result)

    if args.export:
        export_path_csv(args.export, point_objects, result.path)
        print(f"Exported path to {args.export}")

    if not args.no_animation and result.path:
        animate_path(point_objects, result.path)


if __name__ == "__main__":
    main()
