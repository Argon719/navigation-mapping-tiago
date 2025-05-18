# === Imports ===

import numpy as np  # Numerical operations
import py_trees     # Behavior tree implementation
import random       # Random sampling
import math         # Math functions


# === Coordinate Conversion Utilities ===

def world2map(xw, yw):
    """
    Convert world coordinates to map pixel coordinates.

    Parameters:
        xw (float): World x-coordinate.
        yw (float): World y-coordinate.

    Returns:
        tuple: (px, py) pixel coordinates in map.
    """
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))
    px = max(0, min(px, 199))
    py = max(0, min(py, 299))
    return px, py


def map2world(px, py):
    """
    Convert map pixel coordinates back to world coordinates.

    Parameters:
        px (int): Pixel x-coordinate.
        py (int): Pixel y-coordinate.

    Returns:
        tuple: (xw, yw) world coordinates.
    """
    xw = px / 40 - 2.25
    yw = py / -50 + 2
    return xw, yw


# === RRT* Path Planning Algorithm ===

def get_rrt_path(occupancy_map, start, goal, max_iter=2000, step_size=5, goal_threshold=7, goal_bias=0.1, rewire_radius=10):
    """
    RRT* path planning algorithm with rewiring and biasing.

    Parameters:
        occupancy_map (np.ndarray): Binary map (0 = free, 1 = occupied).
        start (tuple): Start pixel coordinates.
        goal (tuple): Goal pixel coordinates.
        max_iter (int): Max number of iterations.
        step_size (int): Step length for tree growth.
        goal_threshold (float): Distance to consider goal reached.
        goal_bias (float): Probability to sample the goal.
        rewire_radius (float): Radius for rewiring.

    Returns:
        list: Path from start to goal as list of (px, py).
    """

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_collision(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if p1 == p2:
            return False
        steps = int(max(abs(x2 - x1), abs(y2 - y1)))
        if steps == 0:
            return False
        for i in range(steps + 1):
            x = int(x1 + (x2 - x1) * i / steps)
            y = int(y1 + (y2 - y1) * i / steps)
            if x < 0 or y < 0 or x >= occupancy_map.shape[0] or y >= occupancy_map.shape[1]:
                return True
            if occupancy_map[x, y] == 1:
                return True
        return False

    def get_nearby_nodes(node, nodes):
        return [n for n in nodes if distance(n, node) < rewire_radius]

    tree = {start: None}
    costs = {start: 0}
    nodes = [start]

    for _ in range(max_iter):
        sample = goal if random.random() < goal_bias else (
            random.randint(0, occupancy_map.shape[0] - 1),
            random.randint(0, occupancy_map.shape[1] - 1)
        )

        nearest = min(nodes, key=lambda n: distance(n, sample))
        theta = math.atan2(sample[1] - nearest[1], sample[0] - nearest[0])
        new_node = (
            int(nearest[0] + step_size * math.cos(theta)),
            int(nearest[1] + step_size * math.sin(theta))
        )

        if (
            0 <= new_node[0] < occupancy_map.shape[0] and
            0 <= new_node[1] < occupancy_map.shape[1] and
            not is_collision(nearest, new_node)
        ):
            nearby_nodes = get_nearby_nodes(new_node, nodes)
            best_parent = nearest
            min_cost = costs[nearest] + distance(nearest, new_node)

            for node in nearby_nodes:
                if not is_collision(node, new_node):
                    c = costs[node] + distance(node, new_node)
                    if c < min_cost:
                        best_parent = node
                        min_cost = c

            tree[new_node] = best_parent
            costs[new_node] = min_cost
            nodes.append(new_node)

            # Rewire nearby nodes
            for node in nearby_nodes:
                if not is_collision(new_node, node):
                    new_cost = costs[new_node] + distance(new_node, node)
                    if new_cost < costs.get(node, float('inf')):
                        tree[node] = new_node
                        costs[node] = new_cost

            if distance(new_node, goal) < goal_threshold:
                tree[goal] = new_node
                break

    # Reconstruct path
    path = []
    current = goal
    while current in tree:
        path.insert(0, current)
        current = tree[current]

    if not path or path[0] != start:
        return []

    return path


# === Planning Behavior Node ===

class Planning(py_trees.behaviour.Behaviour):
    """
    Behavior tree node to compute a path using RRT* and update blackboard with waypoints.
    """

    def __init__(self, name, blackboard, goal):
        super().__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard
        self.goal = goal

    def setup(self):
        self.node = 'rrt_star_planning.py'
        self.timestep = int(self.robot.getBasicTimeStep())
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.display = self.robot.getDevice('display')

    def initialise(self):
        current_position = self.gps.getValues()
        self.start = world2map(current_position[0], current_position[1])
        self.goal_map = world2map(self.goal[0], self.goal[1])

        occupancy = np.load('cspace.npy')
        self.map = (occupancy >= 0.9).astype(np.uint8)

        print(f"{self.node} ->[{self.name}] → Status: INITIALISING  Start path computing")

    def update(self):
        path = get_rrt_path(self.map, self.start, self.goal_map)

        if not path:
            print(f"{self.node} ->[{self.name}]  Failed to compute path with RRT*")
            return py_trees.common.Status.FAILURE

        world_path = [map2world(px, py) for px, py in path]

        # Drop first 15% of path if path is long
        skip = int(len(world_path) * 0.15)
        if skip >= len(world_path) - 1:
            skip = 0
        world_path = world_path[skip:]

        self.blackboard.write('waypoints', world_path)

        path_color = 0xFFFF00  # Yellow
        for waypoint in world_path:
            px, py = world2map(waypoint[0], waypoint[1])
            self.display.setColor(path_color)
            self.display.drawPixel(px, py)

        print(f"{self.node} ->[{self.name}] → Status: SUCCESS  Path planning complete")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        print(f"{self.node} ->[{self.name}] → Status: {new_status.name}  Path plan finished")
