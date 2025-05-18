# === Imports ===

import numpy as np  # Numerical operations
import py_trees     # Behavior trees
import random       # Random sampling
import math         # Trigonometry and math functions


# === Coordinate Conversion Utilities ===

def world2map(xw, yw):
    """
    Convert world coordinates to map pixel coordinates.

    Parameters:
        xw (float): X-coordinate in the world frame.
        yw (float): Y-coordinate in the world frame.

    Returns:
        tuple: (px, py) pixel coordinates in map space.
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
        px (int): Pixel X-coordinate on the map.
        py (int): Pixel Y-coordinate on the map.

    Returns:
        tuple: (xw, yw) world coordinates.
    """
    xw = px / 40 - 2.25
    yw = py / -50 + 2
    return xw, yw


# === RRT Path Planning Algorithm ===

def get_rrt_path(occupancy_map, start, goal, max_iter=3000, step_size=7, goal_threshold=8, goal_bias=0.2):
    """
    Basic Rapidly-exploring Random Tree (RRT) algorithm for path planning.

    Parameters:
        occupancy_map (np.ndarray): Binary map (0 = free, 1 = occupied).
        start (tuple): Start pixel coordinates.
        goal (tuple): Goal pixel coordinates.
        max_iter (int): Max number of nodes to explore.
        step_size (int): Distance to extend toward each sample.
        goal_threshold (int): Distance to consider goal reached.
        goal_bias (float): Probability to sample the goal directly.

    Returns:
        list: Path from start to goal as a list of (px, py).
    """

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_collision(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        steps = int(max(abs(x2 - x1), abs(y2 - y1)))
        for i in range(steps + 1):
            x = int(x1 + (x2 - x1) * i / steps)
            y = int(y1 + (y2 - y1) * i / steps)
            if x < 0 or y < 0 or x >= occupancy_map.shape[0] or y >= occupancy_map.shape[1]:
                return True
            if occupancy_map[x, y] == 1:
                return True
        return False

    tree = {start: None}
    nodes = [start]

    for _ in range(max_iter):
        # Sample random point or goal
        if random.random() < goal_bias:
            sample = goal
        else:
            sample = (
                random.randint(0, occupancy_map.shape[0] - 1),
                random.randint(0, occupancy_map.shape[1] - 1)
            )

        # Find nearest node
        nearest = min(nodes, key=lambda n: distance(n, sample))
        theta = math.atan2(sample[1] - nearest[1], sample[0] - nearest[0])
        new_node = (
            int(nearest[0] + step_size * math.cos(theta)),
            int(nearest[1] + step_size * math.sin(theta))
        )

        # Check collision and add to tree
        if (
            0 <= new_node[0] < occupancy_map.shape[0] and
            0 <= new_node[1] < occupancy_map.shape[1] and
            not is_collision(nearest, new_node)
        ):
            tree[new_node] = nearest
            nodes.append(new_node)
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


# === Planning Behavior Tree Node ===

class Planning(py_trees.behaviour.Behaviour):
    """
    Behavior tree node to compute a path using RRT and update the blackboard with waypoints.
    """

    def __init__(self, name, blackboard, goal):
        super().__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard
        self.goal = goal

    def setup(self):
        self.node = 'rrt_basic_planning.py'
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
        path = get_rrt_path(self.map, self.start, self.goal_map, goal_bias=0.1)

        if not path:
            print(f"{self.node} ->[{self.name}]  Failed to compute path with RRT")
            return py_trees.common.Status.FAILURE

        world_path = [map2world(px, py) for px, py in path]
        simplified_path = world_path[int((len(world_path) * 25 / 100)):]  # Drop first 25%
        self.blackboard.write('waypoints', simplified_path)

        path_color = 0xFFFF00  # Yellow
        for waypoint in world_path:
            px, py = world2map(waypoint[0], waypoint[1])
            self.display.setColor(path_color)
            self.display.drawPixel(px, py)

        print(f"{self.node} ->[{self.name}] → Status: SUCCESS  Path planning complete")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        print(f"{self.node} ->[{self.name}] → Status: {new_status.name}  Path plan finished")
