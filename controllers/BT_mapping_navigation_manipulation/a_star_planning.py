# Import necessary libraries
import numpy as np
from collections import defaultdict
from heapq import heapify, heappush, heappop
import py_trees


# === Utility functions ===

def world2map(xw, yw):
    """
    Convert world coordinates to map pixel coordinates.

    Parameters:
        xw (float): X-coordinate in the world frame.
        yw (float): Y-coordinate in the world frame.

    Returns:
        tuple: (px, py) map coordinates.
    """
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))
    px = max(0, min(px, 199))
    py = max(0, min(py, 299))
    return px, py


def map2world(px, py):
    """
    Convert map pixel coordinates to world coordinates.

    Parameters:
        px (int): X-coordinate on the map.
        py (int): Y-coordinate on the map.

    Returns:
        tuple: (xw, yw) world coordinates.
    """
    xw = px / 40 - 2.25
    yw = py / -50 + 2
    return xw, yw


# === A* Algorithm ===

def getShortestPath(occupancy_map, start, goal):
    """
    A* algorithm to find the shortest path on a 2D grid.

    Parameters:
        occupancy_map (np.ndarray): Binary occupancy grid (0 = free, 1 = obstacle).
        start (tuple): Starting pixel coordinates (px, py).
        goal (tuple): Goal pixel coordinates (px, py).

    Returns:
        list: Path from start to goal as a list of (px, py).
    """

    def get_neighbors(v):
        neighbors = []
        deltas = [(-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1), (0, -1), (1, 0), (0, 1)]
        for dx, dy in deltas:
            u = (v[0] + dx, v[1] + dy)
            if 0 <= u[0] < occupancy_map.shape[0] and 0 <= u[1] < occupancy_map.shape[1]:
                if occupancy_map[u[0], u[1]] == 0:
                    cost = np.sqrt(dx**2 + dy**2)
                    neighbors.append((u, cost))
        return neighbors

    queue = [(0, start)]
    heapify(queue)
    distances = defaultdict(lambda: float("inf"))
    distances[start] = 0
    visited = set()
    parent = {}

    while queue:
        current_dist, u = heappop(queue)
        visited.add(u)

        if u == goal:
            break

        for v, cost_uv in get_neighbors(u):
            if v in visited:
                continue
            new_dist = distances[u] + cost_uv
            if new_dist < distances[v]:
                distances[v] = new_dist
                parent[v] = u
                heuristic = np.sqrt((goal[0] - v[0])**2 + (goal[1] - v[1])**2)
                priority = distances[v] + heuristic
                heappush(queue, (priority, v))

    # Reconstruct path
    path = []
    c = goal
    while c in parent:
        path.insert(0, c)
        c = parent[c]
    path.insert(0, start)
    return path


# === Planning Behavior ===

class Planning(py_trees.behaviour.Behaviour):
    """
    Behavior node to plan a path from current robot position to a target goal
    using A* and publish waypoints to the blackboard.
    """

    def __init__(self, name, blackboard, goal):
        super().__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard
        self.goal = goal
        self.first_path_drawn = False

    def setup(self):
        self.node = 'planning.py'
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

        print(f"{self.node} ->[{self.name}] → Status: INITIALISING 🧭 Start path computing")

    def update(self):
        path = getShortestPath(self.map, self.start, self.goal_map)
        world_path = [map2world(px, py) for px, py in path]
        simplified_path = world_path[int((len(world_path) * 27 / 100)):]  # Drop first 27%

        # Optional: replace line below if you want full path instead
        self.blackboard.write('waypoints', simplified_path)

        path_color = 0xFFFF00  # Yellow
        for waypoint in world_path:
            path_x, path_y = world2map(waypoint[0], waypoint[1])
            self.display.setColor(path_color)
            self.display.drawPixel(path_x, path_y)

        print(f"{self.node} ->[{self.name}] → Status: SUCCESS 🧭 Path planning complete")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        print(f"{self.node} ->[{self.name}] → Status: {new_status.name} 💾 Path plan finished")
