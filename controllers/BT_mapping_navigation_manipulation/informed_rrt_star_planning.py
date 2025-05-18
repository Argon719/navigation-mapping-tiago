# === Imports ===

import numpy as np
import py_trees
import random
import math

# === Utility: Coordinate conversion ===

def world2map(xw, yw):
    """
    Convert world coordinates to map pixel coordinates.
    """
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))
    return max(0, min(px, 199)), max(0, min(py, 299))


def map2world(px, py):
    """
    Convert map pixel coordinates to world coordinates.
    """
    xw = px / 40 - 2.25
    yw = py / -50 + 2
    return xw, yw


# === Geometry & Sampling ===

def dist(a, b):
    """
    Euclidean distance between two 2D points.
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def is_collision(cspace, p1, p2):
    """
    Check for collision between two points on a binary map.
    """
    x1, y1 = p1
    x2, y2 = p2
    steps = max(1, int(max(abs(x2 - x1), abs(y2 - y1))))
    for i in range(steps + 1):
        xi = int(x1 + (x2 - x1) * i / steps)
        yi = int(y1 + (y2 - y1) * i / steps)
        if xi < 0 or yi < 0 or xi >= cspace.shape[0] or yi >= cspace.shape[1]:
            return True
        if cspace[xi, yi] == 1:
            return True
    return False


def sample_in_ellipse(start, goal, c_best, cspace):
    """
    Sample a point inside the informed ellipse defined by the start-goal path length.
    """
    c_min = dist(start, goal)
    if c_best == float('inf') or c_best < c_min:
        return random.randint(0, cspace.shape[0] - 1), random.randint(0, cspace.shape[1] - 1)

    r = math.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    a = c_best / 2.0
    b = math.sqrt(max(c_best**2 - c_min**2, 0.0)) / 2.0
    x_e = a * r * math.cos(theta)
    y_e = b * r * math.sin(theta)

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    phi = math.atan2(dy, dx)

    xr = x_e * math.cos(phi) - y_e * math.sin(phi)
    yr = x_e * math.sin(phi) + y_e * math.cos(phi)

    cx = (start[0] + goal[0]) / 2.0
    cy = (start[1] + goal[1]) / 2.0

    sx = int(min(max(cx + xr, 0), cspace.shape[0] - 1))
    sy = int(min(max(cy + yr, 0), cspace.shape[1] - 1))
    return sx, sy


# === Path smoothing using Catmull-Rom splines ===

def catmull_rom_spline_world(points, subdivisions=8):
    """
    Smooth the path using Catmull-Rom spline interpolation.
    """
    if len(points) < 3 or subdivisions < 1:
        return points

    pts = [points[0]] + points + [points[-1]]
    smoothed = []

    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(subdivisions):
            t = j / float(subdivisions)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2 * p1[0]) +
                       (p2[0] - p0[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (p3[0] - p0[0] + 3 * (p1[0] - p2[0])) * t3)
            y = 0.5 * ((2 * p1[1]) +
                       (p2[1] - p0[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (p3[1] - p0[1] + 3 * (p1[1] - p2[1])) * t3)
            smoothed.append((x, y))

    smoothed.append(points[-1])
    return smoothed


# === Informed RRT* Planning Behavior ===

class Planning(py_trees.behaviour.Behaviour):
    """
    Informed RRT* planner with spline-based smoothing. Outputs waypoints to the blackboard.
    """

    CHUNK = 100  # iterations per tick

    def __init__(self, name, blackboard, goal,
                 max_iter=5000, step_size=10,
                 goal_threshold=10, goal_bias=0.3,
                 rewire_radius=10, spline_subdivisions=8):
        super().__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard
        self.goal = goal
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.spline_subdivisions = spline_subdivisions

        # RRT* state variables
        self.tree = None
        self.cost = None
        self.nodes = None
        self.c_best = None
        self.best_parent = None
        self.iter_count = None
        self.raw_path = None

    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.display = self.robot.getDevice('display')

    def initialise(self):
        pos = self.gps.getValues()
        self.start = world2map(pos[0], pos[1])
        self.goal_map = world2map(self.goal[0], self.goal[1])

        raw = np.load('cspace.npy')
        self.map = (raw >= 0.9).astype(np.uint8)

        self.tree = {self.start: None}
        self.cost = {self.start: 0.0}
        self.nodes = [self.start]
        self.c_best = float('inf')
        self.best_parent = None
        self.iter_count = 0
        self.raw_path = None

        print(f"{self.name} init: start={self.start}, goal={self.goal_map}, map={self.map.shape}")

    def update(self):
        # Expand RRT* tree
        if self.raw_path is None:
            for _ in range(self.CHUNK):
                if self.iter_count >= self.max_iter:
                    break
                self.iter_count += 1

                sample = self.goal_map if random.random() < self.goal_bias else \
                    sample_in_ellipse(self.start, self.goal_map, self.c_best, self.map)

                nearest = min(self.nodes, key=lambda n: dist(n, sample))
                theta = math.atan2(sample[1] - nearest[1], sample[0] - nearest[0])
                new = (
                    int(nearest[0] + self.step_size * math.cos(theta)),
                    int(nearest[1] + self.step_size * math.sin(theta))
                )

                if not (0 <= new[0] < self.map.shape[0] and 0 <= new[1] < self.map.shape[1]):
                    continue
                if is_collision(self.map, nearest, new):
                    continue

                neighbors = [n for n in self.nodes if dist(n, new) < self.rewire_radius]
                parent = nearest
                min_cost = self.cost[nearest] + dist(nearest, new)

                for n in neighbors:
                    if not is_collision(self.map, n, new):
                        cost_n = self.cost[n] + dist(n, new)
                        if cost_n < min_cost:
                            parent = n
                            min_cost = cost_n

                self.tree[new] = parent
                self.cost[new] = min_cost
                self.nodes.append(new)

                # Rewiring
                for n in neighbors:
                    if not is_collision(self.map, new, n):
                        new_cost = self.cost[new] + dist(new, n)
                        if new_cost < self.cost.get(n, float('inf')):
                            self.tree[n] = new
                            self.cost[n] = new_cost

                # Update best path if new is near goal
                if dist(new, self.goal_map) < self.goal_threshold:
                    total_cost = self.cost[new] + dist(new, self.goal_map)
                    if total_cost < self.c_best:
                        self.c_best = total_cost
                        self.best_parent = new

                if self.best_parent is not None:
                    break

            if self.best_parent is None and self.iter_count < self.max_iter:
                print(f"{self.name} planning... {self.iter_count}/{self.max_iter}")
                return py_trees.common.Status.RUNNING

            if self.best_parent is None:
                print(f"{self.name}  No valid path found")
                self.raw_path = []
            else:
                self.tree[self.goal_map] = self.best_parent
                node = self.goal_map
                path = []
                while node is not None:
                    path.append(node)
                    node = self.tree[node]
                self.raw_path = list(reversed(path))

        if not self.raw_path:
            return py_trees.common.Status.FAILURE

        # Smoothing and waypoint output
        world_points = [map2world(x, y) for x, y in self.raw_path]
        smoothed = catmull_rom_spline_world(world_points, self.spline_subdivisions)
        smoothed = smoothed[int(len(smoothed) * 0.05):]  # drop first 5% if needed

        self.blackboard.write('waypoints', smoothed)

        for pt in smoothed:
            px, py = world2map(pt[0], pt[1])
            self.display.setColor(0xFFFF00)
            self.display.drawPixel(px, py)

        print(f"{self.name}  Path complete in {self.iter_count} iterations with {len(smoothed)} points")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        print(f"{self.name} → Terminated with status: {new_status.name}")
