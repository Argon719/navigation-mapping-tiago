# === Imports ===

import py_trees  # Behavior tree framework
import numpy as np  # Numerical computations
from scipy import signal  # Convolution for map smoothing
from matplotlib import pyplot as plt  # For optional debugging visualization


# === Coordinate Conversion Functions ===

def world2map(xw, yw):
    """
    Convert world coordinates to pixel map coordinates.
    :param xw: X in world space.
    :param yw: Y in world space.
    :return: [px, py] in map space (integers clamped to map size).
    """
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))
    px = max(0, min(px, 199))
    py = max(0, min(py, 299))
    return [px, py]


def map2world(px, py):
    """
    Convert map pixel coordinates to world coordinates.
    :param px: X in map space
    :param py: Y in map space
    :return: [xw, yw] in world coordinates
    """
    xw = px / 40 - 2.25
    yw = py / (-50) + 2
    return [xw, yw]


# === Mapping Behavior Tree Node ===

class Mapping(py_trees.behaviour.Behaviour):
    """
    Behavior node that builds a 2D occupancy map from Lidar and GPS data.
    Stores the final result as a NumPy array file (cspace.npy).
    """

    def __init__(self, name, blackboard):
        """
        Initialize the behavior and prepare state.
        :param name: Name of the behavior node.
        :param blackboard: Shared blackboard with robot reference.
        """
        super().__init__(name)
        self.robot = blackboard.read('robot')
        self.hasrun = False  # Track whether update has executed
        self.map = None

    def setup(self):
        """
        Called once before the behavior starts running.
        Initializes devices and enables sensors.
        """
        self.node = 'mapping.py'
        self.timestep = int(self.robot.getBasicTimeStep())

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.display = self.robot.getDevice('display')

    def initialise(self):
        """
        Called when the behavior starts.
        Resets the occupancy map and computes Lidar angles.
        """
        print(f"{self.node} ->[{self.name}] Initializing")
        self.map = np.zeros((200, 300))
        self.angles = np.linspace(4.19 / 2, -4.19 / 2, 667)
        self.angles = self.angles[80:-80]  # Trim unreliable outer edges

    def update(self):
        """
        Main mapping logic: update occupancy map from Lidar points.
        :return: py_trees.common.Status.RUNNING
        """
        self.hasrun = True

        # Read robot position and heading
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        heading = np.arctan2(
            self.compass.getValues()[0],
            self.compass.getValues()[1]
        )

        # Create transformation matrix from robot to world
        w_T_r = np.array([
            [np.cos(heading), -np.sin(heading), xw],
            [np.sin(heading), np.cos(heading), yw],
            [0, 0, 1]
        ])

        # Get and trim Lidar range data
        ranges = np.array(self.lidar.getRangeImage())
        ranges = ranges[80:-80]
        ranges[ranges == np.inf] = 100  # Clip infinite values

        # Convert Lidar points to homogeneous coordinates in robot frame
        X_i = np.array([
            ranges * np.cos(self.angles) + 0.202,  # Offset Lidar origin
            ranges * np.sin(self.angles),
            np.ones(len(ranges))
        ])

        # Transform to world frame
        D = w_T_r @ X_i

        # Update map with measurements
        for d in D.T:
            px, py = world2map(d[0], d[1])
            self.map[px, py] += 0.001
            self.map[px, py] = min(self.map[px, py], 1.0)

            intensity = int(self.map[px, py] * 255)
            gray = intensity * 256**2 + intensity * 256 + intensity
            self.display.setColor(gray)
            self.display.drawPixel(px, py)

        print(f"{self.node} ->[{self.name}] → Status: RUNNING  Mapping in progress...")
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Called when the behavior finishes or is interrupted.
        Saves the final convolved map to disk.
        :param new_status: New behavior status.
        """
        print(f"{self.node} ->[{self.name}] → Status: {new_status.name}  Map saved")

        if self.hasrun:
            # Smooth the map using convolution
            cspace = signal.convolve2d(self.map, np.ones((28, 28)), mode='same')
            np.save('cspace', cspace)

            # Optional debug visualization
            # plt.figure()
            # plt.imshow(cspace > 0.9)
            # plt.show()
