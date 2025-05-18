# === Imports ===

import py_trees  # Behavior tree framework
import numpy as np  # Numerical array handling


class DrawMapFromFile(py_trees.behaviour.Behaviour):
    """
    Behavior node that loads a previously saved occupancy map (cspace.npy)
    and renders it onto the robot's display.
    """

    def __init__(self, name, blackboard):
        """
        Initialize the DrawMapFromFile behavior.

        :param name: Name of the behavior in the tree.
        :param blackboard: Shared storage (should contain the robot reference).
        """
        super().__init__(name)
        self.robot = blackboard.read('robot')
        self.display = None  # Will be initialized in setup()

    def setup(self):
        """
        Set up the Display device used to draw the occupancy map.
        Called once before the behavior runs.
        """
        self.display = self.robot.getDevice('display')
        self.node = 'draw_map.py'  # Optional metadata for debugging

    def update(self):
        """
        Attempt to load and render the occupancy map.

        :returns: SUCCESS if map was loaded and drawn, otherwise FAILURE.
        """
        try:
            # Load the occupancy grid from file
            cspace = np.load("cspace.npy")
            cspace = (cspace >= 0.9).astype(np.uint8)  # Convert to binary

            width, height = cspace.shape

            # Clear the display first (draw black background)
            self.display.setColor(0x000000)
            for x in range(width):
                for y in range(height):
                    self.display.drawPixel(x, y)

            # Draw gray pixels where map is occupied
            self.display.setColor(0x636363)
            for x in range(width):
                for y in range(height):
                    if cspace[x, y]:
                        self.display.drawPixel(x, y)

            status = py_trees.common.Status.SUCCESS
            print(f"[{self.name}] → Status: {status.name} 📂 Map loaded and displayed")
            return status

        except Exception as e:
            status = py_trees.common.Status.FAILURE
            print(f"[{self.name}] → Status: {status.name} ❌ Failed to load map: {e}")
            return status
