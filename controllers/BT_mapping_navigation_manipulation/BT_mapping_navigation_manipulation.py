# === Imports ===

import py_trees  # Behavior tree framework
from py_trees.composites import Sequence, Selector, Parallel  # Composite behavior nodes
from py_trees.common import Status, ParallelPolicy  # Behavior tree status and policies

import numpy as np  # Numerical operations
from os.path import exists  # Check if file exists
from scipy import signal  # Signal processing
import os  # Operating system utilities

# Custom robot modules
from motor_control import MoveToAngleOrDistance  # Drive and turn control using compass/GPS
from manipulation import SetRobotArms  # Joint control for safe or task poses
from draw_map import DrawMapFromFile  # Load and render a saved Lidar map
from mapping import Mapping  # Lidar-based mapping behavior
from lidar_navigation import Navigation  # Obstacle-aware navigation behavior
#from a_star_planning import Planning
#from rrt_basic_planning import Planning
#from rrt_star_planning import Planning
from informed_rrt_star_planning import Planning  # Informed RRT* path planning
from ikpy_camera import IKPYCameraGraspController  # Detect and grasp object with camera + IK
from ikpy_manipulation import IKPYArmController  # Move arm with inverse kinematics

# Webots robot control
from controller import Supervisor, Motor  # Access robot, motors, and simulation functions


# === Blackboard definition ===

class Blackboard:
    """
    A simple shared storage for passing data between behavior tree nodes.
    """
    def __init__(self):
        self.data = {}

    def write(self, key, value):
        self.data[key] = value  # Store value under key

    def read(self, key):
        return self.data.get(key)  # Retrieve value by key


# === Initialization ===

# Create the shared blackboard
blackboard = Blackboard()

# Initialize Webots Supervisor (main robot API)
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())  # Get simulation step size
blackboard.write('robot', robot)  # Save robot object to blackboard

# Access the Display device to draw map pixels later
display = robot.getDevice('display')


# === Waypoints for navigation ===

WAYPOINTS = [
    (-0.5,  0.2),   # Align A
    (-1.67,  0.4),  # Align B
    (0.55,  0.60),  # Waypoint 1
    (0.55, -0.9),   # Waypoint 2
    (0.60, -1.75),  # Waypoint 3
    (0.60, -3.2),   # Waypoint 4
    (-0.58, -3.25), # Waypoint 5
    (-1.67, -3.2),  # Waypoint 6
    (-1.67, -2.39), # Waypoint 7
    (-1.67, -1.29), # Waypoint 8
    (-1.67,  0.2),  # Waypoint 9
    (0.20,  0.37)   # Waypoint 10
]

# Combine forward and reversed paths for round trip
blackboard.write('waypoints', np.concatenate((WAYPOINTS, np.flip(WAYPOINTS, 0)), axis=0))


# === Arm and head joint positions ===

safe_positions = {
    'torso_lift_joint': 0.320,
    'arm_1_joint': 0.71,
    'arm_2_joint': 1.02,
    'arm_3_joint': -2.815,
    'arm_4_joint': 1.011,
    'arm_5_joint': 0,
    'arm_6_joint': 0,
    'arm_7_joint': -1.57,
    'gripper_left_finger_joint': 0.045,
    'gripper_right_finger_joint': 0.045,
    'head_1_joint': 0,
    'head_2_joint': 0
}

protect_positions = {
    'torso_lift_joint': 0.350,
    'arm_1_joint': 1.610,
    'arm_2_joint': 1.02,
    'arm_3_joint': 0.11,
    'arm_4_joint': 2.225,
    'arm_5_joint': -1.724,
    'arm_6_joint': -1.390,
    'arm_7_joint': 0.0,
    'gripper_left_finger_joint': 0.04,
    'gripper_right_finger_joint': 0.04,
    'head_1_joint': 0,
    'head_2_joint': 0
}

class DoesMapExist(py_trees.behaviour.Behaviour):
    """
    Behavior node that checks whether a previously saved map file (cspace.npy) exists.
    If it does, the map is loaded and rendered on the Display.
    """
    def __init__(self, name):
        super().__init__(name)

    def update(self):
        file_exists = exists('cspace.npy')  # Check if map file exists
        if file_exists:
            self.map = np.load('cspace.npy')  # Load occupancy map
            map = (self.map >= 0.9).astype(np.uint8)  # Threshold to binary

            # Draw black pixels for all occupied areas
            for px in range(map.shape[0]):
                for py in range(map.shape[1]):
                    if map[px, py]:
                        display.setColor(0x000000)  # Black
                        display.drawPixel(px, py)

            # Overlay with gray to indicate existing map
            for px in range(map.shape[0]):
                for py in range(map.shape[1]):
                    if map[px, py]:
                        display.setColor(0x636363)  # Gray
                        display.drawPixel(px, py)

            status = Status.SUCCESS
            print(f"[{self.name}] → Status: {status.name} 📂 Map loaded and displayed")
            return status
        else:
            status = Status.FAILURE
            print(f"[{self.name}] → Status: {status.name} 🔎 Map not found")
            return status


class PauseSimulation(py_trees.behaviour.Behaviour):
    """
    Behavior node that pauses the Webots simulation and deletes the grasped_ids.json file if it exists.
    """
    def __init__(self, name, blackboard, pause_condition_key='pause'):
        super().__init__(name)
        self.blackboard = blackboard
        self.pause_condition_key = pause_condition_key
        self.grasped_ids_file = os.path.join(os.path.dirname(__file__), "grasped_ids.json")

    def update(self):
        self.blackboard.write(self.pause_condition_key, True)  # Request simulation pause

        if os.path.exists(self.grasped_ids_file):
            os.remove(self.grasped_ids_file)  # Clean up grasped object record
            print(f"[{self.name}] → Deleted grasped_ids.json")
        else:
            print(f"[{self.name}] → No grasped_ids.json to delete")

        status = Status.SUCCESS
        print(f"[{self.name}] → Status: {status.name} Pausing simulation")
        return status


class PlaceChair(py_trees.behaviour.Behaviour):
    """
    Behavior node that places a Webots object (e.g. a chair) at a specified position and rotation.
    """
    def __init__(self, name, blackboard, chair_def, position, rotation=(0, 1, 0, 0)):
        super().__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')  # Get Supervisor instance
        self.chair_def = chair_def  # DEF name of the chair
        self.position = list(position)  # Desired position [x, y, z]
        self.rotation = list(rotation)  # Desired rotation [axis + angle]

    def update(self):
        chair_node = self.robot.getFromDef(self.chair_def)  # Access the chair object in scene
        if chair_node:
            chair_node.getField("translation").setSFVec3f(self.position)  # Move the object
            chair_node.getField("rotation").setSFRotation(self.rotation)  # Rotate it
            print(f"[{self.name}] → Chair '{self.chair_def}' placed at {self.position}")
            return Status.SUCCESS
        else:
            print(f"[{self.name}] → Chair '{self.chair_def}' not found")
            return Status.FAILURE

# === Define the full behavior tree structure ===

tree = Sequence("Main", memory=True, children=[
    # Step 1: Move arms to safe position at start
    SetRobotArms("Initialize Robot Arms", blackboard, safe_positions),

    # Step 2: Check if map exists, otherwise map + navigate in parallel
    Selector("Does map exist?", memory=True, children=[
        DoesMapExist("Check for map file"),
        Parallel("Perform Mapping", policy=ParallelPolicy.SuccessOnOne(), children=[
            Mapping("Lidar Mapping", blackboard),
            Navigation("Explore Environment", blackboard)
        ])
    ]),

    # Step 3: Place two chairs into the scene
    PlaceChair("Place Chair 1", blackboard, "CHAIR_1", [0.75, -3.48, 0.0]),
    PlaceChair("Place Chair 5", blackboard, "CHAIR_5", [-1.75, -3.41, 0.0]),

    # Step 4: Find and pick up Jar 1 using camera + IK
    IKPYCameraGraspController("Detect and Grasp Jar 1", blackboard),

    # Step 5: Move robot to placement location for Jar 1
    MoveToAngleOrDistance("Drive Backward 0.4m", blackboard, mode="drive", target=-0.35),
    SetRobotArms("Protect Arm Position", blackboard, protect_positions),
    MoveToAngleOrDistance("Turn 50° Right", blackboard, mode="turn", target=-50),

    DrawMapFromFile("Load Map", blackboard),
    Planning("Plan Path to Jar 1 Placement", blackboard, (-1.0, -2.9)),
    Navigation("Navigate to Jar 1 Placement", blackboard),

    # Step 6: Place Jar 1 in three IK stages
    IKPYArmController("Lower Jar 1 Above", blackboard, target_point=[-0.5, -2.9, 0.9],
                      offset=[-0.25, -0.75, 0.0], torso_height=0.250, gripper_opening=0.04),
    IKPYArmController("Place Jar 1", blackboard, target_point=[-0.5, -2.9, 0.475],
                      offset=[-0.25, -0.75, 0.0], torso_height=0.250, gripper_opening=0.04),
    IKPYArmController("Release Jar 1", blackboard, target_point=[-0.5, -2.9, 0.525],
                      offset=[-0.25, -0.75, 0.0], torso_height=0.300, gripper_opening=0.045),

    SetRobotArms("Reset Arm to Safe", blackboard, safe_positions),

    # Step 7: Return to initial position
    DrawMapFromFile("Reload Map", blackboard),
    Planning("Path Back to Start 1", blackboard, (-1.5, 0.2)),
    Navigation("Return to Start 1", blackboard),

    DrawMapFromFile("Reload Map", blackboard),
    Planning("Path Back to Start 2", blackboard, (-0.5, 0.2)),
    Navigation("Return to Start 2", blackboard),

    # Step 8: Detect and place Jar 2
    IKPYCameraGraspController("Detect and Grasp Jar 2", blackboard),
    MoveToAngleOrDistance("Drive Backward 0.4m", blackboard, mode="drive", target=-0.4),
    SetRobotArms("Protect Arm", blackboard, protect_positions),
    MoveToAngleOrDistance("Turn 70° Right", blackboard, mode="turn", target=-70),

    DrawMapFromFile("Reload Map", blackboard),
    Planning("Plan Path to Jar 2 Placement", blackboard, (0.4, -2.1)),
    Navigation("Navigate to Jar 2 Placement", blackboard),

    IKPYArmController("Lower Jar 2 Above", blackboard, target_point=[-0.35, -1.6, 0.9],
                      offset=[0.0, 0.0, 0.0], torso_height=0.250, gripper_opening=0.04),
    IKPYArmController("Place Jar 2", blackboard, target_point=[-0.35, -1.6, 0.55],
                      offset=[0.0, 0.0, 0.0], torso_height=0.250, gripper_opening=0.04),
    IKPYArmController("Release Jar 2", blackboard, target_point=[-0.35, -1.6, 0.60],
                      offset=[0.0, 0.0, 0.0], torso_height=0.300, gripper_opening=0.045),

    SetRobotArms("Reset Arm to Safe", blackboard, safe_positions),
    MoveToAngleOrDistance("Turn 170° Left", blackboard, mode="turn", target=170),

    # Step 9: Prepare for Jar 3
    DrawMapFromFile("Reload Map", blackboard),
    Planning("Back to Start for Jar 3 - A", blackboard, (-1.5, 0.5)),
    Navigation("Return for Jar 3 - A", blackboard),

    MoveToAngleOrDistance("Turn 180°", blackboard, mode="turn", target=180),
    DrawMapFromFile("Reload Map", blackboard),
    Planning("Back to Start for Jar 3 - B", blackboard, (-0.5, 0.5)),
    Navigation("Return for Jar 3 - B", blackboard),

    IKPYCameraGraspController("Detect and Grasp Jar 3", blackboard),
    MoveToAngleOrDistance("Drive Backward 0.4m", blackboard, mode="drive", target=-0.4),
    SetRobotArms("Protect Arm", blackboard, protect_positions),
    MoveToAngleOrDistance("Turn 70° Right", blackboard, mode="turn", target=-70),

    # Step 10: Navigate to Jar 3 placement
    DrawMapFromFile("Reload Map", blackboard),
    Planning("Plan Path to Jar 3 Placement", blackboard, (0.4, -2.5)),
    Navigation("Navigate to Jar 3 Placement", blackboard),

    # Step 11: Place Jar 3 in three stages
    IKPYArmController("Lower Jar 3 Above", blackboard, target_point=[-0.35, -2.0, 0.9],
                      offset=[0.0, 0.0, 0.0], torso_height=0.250, gripper_opening=0.040),
    IKPYArmController("Place Jar 3", blackboard, target_point=[-0.35, -2.0, 0.55],
                      offset=[0.0, 0.0, 0.0], torso_height=0.250, gripper_opening=0.040),
    IKPYArmController("Release Jar 3", blackboard, target_point=[-0.35, -2.0, 0.60],
                      offset=[0.0, 0.0, 0.0], torso_height=0.300, gripper_opening=0.045),

    # Step 12: Return to safe state
    SetRobotArms("Reset Arm to Safe", blackboard, safe_positions),
    MoveToAngleOrDistance("Turn 150° Left", blackboard, mode="turn", target=150),

    # Step 13: Final return to base
    DrawMapFromFile("Reload Map", blackboard),
    Planning("Return Path to Start Point 1", blackboard, (-1.5, 0.2)),
    Navigation("Navigate to Start Point 1", blackboard),

    DrawMapFromFile("Reload Map", blackboard),
    Planning("Return Path to Start Point 2", blackboard, (-0.0, 0.2)),
    Navigation("Navigate to Start Point 2", blackboard),

    # Step 14: Pause simulation and clean up
    PauseSimulation("Pause Simulation", blackboard)
])


# === Setup and Execution ===

# Prepare the full behavior tree
tree.setup_with_descendants()  # Recursively initialize all behaviors

# Main simulation loop
while robot.step(timestep) != -1:
    state = tree.tick_once()  # Advance the behavior tree

    # Check if a pause was requested by any node
    if blackboard.read('pause'):
        print("[Main Loop] Pausing simulation as requested by behavior tree.")
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)  # Pause Webots
        blackboard.write('pause', False)  # Reset pause flag

