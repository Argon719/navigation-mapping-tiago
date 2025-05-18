import re
import math
import tempfile

import numpy as np
import py_trees
from py_trees.common import Status
from controller import Motor
from ikpy.chain import Chain


class IKPYArmController(py_trees.behaviour.Behaviour):
    """
    Behavior Tree node to move a robot arm to a 3D target point using inverse kinematics.
    Controls torso height and gripper opening as well.
    """

    def __init__(
        self,
        name,
        blackboard,
        target_point,
        offset=[0.0, 0.08, 0.25],
        torso_height=0.32,
        gripper_opening=0.045
    ):
        super().__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read("robot")
        self.target_point = np.array(target_point)
        self.offset = np.array(offset)
        self.torso_height = torso_height
        self.gripper_opening = gripper_opening
        self.timestep = int(self.robot.getBasicTimeStep())

        # Enable robot sensors
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        # Define joints
        self.arm_joints = [
            "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
            "arm_5_joint", "arm_6_joint", "arm_7_joint"
        ]
        self.manual_joints = [
            "torso_lift_joint", "gripper_left_finger_joint", "gripper_right_finger_joint"
        ]
        self.head_joints = ["head_1_joint", "head_2_joint"]
        self.all_joints = self.head_joints + self.arm_joints + self.manual_joints

        # Initialize motors and sensors
        self.motors = {}
        self.sensors = {}

        for name in self.all_joints:
            motor = self.robot.getDevice(name)
            sensor_name = (
                name.replace("finger_joint", "sensor_finger_joint")
                if "gripper" in name else f"{name}_sensor"
            )
            sensor = self.robot.getDevice(sensor_name)
            if sensor:
                sensor.enable(self.timestep)
                self.sensors[name] = sensor

            velocity = 0.07 if "torso" in name else 0.05 if "gripper" in name else 1.0
            motor.setVelocity(velocity)
            motor.setPosition(float('inf'))
            self.motors[name] = motor
        # Prepare URDF for inverse kinematics (IK)
        urdf_str = self.robot.getUrdf()
        urdf_str = re.sub(r'type="continuous"', 'type="revolute"', urdf_str)
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as f:
            f.write(urdf_str)
            self.urdf_path = f.name

        # Define IK chain configuration
        base_elements = [
            "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
            "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm"
        ]
        last_link_vector = [0.004, 0.0, -0.1741]
        active_links_mask = [
            False, False, False, False, True, True, True, True,
            True, True, True, False, False, False, False
        ]

        self.chain = Chain.from_urdf_file(
            self.urdf_path,
            base_elements=base_elements,
            last_link_vector=last_link_vector,
            active_links_mask=active_links_mask
        )

        # IK completion flag and error thresholds
        self.ik_done = False
        self.arm_threshold = 0.01
        self.torso_threshold = 0.0003
        self.gripper_threshold = 0.0091

    def get_robot_rotation_yaw(self):
        """
        Calculate the robot's yaw angle (rotation around vertical axis)
        using compass readings.
        """
        compass_values = self.compass.getValues()
        return math.atan2(compass_values[0], compass_values[2])

    def world_to_robot(self, world_target):
        """
        Convert a world coordinate (x, y, z) to the robot's local coordinate frame.
        """
        robot_pos = self.gps.getValues()[:2]
        robot_yaw = self.get_robot_rotation_yaw()

        dx = world_target[0] - robot_pos[0]
        dy = world_target[1] - robot_pos[1]

        x_robot = math.cos(-robot_yaw) * dx - math.sin(-robot_yaw) * dy
        y_robot = math.sin(-robot_yaw) * dx + math.cos(-robot_yaw) * dy
        z_robot = world_target[2]

        return np.array([x_robot, y_robot, z_robot])
        
    def calculate_ik(self, local_target):
        """
        Perform inverse kinematics to reach a local 3D target position.
        Sends position commands to the arm joints.
        """
        adjusted = local_target + self.offset

        # Use current joint values as the initial guess for IK solver
        guess = [0, 0, 0, 0] + [
            self.sensors[name].getValue() for name in self.arm_joints
        ] + [0, 0, 0, 0]

        result = self.chain.inverse_kinematics(
            adjusted,
            initial_position=guess,
            target_orientation=[0, 0, 1],
            orientation_mode="Y"
        )

        joint_targets = {}
        for i, val in enumerate(result):
            name = self.chain.links[i].name
            if name in self.arm_joints:
                joint_targets[name] = val
                self.motors[name].setPosition(val)

        return joint_targets

    def update(self):
        """
        Main behavior tick method.
        Moves the arm to the specified world target using IK.
        Adjusts torso and gripper too.
        """
        if not self.ik_done:
            local_target = self.world_to_robot(self.target_point)
            print(f"[{self.name}] → Target world: {self.target_point}, Local: {local_target}")

            self.joint_targets = self.calculate_ik(local_target)

            print(f"[{self.name}] → Setting torso lift: {self.torso_height:.3f}")
            self.motors['torso_lift_joint'].setPosition(self.torso_height)

            print(f"[{self.name}] → Setting gripper to: {self.gripper_opening:.3f}")
            self.motors['gripper_left_finger_joint'].setPosition(self.gripper_opening)
            self.motors['gripper_right_finger_joint'].setPosition(self.gripper_opening)

            self.ik_done = True
            return Status.RUNNING

        else:
            # Check errors to determine if arm has reached target
            joint_err = sum(
                (self.joint_targets[name] - self.sensors[name].getValue()) ** 2
                for name in self.joint_targets if name in self.sensors
            )
            torso_err = abs(self.torso_height - self.sensors['torso_lift_joint'].getValue())
            grip_err_l = abs(self.gripper_opening - self.sensors['gripper_left_finger_joint'].getValue())
            grip_err_r = abs(self.gripper_opening - self.sensors['gripper_right_finger_joint'].getValue())

            arm_ok = joint_err < self.arm_threshold
            torso_ok = torso_err < self.torso_threshold
            grip_ok = (
                grip_err_l < self.gripper_threshold and
                grip_err_r < self.gripper_threshold
            )

            print(
                f"[{self.name}] → Errors: joints={joint_err:.4f}, "
                f"torso={torso_err:.4f}, grip=[{grip_err_l:.4f}, {grip_err_r:.4f}]"
            )

            if arm_ok and torso_ok and grip_ok:
                print(f"[{self.name}] → Arm + torso + gripper all reached target. ")
                return Status.SUCCESS
            else:
                return Status.RUNNING
