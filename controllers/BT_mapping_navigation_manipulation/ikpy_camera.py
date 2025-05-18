import os
import re
import json
import math
import tempfile

import numpy as np
import py_trees
from py_trees.common import Status
from controller import Motor
from ikpy.chain import Chain


class IKPYCameraGraspController(py_trees.behaviour.Behaviour):
    """
    A behavior tree node that enables a robot to detect and grasp objects
    using its camera and inverse kinematics.
    """

    def __init__(self, name, blackboard):
        super().__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')
        self.timestep = int(self.robot.getBasicTimeStep())

        # Enable robot sensors
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # Setup base wheels
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Define joint groups
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

            # Set motor velocity based on type
            velocity = 0.07 if "torso" in name else 0.05 if "gripper" in name else 1.0
            motor.setVelocity(velocity)
            motor.setPosition(float('inf'))
            self.motors[name] = motor

        # JSON file for persistent grasped object IDs
        self.grasped_ids_file = os.path.join(os.path.dirname(__file__), "grasped_ids.json")
        self.grasped_ids = self.load_grasped_ids()

        # Other internal states
        self.ik_done = False
        self.position_on_image = None

        # Define safe home position for arm
        self.safe_position = {
            'torso_lift_joint': 0.320, 'arm_1_joint': 0.71, 'arm_2_joint': 1.02,
            'arm_3_joint': -2.815, 'arm_4_joint': 1.011, 'arm_5_joint': 0,
            'arm_6_joint': 0, 'arm_7_joint': -1.57,
            'gripper_left_finger_joint': 0.045, 'gripper_right_finger_joint': 0.045,
            'head_1_joint': 0, 'head_2_joint': 0
        }

        # Process URDF to fix joint types for IK
        urdf_str = self.robot.getUrdf()
        urdf_str = re.sub(r'type="continuous"', 'type="revolute"', urdf_str)
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as f:
            f.write(urdf_str)
            urdf_path = f.name

        # Define IK chain structure
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
            urdf_path,
            base_elements=base_elements,
            last_link_vector=last_link_vector,
            active_links_mask=active_links_mask
        )

        # Offset to adjust IK target position
        self.IK_offset = np.array([
            0.0, 0.08, -self.safe_position["torso_lift_joint"] + 0.25
        ])
    def load_grasped_ids(self):
        """Load previously grasped object IDs from a JSON file."""
        if os.path.exists(self.grasped_ids_file):
            try:
                with open(self.grasped_ids_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[IKPYCameraGraspController] Failed to load grasped_ids.json: {e}")
        return []

    def save_grasped_ids(self):
        """Save the list of grasped object IDs to a JSON file."""
        try:
            with open(self.grasped_ids_file, 'w') as f:
                json.dump(self.grasped_ids, f)
        except Exception as e:
            print(f"[IKPYCameraGraspController] Failed to save grasped_ids.json: {e}")

    def limit(self, value, min_val, max_val):
        """Clamp a value between min_val and max_val."""
        return max(min(value, max_val), min_val)

    def move_to_position(self, target, threshold=0.005):
        """
        Move the robot's joints to the specified target positions.
        Stops when error is below threshold.
        """
        print("Moving to position...")
        for name, pos in target.items():
            print(f" → {name}: {pos:.3f} rad")
            self.motors[name].setPosition(pos)
        while self.robot.step(self.timestep) != -1:
            err = sum(
                (pos - self.sensors[name].getValue()) ** 2
                for name, pos in target.items() if name in self.sensors
            )
            if err < threshold:
                break

    def lift_arm(self, position):
        """Raise the torso to a given position."""
        print("Lifting arm...")
        self.motors['torso_lift_joint'].setPosition(position)
        while self.robot.step(self.timestep) != -1:
            if abs(self.sensors['torso_lift_joint'].getValue() - position) < 0.005:
                break

    def close_gripper(self):
        """Close both gripper fingers to grasp an object."""
        print("Closing gripper...")
        self.motors['gripper_left_finger_joint'].setPosition(0.04)
        self.motors['gripper_right_finger_joint'].setPosition(0.04)

    def get_T_0_3(self):
        """
        Compute transformation matrix from base frame to head/camera frame.
        Uses sensor values to calculate orientation.
        """
        z = 0.6 + self.sensors['torso_lift_joint'].getValue()
        t0 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

        theta1 = self.sensors['head_1_joint'].getValue()
        c1, s1 = np.cos(theta1), np.sin(theta1)
        t1 = np.array([
            [c1, -s1, 0, 0.182],
            [s1, c1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        theta2 = self.sensors['head_2_joint'].getValue()
        c2, s2 = np.cos(theta2), np.sin(theta2)
        t2 = np.array([
            [c2, 0, s2, 0.005],
            [0, 1, 0, 0],
            [-s2, 0, c2, 0.098],
            [0, 0, 0, 1]
        ])

        return t0 @ t1 @ t2
    def control_base(self, target_pos):
        """
        Control the robot's base to align with the target object.
        Returns True when aligned.
        """
        MAX_SPEED = 6.28
        Kp_orientation = 4.0
        Kp_linear = 1.5

        x, y = target_pos[0], target_pos[1]
        dist = math.sqrt(x ** 2 + y ** 2)
        angle = math.atan2(y, x)

        target_dist = 1.2  # desired distance to object
        error = dist - target_dist

        lin = Kp_linear * error
        ang = Kp_orientation * angle

        lin = self.limit(lin, -MAX_SPEED, MAX_SPEED)
        ang = self.limit(ang, -MAX_SPEED, MAX_SPEED)

        left = lin - ang
        right = lin + ang

        print(f"Control: Distance={dist:.3f} m, Angle={angle:.3f} rad")
        print(f"         Linear={lin:.3f}, Angular={ang:.3f} → L={left:.3f}, R={right:.3f}")

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

        if abs(angle) < 0.01 and abs(error) < 0.05:
            print("Robot aligned with object.")
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
            return True
        return False

    def calculate_ik(self, position, offset=np.zeros(3), apply=True):
        """
        Perform inverse kinematics for a given 3D position (with optional offset).
        If apply is True, send commands to motors.
        Returns a dictionary of joint targets.
        """
        print("Calculating IK...")
        adjusted = position + offset
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
                if apply:
                    print(f" → {name}: {val:.3f} rad")
                    self.motors[name].setPosition(val)

        return joint_targets

    def wait_for_arm(self, targets, threshold=0.005):
        """
        Wait for the robot arm to reach target positions.
        Continues stepping the simulation until below error threshold.
        """
        print("Waiting for arm to reach target...")
        while self.robot.step(self.timestep) != -1:
            error = sum(
                (targets[name] - self.sensors[name].getValue()) ** 2
                for name in targets if name in self.sensors
            )
            if error < threshold:
                break
    def update(self):
        """
        Main behavior method called on each tick.
        Uses camera recognition to identify and grasp new objects.
        """
        # Always reload grasped IDs to remain synchronized
        self.grasped_ids = self.load_grasped_ids()

        # Get transformation matrix from base to camera
        T_0_3 = self.get_T_0_3()
        objects = self.camera.getRecognitionObjects()

        target = None
        target_id = None
        min_id = float('inf')

        # Identify closest ungrasped object
        for obj in objects:
            pos = obj.getPosition()
            pid = obj.getId()
            image_pos = obj.getPositionOnImage()

            p_cam = np.array([pos[0], pos[1], pos[2], 1.0])
            p_base = T_0_3 @ p_cam

            print(f"Recognized object model: {obj.getModel()} , ID: {pid}")
            print(f"  Camera Frame (x,y,z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"  Position on image (x,y): [{image_pos[0]:.3f}, {image_pos[1]:.3f}]")
            print(f"  Base Frame (x,y,z): [{p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f}]")
            print(f"Grasped object IDs: {self.grasped_ids}")

            if p_base[0] > 0 and pid not in self.grasped_ids and pid < min_id:
                target = p_base
                target_id = pid
                self.position_on_image = image_pos
                min_id = pid

        if target is not None:
            # First alignment using base movement and IK
            if not self.ik_done:
                aligned = self.control_base(target)
                if aligned:
                    joint_targets = self.calculate_ik(
                        target[:3], self.IK_offset, apply=True
                    )
                    print("Grasp configuration reached via IK.")
                    self.wait_for_arm(joint_targets)
                    self.ik_done = True
            else:
                # Final approach and grasp
                if target[0] <= 1.060:
                    print("Close enough to object. Stopping.")
                    self.left_motor.setVelocity(0)
                    self.right_motor.setVelocity(0)

                    self.close_gripper()
                    for _ in range(100):
                        self.robot.step(self.timestep)

                    if target_id not in self.grasped_ids:
                        self.grasped_ids.append(target_id)
                        self.save_grasped_ids()

                    print("Object grasped")

                    print("Lift")
                    self.lift_arm(0.350)

                    print("Repositioning arm...")
                    joint_targets = self.calculate_ik(
                        target[:3], np.array([-0.45, 0.0, 0.0]), apply=True
                    )
                    print("Repositioned via IK")
                    self.wait_for_arm(joint_targets)

                    for _ in range(200):
                        self.robot.step(self.timestep)

                    return Status.SUCCESS
                else:
                    print("Driving slowly toward object...")
                    self.left_motor.setVelocity(0.75)
                    self.right_motor.setVelocity(0.75)

        else:
            # No valid object found — scan by rotating
            print("No valid target object detected. Continuing...")

            if not hasattr(self, 'scan_direction'):
                self.scan_direction = 1  # 1 for left, -1 for right
                self.scan_counter = 0

            scan_limit = 160

            if self.scan_direction == 1:
                self.left_motor.setVelocity(-0.75)
                self.right_motor.setVelocity(0.75)
            else:
                self.left_motor.setVelocity(0.75)
                self.right_motor.setVelocity(-0.75)

            self.scan_counter += 1

            if self.scan_counter > scan_limit:
                self.scan_direction *= -1
                self.scan_counter = 0

        return Status.RUNNING
