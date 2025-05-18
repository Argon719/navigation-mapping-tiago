import py_trees  # Behavior tree framework for control logic
from py_trees.common import Status  # Node status: SUCCESS, FAILURE, RUNNING
from controller import Motor  # Webots motor device
import numpy as np  # For vector calculations
import math  # For heading and angle math


class MoveToAngleOrDistance(py_trees.behaviour.Behaviour):
    """
    Behavior that uses PID control to either rotate the robot by a specified relative angle
    or drive it forward/backward a specific distance.
    """

    def __init__(self, name, blackboard, mode, target, speed=3.14):
        """
        Initialize the behavior.

        :param name: Node name for the behavior tree
        :param blackboard: Shared data container (contains robot instance)
        :param mode: Either "turn" (angle) or "drive" (distance)
        :param target: Target angle (degrees) or distance (meters)
        :param speed: Maximum speed (rad/s or m/s depending on mode)
        """
        super(MoveToAngleOrDistance, self).__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')
        self.mode = mode
        self.target = target
        self.speed = speed
        self.initialized = False

        # PID control variables
        self.angle_integral = 0
        self.angle_prev_error = 0
        self.dist_integral = 0
        self.dist_prev_error = 0
        
    def setup(self):
        """
        Called once at the beginning of the simulation.
        Initializes motors and sensors.
        """
        self.timestep = int(self.robot.getBasicTimeStep())

        # Initialize motors
        self.left_motor = self.robot.getDevice("wheel_left_joint")
        self.right_motor = self.robot.getDevice("wheel_right_joint")
        self.left_motor.setPosition(float('inf'))  # Enable velocity control
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Enable sensors
        self.gps = self.robot.getDevice('gps')
        self.compass = self.robot.getDevice('compass')
        self.gps.enable(self.timestep)
        self.compass.enable(self.timestep)

    def initialise(self):
        """
        Called when the behavior is entered.
        Stores the initial position and heading.
        """
        self.initialized = True
        self.start_position = self.gps.getValues()

        compass_values = self.compass.getValues()
        current_heading = math.degrees(math.atan2(compass_values[0], compass_values[1]))
        if current_heading > 180:
            current_heading -= 360

        if self.mode == "turn":
            # Compute absolute target angle in world coordinates
            self.target_angle = (current_heading + self.target) % 360
            if self.target_angle > 180:
                self.target_angle -= 360

            self.angle_integral = 0
            self.angle_prev_error = 0

        elif self.mode == "drive":
            # Store direction of current heading
            heading_rad = math.atan2(compass_values[0], compass_values[1])
            self.drive_direction = np.array([math.cos(heading_rad), math.sin(heading_rad)])

            self.dist_integral = 0
            self.dist_prev_error = 0

        print(f"[{self.name}] Initialized → Mode: {self.mode}, Start Heading: {current_heading:.2f}°, Target: {self.target}")
        
    def update(self):
        """
        Called every timestep. Executes either turn or drive PID control.
        Returns py_trees.common.Status.
        """
        dt = self.timestep / 1000.0  # Convert ms to seconds

        position = self.gps.getValues()
        compass_values = self.compass.getValues()

        # Get current heading in degrees
        current_heading = math.degrees(math.atan2(compass_values[0], compass_values[1]))
        if current_heading > 180:
            current_heading -= 360

        if self.mode == "turn":
            # Calculate angular error in shortest direction [-180, +180]
            error = (self.target_angle - current_heading + 180) % 360 - 180

            # PID control for rotation
            self.angle_integral += error * dt
            derivative = (error - self.angle_prev_error) / dt if dt > 0 else 0

            # PID coefficients (manually tuned)
            Kp, Ki, Kd = 0.2, 0.00002, 0.05
            angular_speed = Kp * error + Ki * self.angle_integral + Kd * derivative
            angular_speed = max(min(angular_speed, self.speed), -self.speed)

            # Apply velocity to motors
            self.left_motor.setVelocity(-angular_speed)
            self.right_motor.setVelocity(angular_speed)

            self.angle_prev_error = error

            print(f"[{self.name}] Turning → Δ: {error:.2f}°, Speed: {angular_speed:.2f}")

            # Threshold for successful rotation
            if abs(error) < 0.01:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                print(f"[{self.name}] Turn complete")
                return Status.SUCCESS

            return Status.RUNNING

        elif self.mode == "drive":
            # Compute projected distance along initial heading
            dx = position[0] - self.start_position[0]
            dy = position[1] - self.start_position[1]
            moved_vector = np.array([dx, dy])
            moved_distance = np.dot(moved_vector, self.drive_direction)
            error = self.target - moved_distance

            # PID control for driving
            self.dist_integral += error * dt
            derivative = (error - self.dist_prev_error) / dt if dt > 0 else 0

            # PID coefficients for distance
            Kp, Ki, Kd = 10.0, 0.002, 2.0
            linear_speed = Kp * error + Ki * self.dist_integral + Kd * derivative
            linear_speed = max(min(linear_speed, self.speed), -self.speed)

            # Apply velocity to both wheels equally
            self.left_motor.setVelocity(linear_speed)
            self.right_motor.setVelocity(linear_speed)

            self.dist_prev_error = error

            print(f"[{self.name}] Driving → Travelled: {moved_distance:.3f}m / Target: {self.target:.3f}m, Speed: {linear_speed:.2f}")

            # Threshold for success
            if abs(error) < 0.001:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                print(f"[{self.name}] Drive complete")
                return Status.SUCCESS

            return Status.RUNNING

        else:
            # Invalid mode
            print(f"[{self.name}] Unknown mode: {self.mode}")
            return Status.FAILURE

    def terminate(self, new_status):
        """
        Called when the behavior ends (either success, failure, or interruption).
        Stops both motors and resets initialization flag.
        
        :param new_status: py_trees.common.Status that caused the termination
        """
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.initialized = False

        print(f"[{self.name}] Terminated with status: {new_status.name}")
        
