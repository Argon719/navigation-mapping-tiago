# === Lidar-Based Navigation Behavior Node ===
# This node enables the robot to navigate through a set of waypoints using
# Lidar-based obstacle avoidance and PID-based motion control.

import py_trees
import numpy as np
import math


def limit(value, minimum, maximum):
    """
    Clamp a value between a given minimum and maximum.
    """
    return max(min(value, maximum), minimum)


def world2map(xw, yw):
    """
    Convert world coordinates to pixel map coordinates.
    """
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))
    px = max(0, min(px, 199))
    py = max(0, min(py, 299))
    return [px, py]


class Navigation(py_trees.behaviour.Behaviour):
    """
    Behavior tree node for navigating through waypoints while avoiding obstacles
    using Lidar sensor data.
    """

    def __init__(self, name, blackboard):
        super().__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard

    def setup(self):
        """
        Initialize all required devices and variables.
        """
        self.node = 'lidar_navigation.py'
        self.timestep = int(self.robot.getBasicTimeStep())

        # Enable sensors
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Motors
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Display and waypoint marker
        self.wp_marker = self.robot.getFromDef("WpMarker").getField("translation")
        self.display = self.robot.getDevice('display')
        self.map = np.zeros((200, 300))
        self.counter = 0

    def initialise(self):
        """
        Reset navigation state and PID parameters at the beginning.
        """
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.index = 3  # Start from waypoint 4
        print(f"{self.node} -> [{self.name}] → Status: INITIALISING  Start navigation.")

        self.WAYPOINTS = self.blackboard.read('waypoints')
        self.angles = np.linspace(4.19 / 2, -4.19 / 2, 667)
        self.filtered_angles = self.angles[80:-80]

        self.prev_alpha = 0.0
        self.integral_alpha = 0.0
        self.prev_rho = 0.0
        self.integral_rho = 0.0

    def update(self):
        """
        Periodically called to update robot motion and map based on Lidar and GPS.
        """
        self.counter += 1

        # Get current robot pose
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        heading = math.atan2(self.compass.getValues()[0], self.compass.getValues()[1])

        # Draw trajectory
        px, py = world2map(xw, yw)
        self.display.setColor(0xFF0000)
        self.display.drawPixel(px, py)

        # Compute distance and angle to current waypoint
        dx = xw - self.WAYPOINTS[self.index][0]
        dy = yw - self.WAYPOINTS[self.index][1]
        rho = math.sqrt(dx ** 2 + dy ** 2)

        alpha = math.atan2(
            self.WAYPOINTS[self.index][1] - yw,
            self.WAYPOINTS[self.index][0] - xw
        ) - heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]

        # Process Lidar data
        ranges = np.array(self.lidar.getRangeImage())
        ranges[ranges == np.inf] = 100
        filtered_ranges = ranges[80:-80]
        num_data_points = len(filtered_ranges)

        # Compute Lidar point cloud in world frame
        w_T_r = np.array([
            [math.cos(heading), -math.sin(heading), xw],
            [math.sin(heading),  math.cos(heading), yw],
            [0, 0, 1]
        ])

        X_i = np.array([
            filtered_ranges * np.cos(self.filtered_angles) + 0.202,
            filtered_ranges * np.sin(self.filtered_angles),
            np.ones(num_data_points)
        ])

        D = w_T_r @ X_i

        # Update map display with Lidar points
        for d in D.T:
            px, py = world2map(d[0], d[1])
            self.map[px, py] = min(self.map[px, py] + 0.01, 1.0)
            v = int(self.map[px, py] * 255)
            gray = v * 256**2 + v * 256 + v
            self.display.setColor(gray)
            self.display.drawPixel(px, py)

        # === PID motion control ===
        dt = self.timestep / 1000.0

        Kp_a, Ki_a, Kd_a = 4.75, 0.00475, 0.0475
        Kp_r, Ki_r, Kd_r = 3.5, 0.00035, 0.35
        MAX_SPEED = 6.28

        angular_error = alpha
        self.integral_alpha += angular_error * dt
        angular_derivative = (angular_error - self.prev_alpha) / dt if dt > 0 else 0.0
        angular_speed = Kp_a * angular_error + Ki_a * self.integral_alpha + Kd_a * angular_derivative
        self.prev_alpha = angular_error

        linear_error = rho
        self.integral_rho += linear_error * dt
        linear_derivative = (linear_error - self.prev_rho) / dt if dt > 0 else 0.0
        linear_speed = Kp_r * linear_error + Ki_r * self.integral_rho + Kd_r * linear_derivative
        self.prev_rho = linear_error

        # === Obstacle avoidance logic (7 sectors) ===
        if self.counter % 2 != 0:
            sectors = np.array_split(filtered_ranges, 7)
            min_distances = [np.min(s) if s.size else 100 for s in sectors]
            FL, L, ML, F, MR, R, FR = min_distances

            front_thresh = 0.6
            side_thresh = 0.4

            if F < front_thresh:
                if L + ML > R + MR:
                    print(f"{self.node} ->[{self.name}] Obstacle ahead, turning left")
                    angular_speed = MAX_SPEED / 2
                    linear_speed = 0
                else:
                    print(f"{self.node} ->[{self.name}] Obstacle ahead, turning right")
                    angular_speed = -MAX_SPEED / 2
                    linear_speed = 0
            elif L < side_thresh or FL < side_thresh:
                print(f"{self.node} ->[{self.name}] Obstacle on left, turning right")
                angular_speed = -MAX_SPEED / 4
                linear_speed = MAX_SPEED / 2
            elif R < side_thresh or FR < side_thresh:
                print(f"{self.node} ->[{self.name}] Obstacle on right, turning left")
                angular_speed = MAX_SPEED / 4
                linear_speed = MAX_SPEED / 2

        # Apply final wheel velocities
        left_speed = limit(linear_speed - angular_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = limit(linear_speed + angular_speed, -MAX_SPEED, MAX_SPEED)

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # === Waypoint reached ===
        if rho < 0.5:
            self.index += 1
            print(f"{self.node} ->[{self.name}] WayPoint: {self.index} / {len(self.WAYPOINTS)} - ρ: {rho:.3f} m - α: {math.degrees(alpha):.1f}°")

            if self.index == len(self.WAYPOINTS):
                print(f"{self.node} ->[{self.name}] → Status: SUCCESS Navigation complete.")
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Stop motors when the behavior is terminated.
        """
        print(f"{self.node} ->[{self.name}] → Status: {new_status.name} Navigation terminated.")
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
