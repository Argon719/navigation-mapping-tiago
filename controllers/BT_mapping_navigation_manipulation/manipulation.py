# Import py_trees for behavior tree implementation
import py_trees


def initialize_joints(robot, positions, timestep):
    """
    Initialize the robot's joints by setting their target positions and enabling encoders.

    Parameters:
        robot (Supervisor): The robot controller instance.
        positions (dict): A dictionary mapping joint names to their target positions.
        timestep (int): The simulation timestep.

    Returns:
        tuple: Two dictionaries containing motor and encoder devices keyed by joint names.
    """
    motors = {}    # Dictionary to store motor devices
    encoders = {}  # Dictionary to store encoder devices

    for joint_name, target_position in positions.items():
        # Retrieve the motor device for the joint
        motor = robot.getDevice(joint_name)
        motors[joint_name] = motor
        motor.setPosition(target_position)  # Set the target position for the motor

        # Determine the corresponding encoder name based on joint naming conventions
        if joint_name.startswith("gripper"):
            # For gripper joints, replace 'finger_joint' with 'sensor_finger_joint'
            encoder_name = joint_name.replace("finger_joint", "sensor_finger_joint")
        else:
            # For other joints, append '_sensor' to the joint name
            encoder_name = f"{joint_name}_sensor"

        # Retrieve the encoder device for the joint
        encoder = robot.getDevice(encoder_name)
        encoder.enable(timestep)  # Enable the encoder with the given timestep
        encoders[joint_name] = encoder  # Store the encoder in the dictionary

    return motors, encoders  # Return the dictionaries of motors and encoders


def set_joints_position(robot, motors, encoders, target_positions, timestep, threshold=0.001):
    """
    Move the robot's joints to the specified target positions and wait until they reach close enough.

    Parameters:
        robot (Supervisor): The robot controller instance.
        motors (dict): Dictionary of motor devices keyed by joint names.
        encoders (dict): Dictionary of encoder devices keyed by joint names.
        target_positions (dict): Dictionary of target positions for each joint.
        timestep (int): The simulation timestep.
        threshold (float): The acceptable squared error threshold to consider the joint position as reached.
    """
    # Set the target position for each joint's motor
    for joint_name, position in target_positions.items():
        motors[joint_name].setPosition(position)

    # Continuously step the simulation until all joints reach their target positions within the threshold
    while robot.step(timestep) != -1:
        total_error = 0  # Initialize total error

        for joint_name, position in target_positions.items():
            current_position = encoders[joint_name].getValue()  # Get the current position from the encoder
            total_error += (position - current_position) ** 2  # Accumulate the squared error

        if total_error < threshold:
            break  # Exit the loop if the total error is below the threshold


class SetRobotArms(py_trees.behaviour.Behaviour):
    """
    Behavior tree node to set the robot's arms to specified positions.
    """

    def __init__(self, name, blackboard, safe_positions):
        """
        Initialize the SetRobotArms behavior.

        Parameters:
            name (str): The name of the behavior.
            blackboard (Blackboard): Reference to the shared blackboard for data storage.
            safe_positions (dict): Dictionary of joint names to their safe positions.
        """
        super(SetRobotArms, self).__init__(name)  # Initialize the parent class
        self.blackboard = blackboard  # Store the blackboard reference
        self.safe_positions = safe_positions  # Store the target safe positions
        self.robot = blackboard.read('robot')  # Retrieve the robot instance from the blackboard
        self._done = False  # Flag to indicate if the behavior has completed

    def setup(self):
        """
        Setup method to initialize devices required by the behavior.
        Called once before the behavior starts running.
        """
        self.node = 'manipulation.py'
        self.timestep = int(self.robot.getBasicTimeStep())  # Get the simulation timestep
        self.camera = self.robot.getDevice("camera")  # Retrieve the camera device
        self.camera.enable(self.timestep)  # Enable the camera with the given timestep
        self.camera.recognitionEnable(self.timestep)  # Enable object recognition on the camera

        print(f"[{self.name}] → Status: SETUP - Initializing robot arm controller")

    def update(self):
        """
        Update method called periodically to perform the behavior's actions.

        Returns:
            py_trees.common.Status: Status of the behavior (SUCCESS or RUNNING).
        """
        if self._done:
            status = py_trees.common.Status.SUCCESS
            print(f"[{self.name}] → Status: {status.name} Already completed.")
            return status

        # Retrieve the left and right wheel motors
        left_motor = self.robot.getDevice('wheel_left_joint')
        right_motor = self.robot.getDevice('wheel_right_joint')

        # Set both wheels' velocities to zero to stop the robot
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        print(f"[{self.name}] → Status: ACTION Wheels set to 0 velocity")

        # Initialize and move arms to safe positions
        motors, encoders = initialize_joints(self.robot, self.safe_positions, self.timestep)
        set_joints_position(self.robot, motors, encoders, self.safe_positions, self.timestep, threshold=0.001)

        # Optionally store motors and encoders in the blackboard for future use
        self.blackboard.write('motors', motors)
        self.blackboard.write('encoders', encoders)
        
        # for _ in range(50):
            # self.robot.step(self.timestep)

        status = py_trees.common.Status.SUCCESS
        print(f"[{self.name}] → Status: {status.name} Safe position reached. Done.")
        self._done = True  # Mark the behavior as done
        return status

    def terminate(self, new_status):
        """
        Terminate method called when the behavior is stopped or interrupted.

        Parameters:
            new_status (py_trees.common.Status): The new status after termination.
        """
        print(f"[{self.name}] → Terminate with status: {new_status.name}")  # Log the termination status
        # No additional termination actions are required in this behavior
