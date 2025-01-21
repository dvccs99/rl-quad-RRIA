from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from typing import Dict, Tuple, Union

base_dir = Path(__file__).resolve().parent
ROBOT_PATH = str(base_dir / "robot/anybotics_anymal_c/scene.xml")
QPOS_ROBOT_POSE_SIZE = 7
DEFAULT_CAMERA = {
    "distance": 4.0,
}


class QuadEnv(MujocoEnv):
    """
    Custom Gymnasium environment used for training a quadruped robot with
    deep RL.
    """

    ENVIRONMENT_NAME = "QuadEnv - Reinforcement Learning Environment"
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file: str = ROBOT_PATH,
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA,
        forward_reward_weight: float = 5,
        position_reward_weight: float = 9,
        control_cost_weight: float = 0.05,
        contact_cost_weight: float = 5e-4,
        action_change_cost_weight: float = 1,
        healthy_reward_value: float = 10,
        orientation_cost_weight: float = 4,
        forward_sigma: float = 0.25,
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = False,
        healthy_z_range: Tuple[float, float] = (0.35, 0.9),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        include_contact_forces: bool = False,
        include_qvel: bool = True,
        render_mode="rgb_array",
        **kwargs,
    ):
        self.forward_reward_weight = forward_reward_weight
        self.control_cost_weight = control_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.action_change_cost_weight = action_change_cost_weight
        self.orientation_cost_weight = orientation_cost_weight
        self.position_reward_weight = position_reward_weight
        self.healthy_reward_value = healthy_reward_value
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_z_range = healthy_z_range
        self.contact_force_range = contact_force_range
        self.main_body = main_body
        self.reset_noise_scale = reset_noise_scale
        self.include_contact_forces = include_contact_forces
        self.include_qvel = include_qvel
        self.forward_sigma = forward_sigma
        self.render_mode = render_mode
        self.current_action = None
        self.previous_action = None

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size - QPOS_ROBOT_POSE_SIZE
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "qpos": self.data.qpos.size - QPOS_ROBOT_POSE_SIZE,
            "qvel": self.data.qvel.size * include_qvel,
            "contact_forces": self.data.cfrc_ext[1:0] * include_contact_forces,
        }

    @property
    def is_healthy(self) -> bool:
        """
        Determines if the robot is "healthy", that is, if the values of its
        state space are finite and its z coordinate is inside the specified
        parameters.

        Returns:
            bool: True if the robot is healthy, False otherwise
        """
        state = self.state_vector()
        is_healthy = np.isfinite(state).all() and self.healthy_z_range[0] <= state[2]
        return is_healthy

    @property
    def healthy_reward(self) -> float:
        """
        Calculates the reward given if the robot is healthy.

        Returns:
            float: reward value
        """
        return self.is_healthy * self.healthy_reward_value

    @property
    def orientation_cost(self) -> float:
        """
        Calculates the cost given if the robot maintains a non-reasonable
        orientation.

        Returns:
            float: cost
        """
        ANYMAL_C_STANDARD_HEIGHT = 0.6

        euler_angles = R.from_quat(self.data.qpos[4:8]).as_euler(seq='xyz')
        roll = euler_angles[0]
        roll_norm = np.linalg.norm(roll)

        pitch = euler_angles[1]
        pitch_norm = np.linalg.norm(pitch)

        state = self.state_vector()
        z = state[2]
        z_norm = np.linalg.norm(z-ANYMAL_C_STANDARD_HEIGHT)

        cost = self.orientation_cost_weight * (pitch_norm + roll_norm + z_norm)
        return self.orientation_cost_weight * cost

    @property
    def action_change_cost(self) -> float:
        """
        Calculates the cost associated to the change in the action taken by the
        robot.

        Returns:
            float: cost value
        """
        if self.previous_action is None:
            return 0

        action_difference = self.current_action - self.previous_action
        action_change = np.square(np.linalg.norm(action_difference))

        return self.action_change_cost_weight * action_change

    def control_cost(self, action: np.ndarray) -> float:
        """
        Gives the cost associated to the size of the taken action.

        Args:
            action (np.ndarray): Torques of all 12 joints.

        Returns:
            float: control cost
        """
        control_cost = self.control_cost_weight * (np.sum(np.square(action)) + np.sum(np.abs(action)))
        return control_cost

    @property
    def contact_forces(self) -> np.ndarray:
        """
        Returns the external contact forces suffered by the model, clipped by
        the minimal contact force and the maximal contact force.

        Returns:
            np.ndarray: array containing the clipped forces.
        """
        MIN_CONTACT_FORCE = -1.0
        MAX_CONTACT_FORCE = 1.0

        raw_contact_forces = self.data.cfrc_ext
        contact_forces = np.clip(
            raw_contact_forces, MIN_CONTACT_FORCE, MAX_CONTACT_FORCE
        )
        return contact_forces

    @property
    def contact_cost(self) -> float:
        """
        Cost created by external contact forces.

        Returns:
            float: cost value
        """
        contact_forces_value = np.sum(np.square(self.contact_forces))
        contact_cost = self.contact_cost_weight * contact_forces_value
        return contact_cost

    @property
    def position_reward(self) -> float:
        """
        Reward based on the x position of the robot.

        Returns:
            float: reward value
        """
        current_x_position = self.data.qpos[0]
        if current_x_position < 0:
            return 0
        else:
            return current_x_position * self.position_reward_weight

    def forward_reward(self, x_velocity) -> float:
        """
        Reward based on the x velocity of the robot.

        Returns:
            float: reward value
        """
        if x_velocity * self.forward_reward_weight > 10:
            forward_reward = 5
        else:
            forward_reward = x_velocity * self.forward_reward_weight
        return forward_reward

    def execute_action(self, action: np.ndarray) -> tuple:
        """
        Executes the given action and returns the x and y velocity of the robot

        Args:
            action (np.ndarray): Torques of all 12 joints

        Returns:
            tuple: x and y velocity of the robot
        """
        xy_position_before = self.data.body(1).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(1).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        return x_velocity, y_velocity

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Environment step function. It receives the action to be taken by the
        robot and returns the observation, reward, termination status,
        truncation status, and additional information.

        Args:
            action (np.ndarray): Torques of all 12 joints

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]:
                - Observation
                - Reward
                - Terminated
                - Truncated: Is set to false as the time limit is handled by
                  the `TimeLimit` wrapper added during `make`
                - Info

        """
        if self.previous_action is None:
            self.previous_action = action
            self.current_action = action
        else:
            self.previous_action = self.current_action
            self.current_action = action

        x_velocity, _ = self.execute_action(action)
        observation = self.__get_obs()
        reward, reward_info = self.__get_rew(x_velocity, action)
        terminated = not self.is_healthy
        truncated = False
        info = reward_info

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def __get_rew(self, x_velocity: float, action: np.ndarray) -> tuple:
        """
        Calculates the reward and reward information based on the x and y
        velocity of the robot, the action taken, and the current state of the
        robot. The forward reward is calculated based on the x velocity of the
        robot and has a maximum value of 5. The position reward is calculated
        based on the x position of the robot and is zero if the x position is
        zero.

        Args:
            x_velocity (float): x velocity of the robot
            action (np.ndarray): Torques of all 12 joints
        Returns:
            tuple: reward and reward information
        """
        position_reward = self.position_reward
        healthy_reward = self.healthy_reward
        forward_reward = self.forward_reward(x_velocity)
        rewards = forward_reward + healthy_reward + position_reward

        control_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        orientation_cost = self.orientation_cost
        action_change_cost = self.action_change_cost
        costs = control_cost + orientation_cost + action_change_cost

        reward = rewards - costs
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -control_cost,
            "reward_contact": -contact_cost,
            "reward_orientation": -orientation_cost,
            "reward_survive": healthy_reward,
            "reward_action_change": -action_change_cost,
            "reward_position": position_reward,
        }

        return reward, reward_info

    def __get_obs(self) -> np.ndarray:
        """
        Environment observation function. It returns the observation of the
        environment, which is the position of the joints, the velocity of the
        joints, and the contact forces.

        Returns:
            np.ndarray: observation
        """
        position = self.data.qpos[QPOS_ROBOT_POSE_SIZE:].flatten()
        obs = position

        if self.include_qvel:
            velocity = self.data.qvel.flatten()
            obs = np.concatenate((obs, velocity))
        if self.include_contact_forces:
            contact_force = self.contact_forces[1:].flatten()
            obs = np.concatenate((obs, contact_force))

        obs = np.array(obs, dtype=np.float32)
        return obs

    def reset_model(self) -> np.ndarray:
        """
        Resets the model's joint positions and joint velocities.

        Returns:
            np.ndarray: observation after reset.
        """

        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel_noise = self.np_random.standard_normal(self.model.nv)
        qvel = self.init_qvel + self.reset_noise_scale * qvel_noise

        self.set_state(qpos, qvel)
        observation = self.__get_obs()
        return observation
