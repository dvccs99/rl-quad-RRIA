from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium import utils
from typing import Dict, Tuple, Union

base_dir = Path(__file__).resolve().parent

ROBOT_PATH = str(base_dir / "robot/anybotics_anymal_c/scene.xml")
DEFAULT_CAMERA = {
    "distance": 4.0,
}


class QuadEnv(MujocoEnv, utils.EzPickle):
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
        forward_reward_weight: float = 12,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.2,
        orientation_cost_weight: float = 2,
        running_time_constant: float = 0.01,
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.4, 0.9),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions: bool = True, # TODO: useless fix this later
        include_contact_forces: bool = False,
        include_qvel: bool = False,
        render_mode="rgb_array",
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions,
            include_contact_forces,
            include_qvel,
            render_mode,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._orientation_cost_weight = orientation_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._main_body = main_body
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions = (exclude_current_positions)
        self._include_contact_forces = include_contact_forces
        self._include_qvel = include_qvel
        self._running_time_constant = running_time_constant
        self.render_mode = render_mode
        self.running_time = 0

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

        obs_size = 12
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions,
            "qpos": self.data.qpos.size - 7,
            "qvel": self.data.qvel.size * include_qvel,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_contact_forces,
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
        is_healthy = np.isfinite(state).all() and self._healthy_z_range[0] <= state[2]
        return is_healthy

    @property
    def healthy_reward(self) -> float:
        """
        Calculates the reward given if the robot is healthy.

        Returns:
            float: reward value
        """
        return self.is_healthy * self._healthy_reward + self.running_time

    @property
    def orientation_cost(self) -> float:
        """
        Calculates the cost given if the robot maintains a non-reasonable
        orientation.

        Returns:
            float: cost
        """
        euler_angles = R.from_quat(self.data.qpos[4:8]).as_euler(seq='xyz')
        roll = euler_angles[0]
        pitch = euler_angles[1]
        cost = np.linalg.norm(pitch) * self._orientation_cost_weight
        cost += np.linalg.norm(roll) * self._orientation_cost_weight
        return cost

    def control_cost(self, action: np.ndarray) -> float:
        """
        Gives the cost associated to the size of the taken action.

        Args:
            action (np.ndarray): Torques of all 12 joints.

        Returns:
            float: control cost
        """
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
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
        contact_cost = self._contact_cost_weight * contact_forces_value
        return contact_cost

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """


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
        self.running_time += self._running_time_constant
        xy_position_before = self.data.body(1).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(1).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self.__get_obs()
        reward, reward_info = self.__get_rew(x_velocity, y_velocity, action)
        terminated = not self.is_healthy
        if terminated:
            self.running_time = 0
        truncated = False
        info = reward_info

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def __get_rew(self, x_velocity: float, y_velocity: float, action) -> tuple:
        """_summary_

        Args:
            x_velocity (float): _description_
            action (_type_): _description_

        Returns:
            tuple: _description_
        """
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        orientation_cost = self.orientation_cost
        orientation_cost *= self._orientation_cost_weight
        costs = ctrl_cost + orientation_cost + contact_cost

        reward = rewards - costs
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_orientation": -orientation_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def __get_obs(self) -> np.ndarray:
        """
        Gymnasium observation function. Needs more detail.

        Returns:
            np.ndarray: _description_
        """
        position = self.data.qpos[7:].flatten()
        obs = position
        if self._include_qvel:
            velocity = self.data.qvel.flatten()
            obs = np.concatenate((obs, velocity))
        if self._include_contact_forces:
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

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel_noise = self.np_random.standard_normal(self.model.nv)
        qvel = self.init_qvel + self._reset_noise_scale * qvel_noise

        self.set_state(qpos, qvel)
        observation = self.__get_obs()
        return observation
