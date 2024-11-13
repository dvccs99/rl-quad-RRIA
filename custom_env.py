from typing import Union
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"distance": 4.0}


class QuadEnv(MujocoEnv):
    """
    """

    def __init__(self, env_parameters, mujoco_parameters):
        self.healthy_reward_weight = env_parameters["healthy_reward_weight"]
        self.reset_noise_scale = env_parameters["reset_noise_scale"]
        self.forward_reward_weight = env_parameters["forward_reward_weight"]
        self._ctrl_cost_weight = env_parameters["ctrl_cost_weight"]
        self._contact_cost_weight = env_parameters["contact_cost_weight"]
        self._contact_force_range = env_parameters["contact_force_range"]
        self._main_body: Union[int, str] = 1

        self.metadata = {"render_modes": [
                         "human",
                         "rgb_array",
                         "depth_array"]}

        MujocoEnv.__init__(self,
                           model_path=mujoco_parameters['xml_file'],
                           frame_skip=mujoco_parameters['frame_skip'],
                           observation_space=mujoco_parameters['observation_space'],  # needs to be defined after
                           default_camera_config=DEFAULT_CAMERA_CONFIG,
                           render_mode=mujoco_parameters['render_mode'])

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2
        obs_size += self.data.cfrc_ext[1:].size

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size, ), dtype=np.float64)

    @property
    def is_healthy(self) -> bool:
        """
        Determines if the robot is "healthy", that is, if the values of its
        state space are finite and its z coordinate is inside the interval
        [MIN_Z, MAX_Z].

        Returns:
            bool: True if the robot is healthy, False otherwise
        """
        MIN_Z = 0.2
        MAX_Z = 1.0

        state = self.state_vector()
        is_healthy = np.isfinite(state).all() and MIN_Z <= state[2] <= MAX_Z
        return is_healthy

    @property
    def healthy_reward(self) -> float:
        """
        Calculates the reward given if the robot is healthy.

        Returns:
            float: reward value
        """
        return self.is_healthy * self.healthy_reward_weight

    def control_cost(self, action):
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
        contact_forces = np.clip(raw_contact_forces,
                                 MIN_CONTACT_FORCE,
                                 MAX_CONTACT_FORCE)
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

    def step(self, action):
        xy_position_before = self.data.body(1).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(1).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = not self.is_healthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self.forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {"reward_forward": forward_reward,
                       "reward_ctrl": -ctrl_cost,
                       "reward_contact": -contact_cost,
                       "reward_survive": healthy_reward}

        return reward, reward_info

    def _get_obs(self) -> np.ndarray:
        """
        Gymnasium observation function. Needs more detail.

        Returns:
            np.ndarray: _description_
        """
        position = self.data.qpos.flatten()
        position = position[2:]
        velocity = self.data.qvel.flatten()
        contact_force = self.contact_forces[1:].flatten()
        return np.concatenate((position, velocity, contact_force))

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = (self.init_qvel + self.reset_noise_scale * self.np_random.standard_normal(self.model.nv))
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {"x_position": self.data.qpos[0],
                "y_position": self.data.qpos[1],
                "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2)}
