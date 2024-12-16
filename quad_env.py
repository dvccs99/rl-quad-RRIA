from typing import Union
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class QuadEnv(MujocoEnv):
    """
    """

    def __init__(self, env_parameters, mujoco_parameters):
        self.HEALTHY_REWARD_WEIGHT = env_parameters["HEALTHY_REWARD_WEIGHT"]
        self.TRACKING_REWARD_WEIGHT = env_parameters["TRACKING_REWARD_WEIGHT"]
        self.RESET_NOISE_SCALE = env_parameters["RESET_NOISE_SCALE"]
        self.FORWARD_REWARD_WEIGHT = env_parameters["FORWARD_REWARD_WEIGHT"]
        self.CONTROL_COST_WEIGHT = env_parameters["CONTROL_COST_WEIGHT"]
        self.CONTACT_COST_WEIGHT = env_parameters["CONTACT_COST_WEIGHT"]
        self.CONTACT_FORCE_RANGE = env_parameters["CONTACT_FORCE_RANGE"]
        self._main_body: Union[int, str] = 1

        self.metadata = {"render_modes": [
                         "human",
                         "rgb_array",
                         "depth_array"]}

        MujocoEnv.__init__(self,
                           model_path="./robot/anybotics_anymal_c/scene.xml",
                           frame_skip=mujoco_parameters['frame_skip'],
                           observation_space=None,  # needs to be defined after
                           default_camera_config={"distance": 4.0},
                           camera_name='track',
                           render_mode=mujoco_parameters['render_mode'])

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        obs_size = self.data.qpos.size + self.data.qvel.size

        # TODO: EXCLUDE X AND Y?
        obs_size -= 2
        obs_size += self.data.cfrc_ext[1:].size

        self.observation_space = Box(low=-np.inf,
                                     high=np.inf,
                                     shape=(obs_size,),
                                     dtype=np.float64)

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
        return self.is_healthy * self.HEALTHY_REWARD_WEIGHT

    def control_cost(self, action: np.ndarray) -> float:
        """
        Gives the cost associated to the size of the taken action.

        Args:
            action (np.ndarray): Torques of all 12 joints.

        Returns:
            float: control cost
        """
        control_cost = self.CONTROL_COST_WEIGHT * np.sum(np.square(action))
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
        contact_cost = self.CONTACT_COST_WEIGHT * contact_forces_value
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

        xy_position_before = self.data.body(1).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(1).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # TODO: PASSAR A VELOCIDADE ANGULAR PARA GET_REW

        observation = self.__get_obs()
        reward, reward_info = self.__get_rew(x_velocity, y_velocity, action)
        terminated = not self.is_healthy
        truncated = False
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
        return observation, reward, terminated, truncated, info

    def __get_rew(self, x_velocity: float, y_velocity: float, action) -> tuple:
        """_summary_

        Args:
            x_velocity (float): _description_
            action (_type_): _description_

        Returns:
            tuple: _description_
        """
        forward_reward = x_velocity * self.FORWARD_REWARD_WEIGHT
        healthy_reward = self.healthy_reward
        vel_reward = self.reward_tracking_lin_vel(commands=action,
                                                  x_velocity=x_velocity,
                                                  y_velocity=y_velocity)

        rewards = forward_reward + healthy_reward + vel_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {"reward_forward": forward_reward,
                       "reward_ctrl": -ctrl_cost,
                       "reward_contact": -contact_cost,
                       "reward_survive": healthy_reward}

        return reward, reward_info

    def __get_obs(self) -> np.ndarray:
        """
        Gymnasium observation function. Needs more detail.

        Returns:
            np.ndarray: _description_
        """
        position = self.data.qpos.flatten()
        position = position[2:]
        velocity = self.data.qvel.flatten()
        contact_force = self.contact_forces[1:].flatten()
        obs = np.concatenate((position, velocity, contact_force))
        print(obs.shape)
        return obs

    def reset_model(self) -> np.ndarray:
        """
        Resets the model's joint positions and joint velocities.

        Returns:
            np.ndarray: observation after reset.
        """

        noise_low = -self.RESET_NOISE_SCALE
        noise_high = self.RESET_NOISE_SCALE

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)

        qvel_noise = self.np_random.standard_normal(self.model.nv)
        qvel = (self.init_qvel + self.RESET_NOISE_SCALE * qvel_noise)

        self.set_state(qpos, qvel)
        observation = self.__get_obs()
        return observation

    def sample_command(self) -> np.array:
        """


        Returns:
            np.array: _description_
        """
        MAX_LINEAR_SPEED = 1
        MAX_ANGULAR_SPEED = 0.5

        rng = np.random.default_rng(seed=3141592)

        lin_vel_x, lin_vel_y = rng.uniform(low=-MAX_LINEAR_SPEED,
                                           high=MAX_LINEAR_SPEED,
                                           size=2)

        ang_vel_yaw = rng.uniform(low=-MAX_ANGULAR_SPEED,
                                  high=MAX_ANGULAR_SPEED)

        cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_yaw])
        return cmd

    def reward_tracking_lin_vel(
        self,
        commands: np.array,
        x_velocity,
        y_velocity
            ) -> np.array:

        distribution_sigma = self.TRACKING_REWARD_WEIGHT

        current_vel = np.array([x_velocity, y_velocity])
        vel_command = np.array([commands[0], commands[1]])

        lin_vel_error = np.sum(np.square(vel_command - current_vel))
        lin_vel_reward = np.exp(-lin_vel_error / distribution_sigma)

        return lin_vel_reward


    # TODO implementar reward pra velocidade angular usando cvel
