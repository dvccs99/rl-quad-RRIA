from datetime import datetime
from pathlib import Path
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union, List
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base, math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model


def get_config() -> Dict:
    """
    Returns reward configuration for quadruped training environment. All
    physical quantities are in SI units, if not speficied otherwise, i.e.
    joint positions in rad, positions in meters, torques in Nm and time in
    seconds and forces in Newtons.

    Returns:
        Dict: Dictionary with configuration values.
    """

    default_config = {"tracking_lin_vel": 1.5,
                      "tracking_ang_vel": 0.8,
                      "lin_vel_z": -2.0,
                      "ang_vel_xy": -0.05,
                      "orientation": -5.0,
                      "torques": -0.0002,
                      "action_rate": -0.01,
                      "feet_air_time": 0.2,
                      "stand_still": -0.5,
                      "termination": -1.0,
                      "foot_slip": -0.1,
                      "tracking_sigma": 0.25}

    return default_config


class QuadEnv(PipelineEnv):
    """
    Environment for quadruped robot training in MJX.
    Features a joystick controller.
    """

    def __init__(
        self,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        robot_name: str = 'anybotics_anymal_c',
        **kwargs,
    ):
        ROBOT_SCENE_PATH = Path('rl-quad-rria', 'robots', robot_name, 'scene_mjx.xml')
        self.sys = mjcf.load(ROBOT_SCENE_PATH)
        self._dt = 0.02  # this environment is 50 fps
        self.sys = self.sys.tree_replace({'opt.timestep': 0.004})

        n_frames = kwargs.pop('n_frames', int(self._dt / self.sys.opt.timestep))
        super().__init__(self.sys, backend='mjx', n_frames=n_frames)

        self.reward_config = get_config()
        self._torso_idx = mujoco.mj_name2id(
                          self.sys.mj_model,
                          mujoco.mjtObj.mjOBJ_BODY.value,
                          'torso'
                        )
        self.action_scale = action_scale
        self.obs_noise = obs_noise
        self.kick_vel = kick_vel
        self.init_q = jp.array(self.sys.mj_model.keyframe('standing').qpos)
        self.default_pose = self.sys.mj_model.keyframe('standing').qpos[7:]
        self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jp.array([0.52, 2.1, 2.1] * 4)

        self._foot_radius = 0.0175
        self._nv = self.sys.nv

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.6, 1.5]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self.init_q, jp.zeros(self._nv))

        state_info = {
            'rng': rng,
            'last_act': jp.zeros(12),
            'last_vel': jp.zeros(12),
            'command': self.sample_command(key),
            'last_contact': jp.zeros(4, dtype=bool),
            'feet_air_time': jp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'kick': jp.array([0.0, 0.0]),
            'step': 0,
        }

        obs_history = jp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2)
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

        # kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self.kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # physics step
        motor_targets = self.default_pose + action * self.action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # # foot contact data based on z-position
        # foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        # foot_contact_z = foot_pos[:, 2] - self._foot_radius
        # contact = foot_contact_z < 1e-3  # a mm or less off the floor
        # contact_filt_mm = contact | state.info['last_contact']
        # contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        # first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        # state.info['feet_air_time'] += self.dt

        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # reward
        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x),
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            # 'feet_air_time': self._reward_feet_air_time(
            #     state.info['feet_air_time'],
            #     first_contact,
            #     state.info['command'],
            # ),
            # 'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step']),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        # state.info['feet_air_time'] *= ~contact_filt_mm
        # state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # sample new command if more than 500 timesteps achieved
        state.info['command'] = jp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        # reset the step counter when done
        state.info['step'] = jp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate([
            jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
            math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
            state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
            pipeline_state.q[7:] - self.default_pose,           # motor angles
            state_info['last_act'],                              # last action
        ])

        # clip, noise
        obs = jp.clip(obs, -100.0, 100.0) + self.obs_noise * jax.random.uniform(
            state_info['rng'], obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        return obs

# ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(
        self, act: jax.Array, last_act: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_config['tracking_sigma']
        )
        return lin_vel_reward

# Tracking of angular velocity commands (yaw)

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:

        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config['tracking_sigma'])

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    # Penalize motion at zero commands
    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        return jp.sum(jp.abs(joint_angles - self.default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    # def _reward_foot_slip(
    #     self, pipeline_state: base.State, contact_filt: jax.Array
    # ) -> jax.Array:
    #     # get velocities at feet which are offset from lower legs
    #     pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
    #     feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
    #     offset = base.Transform.create(pos=feet_offset)
    #     foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
    #     foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

    #     # Penalize large feet velocity for feet that are in contact with the ground.
    #     return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def render(
        self, trajectory: List[base.State], camera: str | None = None,
        width: int = 240, height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or 'track'
        return super().render(trajectory, camera=camera, width=width, height=height)
