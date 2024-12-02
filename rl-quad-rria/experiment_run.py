from rl_quad import QuadEnv
from rl_quad import get_config
from brax import envs
import wandb
import matplotlib.pyplot as plt
import jax
from jax import numpy as jp

from brax.io import image

initial_config = get_config()
env_name = 'Quad'
envs.register_environment(env_name, QuadEnv)
env = envs.get_environment(env_name)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

for i in range(1000):
    ctrl = -0.1 * jp.ones(env.sys.nu)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)
env.render(trajectory=rollout)
