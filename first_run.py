import gymnasium as gym
import numpy as np
from custom_env import AntEnv
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

ROBOT_NAME = 'boston_dynamics_spot'

# Create the environment
env = AntEnv(
    xml_file='./robots/'+ROBOT_NAME+'/scene.xml',
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
)
# Initial reset
obs, _ = env.reset()  
total_reward = 0
while True:
    env.render()

env.close()