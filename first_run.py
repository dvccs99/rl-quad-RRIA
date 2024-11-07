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




# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Discretize the action space (assuming each dimension has 3 choices: [-1, 0, 1])
action_choices = [-1, 0, 1]
action_space_size = len(action_choices) ** env.action_space.shape[0]

class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Hyperparameters
learning_rate = 1e-3
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 500
target_update_frequency = 10
batch_size = 64
memory_size = 10000

# Initialize DQN and target network
obs_dim = env.observation_space.shape[0]
action_dim = action_space_size
q_network = DQN(obs_dim, action_dim).to(device)
target_network = DQN(obs_dim, action_dim).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# Helper functions
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)  # Random action
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return q_network(state).argmax().item()

def compute_loss(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Ensure all tensors are of type float32
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)  # Actions are typically integers
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = target_network(next_states).max(1)[0]
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target)
    return loss

# Training loop
num_episodes = 500
max_steps = 200
epsilon = epsilon_start
for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)  # Ensure state is float32
    total_reward = 0
    
    for step in range(max_steps):
        # Select action and perform step
        action_idx = select_action(state, epsilon)
        action = np.array([action_choices[action_idx // 3**i % 3] for i in range(env.action_space.shape[0])], dtype=np.float32)
        next_state, reward, done, _, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)  # Ensure next_state is float32
        
        # Store in memory
        memory.append((state, action_idx, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        # Sample from memory and update the Q-network
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            loss = compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update the target network
        if step % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        if done:
            break

    # Update epsilon
    epsilon = max(epsilon_end, epsilon * np.exp(-1 / epsilon_decay))
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()