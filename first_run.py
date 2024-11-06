import gymnasium as gym
import numpy as np

ROBOT_NAME = 'my_spot'
# Create the environment
env = gym.make(
    'Ant-v5',
    xml_file='./boston_dynamics_spot/scene.xml',
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
    max_episode_steps=1000,
    render_mode='human'
)

# Initial reset
obs, _ = env.reset()  
total_reward = 0

for step in range(1000):  # Run for 1000 steps
    action = np.zeros(env.action_space.shape)  # Create an action that keeps the robot standing still
    obs, reward, done, truncated, _ = env.step(action)  # Take a step in the environment
    total_reward += reward
    env.render()  # Render environment (optional, depending on setup)

    # if done or truncated:  # Reset if episode is done or truncated
    #     obs, _ = env.reset()
    #     print(f"Episode finished with total reward: {total_reward}")
    #     total_reward = 0

env.close()  # Close the environment