from custom_env import QuadEnv
from stable_baselines3 import SAC

env_parameters = {
    'forward_reward_weight': 1,
    'ctrl_cost_weight': 0.05,
    'contact_cost_weight': 5e-4,
    'healthy_reward': 1,
    'main_body': 1,
    'healthy_z_range': (0.195, 0.75),
    'include_cfrc_ext_in_observation': True,
    'exclude_current_positions_from_observation': False,
    'terminate_when_unhealthy': True,
    'reset_noise_scale': 0.1,
    'contact_force_range': (-1.0, 1.0)
}

mujoco_parameters = {
    'xml_file': './robots/boston_dynamics_spot/scene.xml',
    'frame_skip': 50,
    'observation_space': None,  # needs to be defined after
    'default_camera_config': 'default_camera_config',
    'render_mode': 'human'
}

env = QuadEnv(env_parameters, mujoco_parameters)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
env.close()
