from custom_env import QuadEnv
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb


env_parameters = {
    'forward_reward_weight': 1,
    'ctrl_cost_weight': 0.05,
    'contact_cost_weight': 5e-4,
    'healthy_reward_weight': 1,
    'main_body': 1,
    'healthy_z_range': (0.195, 0.75),
    'reset_noise_scale': 0.1,
    'contact_force_range': (-1.0, 1.0)
}

mujoco_parameters = {
    'xml_file': './robots/boston_dynamics_spot/scene.xml',
    'frame_skip': 50,
    'observation_space': None,  # needs to be defined after
    'default_camera_config': 'default_camera_config',
    'render_mode': 'rgb_array'
}

env = QuadEnv(env_parameters, mujoco_parameters)
env = Monitor(env)

run = wandb.init(
    project="Quad_Mujoco",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

model = SAC("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}")

vec_env = model.get_env()
# vec_env = VecVideoRecorder(vec_env,
#                            f"videos/{run.id}",
#                            record_video_trigger=lambda x: x % 2000 == 0,
#                            video_length=250,)

model.learn(
    total_timesteps=10000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path='models'
    ))


obs = vec_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render('human')
env.close()
run.finish()
