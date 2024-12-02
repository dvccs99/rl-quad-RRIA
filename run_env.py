from quad_env import QuadEnv
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
import wandb


env_parameters = {
    'FORWARD_REWARD_WEIGHT': 1,
    'CONTROL_COST_WEIGHT': 0.05,
    'CONTACT_COST_WEIGHT': 5e-4,
    'HEALTHY_REWARD_WEIGHT': 1,
    'MAIN_BODY': 1,
    'HEALTHY_Z_RANGE': (0.195, 0.75),
    'RESET_NOISE_SCALE': 0.1,
    'CONTACT_FORCE_RANGE': (-1.0, 1.0)
}

mujoco_parameters = {
    'frame_skip': 50,
    'observation_space': None,
    'render_mode': 'human'
}

env = QuadEnv(env_parameters, mujoco_parameters)
env = Monitor(env)

run = wandb.init(
    project="Quad_Mujoco",
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

model = SAC("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}")

vec_env = model.get_env()

model.learn(
    total_timesteps=15000,
    callback=WandbCallback(
        verbose=1,
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
