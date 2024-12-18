from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
import wandb
import rl_quad

NUM_EPISODES = 50000
NUM_TIME_STEPS = 5000000
GRADIENT_SAVE_FREQ = 100
ENVS_NUMBER = 50
MAX_EPISODE_STEPS = 10000
LEARNING_RATE = 0.003

ENV_PARAMETERS = {
    'FORWARD_REWARD_WEIGHT': 1.2,
    'TRACKING_REWARD_WEIGHT': 0.5,
    'CONTROL_COST_WEIGHT': 0.5,
    'CONTACT_COST_WEIGHT': 5e-4,
    'HEALTHY_REWARD_WEIGHT': 1,
    'MAIN_BODY': 1,
    'HEALTHY_Z_RANGE': (0.195, 0.75),
    'RESET_NOISE_SCALE': 0.1,
    'CONTACT_FORCE_RANGE': (-1.0, 1.0)
}

MUJOCO_PARAMETERS = {
    'frame_skip': 25,
    'observation_space': None,
    'render_mode': ''
}


def env_initialization(model_name: str):
    """
    Initializes vectorized SB3 environment and starts WandB run.
    Args:
        model_name (str): SB3 algorithm name

    Returns:
        List: list containing vectorized environment and Wandb run
    """
    vec_env = make_vec_env(env_id='rl_quad/quad_env',
                           n_envs=ENVS_NUMBER,
                           env_kwargs={
                            "mujoco_parameters": MUJOCO_PARAMETERS,
                            "env_parameters": ENV_PARAMETERS,
                            "max_episode_steps": MAX_EPISODE_STEPS})
    wandb.login(key="3664f3e41560a5c33e5f3f0e6e7d335e5189c5ec")
    run = wandb.init(name=model_name,
                     project="Quad_Mujoco",
                     sync_tensorboard=True,
                     monitor_gym=False,
                     save_code=True)
    return vec_env, run


def training_initialization(algorithm_model):
    """
    Initializes the SB3 algorithm, vectorized environment, WandB run and
    initial observation.

    Args:
        algorithm_model (class): SB3 algorithm to be used

    Returns:
        list: List containing all of the above output.
    """
    algorithm_name = algorithm_model.__name__
    vec_env, run = env_initialization(algorithm_name)
    device = "cpu" if algorithm_name == "PPO" else "cuda"
    model = algorithm_model(
        "MlpPolicy",
        vec_env,
        device=device,
        verbose=1,
        tensorboard_log=f"runs/{algorithm_name}",
        learning_rate=LEARNING_RATE
    )

    obs = vec_env.reset()
    model.learn(
        total_timesteps=NUM_TIME_STEPS,
        callback=WandbCallback(
            verbose=1,
            gradient_save_freq=GRADIENT_SAVE_FREQ,
            model_save_path=f'training_outputs/{algorithm_model.__name__}'
        ))

    return model, vec_env, run, obs


def train(model, vec_env, run, obs):
    """
    Training function.

    Args:
        model (class): SB3 algorithm to be used.
        vec_env (VecEnv): Vectorized quadruped environment
        run (Run): WandB Run
        obs (VecEnvObs): Vectorized environment observation
    """
    for _ in range(NUM_EPISODES):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)


if __name__ == "__main__":
    algorithms_list = [SAC, PPO, DDPG]
    for algorithm in algorithms_list:
        model, vec_env, run, obs = training_initialization(algorithm)
        train(model, vec_env, run, obs)
        vec_env.close()
        run.finish()
