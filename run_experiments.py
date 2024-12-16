from quad_env import QuadEnv
from stable_baselines3 import SAC, PPO, DDPG
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
import wandb

NUM_EPISODES = 10
NUM_TIME_STEPS = 5000
GRADIENT_SAVE_FREQ = 100

ENV_PARAMETERS = {
    'FORWARD_REWARD_WEIGHT': 1,
    'TRACKING_REWARD_WEIGHT': 0.05,
    'CONTROL_COST_WEIGHT': 0.05,
    'CONTACT_COST_WEIGHT': 5e-4,
    'HEALTHY_REWARD_WEIGHT': 1,
    'MAIN_BODY': 1,
    'HEALTHY_Z_RANGE': (0.195, 0.75),
    'RESET_NOISE_SCALE': 0.1,
    'CONTACT_FORCE_RANGE': (-1.0, 1.0)
}

MUJOCO_PARAMETERS = {
    'frame_skip': 50,
    'observation_space': None,
    'render_mode': 'human'
}


def env_initialization(model_name):
    """
    Args:
        model_name (_type_): _description_

    Returns:
        _type_: _description_
    """

    env = QuadEnv(ENV_PARAMETERS, MUJOCO_PARAMETERS)
    env = Monitor(env)
    run = wandb.init(name=model_name,
                     project="Quad_Mujoco",
                     sync_tensorboard=True,
                     monitor_gym=True,
                     save_code=True)
    return env, run


def training_initialization(algorithm_model):
    """
    

    Args:
        algorithm_model (_type_): _description_

    Returns:
        _type_: _description_
    """
    algorithm_name = algorithm_model.__name__
    env, run = env_initialization(algorithm_name)
    device = "cpu" if algorithm_name == "PPO" else "cuda"
    model = algorithm_model(
        "MlpPolicy",
        env,
        device=device,
        verbose=1,
        tensorboard_log=f"runs/{algorithm_name}"
    )

    vec_env = model.get_env()
    obs = vec_env.reset()

    model.learn(
        total_timesteps=NUM_TIME_STEPS,
        callback=WandbCallback(
            verbose=1,
            gradient_save_freq=GRADIENT_SAVE_FREQ,
            model_save_path=f'models/{algorithm_model.__name__}'
        ))

    return model, vec_env, run, obs


def train(model, vec_env, run, obs):
    for _ in range(NUM_EPISODES):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render('human')


if __name__ == "__main__":
    algorithms_list = [PPO, SAC, DDPG]
    for algorithm in algorithms_list:
        model, vec_env, run, obs = training_initialization(algorithm)
        train(model, vec_env, run, obs)
        vec_env.close()
        run.finish()
