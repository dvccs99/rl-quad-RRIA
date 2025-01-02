import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC, DDPG
import numpy as np

ppo = PPO.load("models/PPO/v0/model.zip")
# sac = SAC.load("models/SAC/v4/model.zip")
# ddpg = DDPG.load("models/DDPG/v4/model.zip")

algorithm = ppo

xml_path = "robot/anybotics_anymal_c/scene.xml"

mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(mujoco_model)


def get_obs():
    MIN_CONTACT_FORCE = -1.0
    MAX_CONTACT_FORCE = 1.0
    raw_contact_forces = data.cfrc_ext
    contact_forces = np.clip(raw_contact_forces,
                             MIN_CONTACT_FORCE,
                             MAX_CONTACT_FORCE)
    new_contact_forces = contact_forces[1:].flatten()
    obs = np.concatenate([data.qpos[2:], data.qvel, new_contact_forces])
    return obs


with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
    while viewer.is_running():
        obs = get_obs()
        # action, _states = algorithm.predict(obs, deterministic=True)
        action = [0]*12
        with viewer.lock():
            data.ctrl[:] = action
        mujoco.mj_step(mujoco_model, data)
        viewer.sync()
