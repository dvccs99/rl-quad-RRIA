import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
import numpy as np

rl_model = SAC.load("model.zip", print_system_info=True)

xml_path = "robot/anybotics_anymal_c/scene.xml"

mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(mujoco_model)

with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
    while viewer.is_running():
        MIN_CONTACT_FORCE = -1.0
        MAX_CONTACT_FORCE = 1.0
        raw_contact_forces = data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces,
                                 MIN_CONTACT_FORCE,
                                 MAX_CONTACT_FORCE)
        new_contact_forces = contact_forces[1:].flatten()
        obs = np.concatenate([data.qpos[2:], data.qvel, new_contact_forces])
        action, _states = rl_model.predict(obs, deterministic=True)
        data = mujoco.MjData(mujoco_model)
        # print(data.cvel)
        with viewer.lock():
            data.ctrl[:] = action
        mujoco.mj_step(mujoco_model, data)
        viewer.sync()
