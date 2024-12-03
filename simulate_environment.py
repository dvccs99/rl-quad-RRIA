# import mujoco
# import mujoco.viewer
# from stable_baselines3 import SAC

# rl_model = SAC.load("model.zip", print_system_info=True)
# # Path to your MuJoCo XML file
# xml_path = "robot/anybotics_anymal_c/scene.xml"

# # Load the model and create the simulation
# mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
# data = mujoco.MjData(mujoco_model)

# # Launch the viewer
# with mujoco.viewer.launch(mujoco_model, data) as viewer:
#     while viewer.is_running():
#         mujoco.mj_step(mujoco_model, data)  # Step the simulation
#         pred = rl_model.predict(data)
#         print(pred)


import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
import numpy as np

# Load the trained Stable Baselines 3 SAC model
rl_model = SAC.load("model.zip", print_system_info=True)

# Path to your MuJoCo XML file
xml_path = "robot/anybotics_anymal_c/scene.xml"

# Load the MuJoCo model and create the simulation
mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(mujoco_model)

# Launch the viewer and run the simulation
with mujoco.viewer.launch(mujoco_model, data) as viewer:
    while viewer.is_running():
        obs = np.concatenate([data.qpos, data.qvel])
        action, _states = rl_model.predict(obs, deterministic=True)
        data = mujoco.MjData(mujoco_model)
        data.ctrl[:] = action

        mujoco.mj_step(mujoco_model, data)
        print("oi")
