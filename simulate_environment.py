import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

ROBOT_PATH = "rl_quad/envs/robot/anybotics_anymal_c/scene_uneven.xml"
mujoco_model = mujoco.MjModel.from_xml_path(ROBOT_PATH)
data = mujoco.MjData(mujoco_model)


def get_obs() -> np.ndarray:
    """
    Gymnasium observation function. Needs more detail.

    Returns:
        np.ndarray: _description_
    """
    obs = data.qpos[7:].flatten()
    obs = np.array(obs, dtype=np.float32)
    return obs


with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
    while viewer.is_running():
        obs = get_obs()
        action = [0]*12
        with viewer.lock():
            data.ctrl[:] = action
        mujoco.mj_step(mujoco_model, data)
        viewer.sync()
