import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC, DDPG
import numpy as np
from scipy.spatial.transform import Rotation as R


# ppo = PPO.load("models/PPO/v0/model.zip")
# sac = SAC.load("models/SAC/v4/model.zip")
# ddpg = DDPG.load("models/DDPG/v4/model.zip")
# algorithm = ppo

ROBOT_PATH = "rl_quad/envs/robot/anybotics_anymal_c/scene.xml"
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


def rotation_matrix_to_rpy(R):
    """
    Converts a rotation matrix to roll, pitch, and yaw angles.

    Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: (roll, pitch, yaw) in radians.
    """
    assert R.shape == (3, 3), "Input must be a 3x3 matrix"
    
    # Check for gimbal lock
    if np.isclose(R[2, 0], -1.0):  # Singularity at pitch = -90 degrees
        pitch = np.pi / 2
        roll = np.arctan2(R[0, 1], R[0, 2])
        yaw = 0
    elif np.isclose(R[2, 0], 1.0):  # Singularity at pitch = +90 degrees
        pitch = -np.pi / 2
        roll = np.arctan2(-R[0, 1], -R[0, 2])
        yaw = 0
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    return roll, pitch, yaw


angle = np.radians(90)
qw = np.cos(angle / 2)  # Scalar part
qx = 0                 # X-axis component
qy = 0                 # Y-axis component
qz = np.sin(angle / 2)  # Z-axis component

with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
    while viewer.is_running():
        obs = get_obs()
        # action, _states = algorithm.predict(obs, deterministic=True)
        action = [0]*12
        # euler_angles = R.from_quat(data.body(1).xquat).as_matrix()
        # pitch = euler_angles[1]

        with viewer.lock():
            data.ctrl[:] = action
            euler_angles = R.from_quat(data.qpos[4:8]).as_euler(seq='xyz')
            print(euler_angles)
        mujoco.mj_step(mujoco_model, data)
        viewer.sync()
