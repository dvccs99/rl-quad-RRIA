from gymnasium.envs.registration import register

register(
    id="QuadEnv",
    entry_point="rl_quad.envs:QuadEnv",
)
