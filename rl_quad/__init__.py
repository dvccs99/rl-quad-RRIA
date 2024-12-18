from gymnasium.envs.registration import register

register(
    id="rl_quad/quad_env",
    entry_point="rl_quad.envs:QuadEnv",
)
