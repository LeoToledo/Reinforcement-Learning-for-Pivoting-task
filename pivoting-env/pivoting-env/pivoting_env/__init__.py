from gym.envs.registration import register

register(
    id='pivoting-v0',
    entry_point='pivoting_env.envs:PivotingEnv',
)
